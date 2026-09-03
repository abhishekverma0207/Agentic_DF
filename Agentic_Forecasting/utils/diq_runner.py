"""
DIQ Forecast Orchestrator — single-notebook entry point.

Drives a FORWARD-FORECAST-ONLY run across all requested categories from
a single Databricks notebook. Supports both UK and DE markets via the
``market`` parameter (UK default; DE uses ``market='de'`` plus the
matching ``cats_subdir='DT'`` and ``category_col='Category'`` widgets).

Expects TWO Parquet inputs directly under ``{PARENT_DIR}/``:

    * ``{cats_subdir}/`` — driver/cats frame.
        UK default: ``01_all_keys_dr_adjusted``
        DE        : ``DT``
    * ``{fs_subdir}/``   — feature-store columns. Shared across markets,
        default ``FS``.

which are inner-joined on ``['key', 'year_week']`` (FS contributes only
columns NOT already in cats and NOT ending with ``_dep``). The joined
frame is then used as the unified input — no other preprocessing is
done. This module ONLY:

    1. Reads + joins those two tables.
    2. For each category, filters the joined frame to that category's rows.
    3. Writes the sub-set to a snapshot-scoped folder as CSV.
    4. Builds a runtime config (template YAML + runtime overrides) that
       points at the new CSV and sets time splits from history_till /
       snapshot_week. The template is picked from
       ``config/config_{market}_{slug}_databricks.yaml``.
    5. Runs the existing deterministic pipeline (EDA → segmentation →
       features → training → inference) with backtest SKIPPED via
       ``run_mode='forecast_only'``.
    6. Stacks every category's ``inference_forecast.csv``, applies MOQ
       post-processing across the whole stacked frame, writes the
       final output to ``{PARENT_DIR}/DIQ/TF_DIQ/``.

**No other transformation is done on the joined data.** The indicator-
collapse and calendar-feature-injection that the legacy UK
preprocessing did are deliberately NOT repeated here — the upstream
delivers data that is ready for the pipeline as-is.

Directory layout after one run at snapshot=202615 (UK shown; for DE the
top-level input folder is ``DT/`` instead of ``01_all_keys_dr_adjusted/``):

    {PARENT_DIR}/
        01_all_keys_dr_adjusted/                 # caller-provided input (UK)
        FS/                                      # caller-provided input
        DIQ/
            202615/                              # snapshot_week
                COOKING_AID/
                    sourcedata/COOKING_AID.csv
                    artifacts/
                        eda_output/
                        seg_output/
                        feature_output/
                        model_artifacts/
                            inference_forecast.csv   # per-category output
                        ...
                CONDIMENT/  ...
                (one folder per category in category_list)
            TF_DIQ/
                202615/                          # snapshot_week
                    inference_forecast.parquet   # STACKED + MOQ-processed

Usage from a Databricks notebook:

    from utils.diq_runner import run_diq_forecast
    run_diq_forecast(
        parent_dir=dbutils.widgets.get("PARENT_DIR"),
        history_till=dbutils.widgets.get("history_till"),
        snapshot_week=dbutils.widgets.get("snapshot_week"),
        category_list=dbutils.widgets.get("category_list").split('#'),
        market=dbutils.widgets.get("MARKET"),          # "uk" | "de"
        cats_subdir=dbutils.widgets.get("CATS_SUBDIR"),  # UK: 01_all_keys_dr_adjusted, DE: DT
        category_col=dbutils.widgets.get("category_col"),  # UK: category_name, DE: Category
    )
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# CRITICAL: kill EVERY auto-instrumentation hook at MODULE IMPORT time.
#
# Setting these later (inside the run_diq_forecast function body) is too
# late: by then statsmodels has often already been imported by EDA, autolog
# has registered, OpenTelemetry has spun up exporters, and every subsequent
# OLS / STL / GrangerCausality fit creates an MLflow Run + a TraceSpan.
#
# This block matches what `databricks_runner*.py` already do at the very
# top - we mirror it here because diq_runner is a separate notebook entry
# point.
#
# Override individual disables by setting the corresponding env var to "0"
# BEFORE importing this module.
# -----------------------------------------------------------------------------
import os as _early_os
if _early_os.environ.get("DIQ_RUNNER_KEEP_MLFLOW_AUTOLOG", "").lower() not in (
    "1", "true", "yes",
):
    # MLflow autologging — Databricks auto-creates a Run per OLS / STL /
    # lgb.train(), with artifact uploads + the "🏃 View run ..." notebook
    # spam.  Per-fit overhead is ~3-6 seconds; with 100s of fits per
    # backtest origin this single setting causes 100× slowdowns.
    _early_os.environ.setdefault("MLFLOW_AUTOLOGGING_DISABLE", "1")
    _early_os.environ.setdefault("MLFLOW_DATABRICKS_AUTOLOG_DISABLED", "true")
    _early_os.environ.setdefault("DISABLE_MLFLOW_INTEGRATION", "TRUE")
    # OpenTelemetry — CrewAI auto-enables an OTel exporter that traces
    # every Agent/Task call.  Even without LLM calls (e.g. our crews
    # imports for module side-effects) it adds a span per Python call
    # frame.  databricks_runner*.py already disable it; mirror here.
    _early_os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    _early_os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
    _early_os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
    _early_os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")
    # CrewAI's anonymous telemetry pings home on every Agent run.  Two
    # env-var spellings are honoured by different CrewAI versions.
    _early_os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
    _early_os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    _early_os.environ.setdefault("CREWAI_DO_NOT_TRACK", "1")
    # LiteLLM telemetry — sends usage stats per call.  Doesn't fire in
    # --no-llm but harmless to disable.
    _early_os.environ.setdefault("LITELLM_TELEMETRY", "False")
    # HuggingFace tokenizers — chatty when forking from a parent that
    # already has tokenizers loaded.  Setting this silences the warning
    # AND dodges a multiprocessing slowdown.
    _early_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Posthog (CrewAI uses it for opt-in analytics) - disable.
    _early_os.environ.setdefault("POSTHOG_DISABLED", "true")
del _early_os

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category token → YAML-template mapping
# ---------------------------------------------------------------------------
# MLflow autologging silencer.
#
# Databricks notebooks turn `mlflow.autolog()` ON by default.  Every single
# `lgb.train()` we run inside DMH then creates a fresh MLflow Run, complete
# with parameter logging, metric streaming, and artifact uploads.  For a
# DMH backtest with H horizons × n_seeds × n_ensemble_heads × n_origins
# fits, that's hundreds-to-thousands of MLflow runs - each with its own
# network round-trip - and the user sees an avalanche of
# `🏃 View run burly-ant-45 at: ...` messages while runtime balloons by
# orders of magnitude.
#
# We disable autologging globally before any model training begins.  Set
# DIQ_RUNNER_KEEP_MLFLOW_AUTOLOG=1 to opt back in (for debugging individual
# fits when you really want every run logged).
# ---------------------------------------------------------------------------


def _disable_mlflow_autolog_unless_requested() -> None:
    """Turn MLflow autologging OFF.  Safe to call multiple times.

    Covers EVERY MLflow flavor that Databricks autologs on its own,
    including the one that surfaced as "OLS Regression Results" runs in
    the user's MLflow UI: ``mlflow.statsmodels.autolog()`` - fired by
    every STL fit, seasonal_decompose, grangercausalitytests, etc. that
    EDA runs.

    Also sweeps existing autolog hooks from `AUTOLOGGING_INTEGRATIONS`
    so any flavor that was already registered before this function ran
    gets disabled too.
    """
    import os as _os
    if _os.environ.get("DIQ_RUNNER_KEEP_MLFLOW_AUTOLOG", "").lower() in (
        "1", "true", "yes",
    ):
        logger.info(
            "DIQ_RUNNER_KEEP_MLFLOW_AUTOLOG set; leaving MLflow autologging ON",
        )
        return

    # Defensive env vars (already set at module import; reaffirm here in
    # case someone unset them between import and call).
    _os.environ["MLFLOW_AUTOLOGGING_DISABLE"] = "1"
    _os.environ["MLFLOW_DATABRICKS_AUTOLOG_DISABLED"] = "true"
    _os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"

    try:
        import mlflow
        # Global disable - covers all integrations registered through the
        # standard MLflow autolog framework.
        try:
            mlflow.autolog(disable=True, silent=True)
        except Exception:
            pass

        # Per-flavor disable - belt-and-braces because Databricks's own
        # autolog hook may register flavors AFTER `mlflow.autolog` runs.
        # 'statsmodels' is the one that produced the OLS Regression
        # artifacts in EDA (STL, seasonal_decompose, granger).
        _flavors = (
            "lightgbm", "xgboost", "sklearn", "pytorch", "catboost",
            "statsmodels", "tensorflow", "spark", "transformers",
            "fastai", "gluon", "h2o", "keras", "paddle", "pyspark",
            "shap", "pycallback",
        )
        for _flavor in _flavors:
            try:
                _flavor_mod = getattr(mlflow, _flavor, None)
                if _flavor_mod is not None:
                    _flavor_mod.autolog(disable=True, silent=True)
            except Exception:
                pass

        # Sweep the AUTOLOGGING_INTEGRATIONS dict so flavors that
        # registered themselves between our import and this call are
        # marked disabled too.
        try:
            from mlflow.utils.autologging_utils import (
                AUTOLOGGING_INTEGRATIONS,
            )
            for _intg, _cfg in list(AUTOLOGGING_INTEGRATIONS.items()):
                if isinstance(_cfg, dict):
                    _cfg["disable"] = True
        except Exception:
            pass

        # Silence the "🏃 View run ..." prints + autolog WARN spam.
        try:
            import logging as _logging
            _logging.getLogger("mlflow").setLevel(_logging.WARNING)
            _logging.getLogger("mlflow.tracking._tracking_service.client").setLevel(
                _logging.ERROR
            )
            _logging.getLogger("mlflow.utils.autologging_utils").setLevel(
                _logging.ERROR
            )
            _logging.getLogger("mlflow.tracking.fluent").setLevel(
                _logging.ERROR
            )
        except Exception:
            pass

        logger.info(
            "MLflow autologging disabled across %d flavors "
            "(DIQ_RUNNER_KEEP_MLFLOW_AUTOLOG=1 to opt in)",
            len(_flavors),
        )
    except ImportError:
        # No MLflow installed → nothing to do.
        pass
    except Exception as _exc:
        logger.warning("Could not disable MLflow autologging: %s", _exc)


# Run the disabler once at module-import time so ANY downstream import
# (statsmodels, sklearn, lightgbm, ...) that happens before
# run_diq_forecast() is called still has autolog off.
try:
    _disable_mlflow_autolog_unless_requested()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Dependency installer — diq_runner is a notebook entry point that bypasses
# databricks_runner.py's install_dependencies() step, so on a fresh
# Databricks job cluster it would hit `ModuleNotFoundError: No module named
# 'crewai'` (and pydantic v2, lightgbm, ...).  We mirror the same install
# logic here so both entry points behave identically.
# ---------------------------------------------------------------------------


def _ensure_dependencies(project_root: str, force: bool = False) -> bool:
    """Install requirements.txt into the active interpreter if crewai or
    pydantic v2 is missing.  Idempotent — exits fast when both are already
    importable.  Mirrors `databricks_runner.install_dependencies` so the
    DIQ notebook entry point handles its own bootstrap.

    Parameters
    ----------
    project_root : str
        Repo root containing requirements.txt.
    force : bool
        Reinstall even if dependencies look present.  Set when the
        DIQ_RUNNER_FORCE_INSTALL env var is "1".

    Returns
    -------
    bool
        True on success (or skip), False on irrecoverable error.
    """
    import importlib
    import os as _os
    import subprocess
    import sys as _sys

    # ----------------------------------------------------------------
    # Smart default behaviour - install ONLY when actually needed.
    #
    # Three states the cluster can be in:
    #
    #  (a) `crewai` + pydantic v2 already importable
    #      => Healthy.  Skip install.  Fast path.
    #      => Used when requirements.txt is linked at cluster level
    #         (UK/TH production clusters).
    #
    #  (b) Probe fails AND user has set DIQ_RUNNER_SKIP_INSTALL=1
    #      => User explicitly opted out.  Log a warning, continue;
    #         downstream imports will fail loudly with a clear stack.
    #
    #  (c) Probe fails AND no opt-out env var set
    #      => Auto-install via the 4-strategy fallback chain below.
    #         This is what unblocks fresh / lego clusters that don't
    #         link requirements.txt at the cluster level - we install
    #         crewai + pydantic v2 + lightgbm on demand.
    #
    # Opt-in to FORCE re-install on every run (debug only):
    #   DIQ_RUNNER_FORCE_INSTALL=1
    # ----------------------------------------------------------------
    _force_env = _os.environ.get("DIQ_RUNNER_FORCE_INSTALL", "").lower() in (
        "1", "true", "yes",
    )
    _skip_env = _os.environ.get("DIQ_RUNNER_SKIP_INSTALL", "").lower() in (
        "1", "true", "yes",
    )
    force = bool(force or _force_env)

    # Probe imports so the log gives a clear health snapshot regardless
    # of which path we take next.
    _probe_ok = False
    _probe_msg = ""
    try:
        importlib.invalidate_caches()
        import crewai  # noqa: F401
        import pydantic
        _probe_ok = str(getattr(pydantic, "VERSION", "1")).startswith("2")
        _probe_msg = (
            f"crewai={getattr(crewai, '__version__', '?')}, "
            f"pydantic={pydantic.VERSION}"
            + ("" if _probe_ok else " (still v1)")
        )
    except ImportError as _e:
        _probe_msg = f"ImportError({_e})"
    except Exception as _e:
        _probe_msg = f"probe error: {_e}"

    # Case (a): everything's already there, fast path.
    if _probe_ok and not force:
        logger.info(
            "DIQ deps OK from cluster libraries: %s; skipping install bootstrap.",
            _probe_msg,
        )
        return True

    # Case (b): probe fails but user explicitly skipped.  Don't install
    # but warn loudly so the failure ten seconds later isn't a mystery.
    if _skip_env and not force:
        logger.warning(
            "DIQ_RUNNER_SKIP_INSTALL set but probe shows issues (%s). "
            "Skipping install as requested - downstream imports WILL "
            "FAIL (e.g. ModuleNotFoundError: No module named 'crewai'). "
            "Unset the env var or pass --force-install to fix.",
            _probe_msg,
        )
        return True

    # Case (c) / force: install path falls through.  Tell the user what
    # we're about to do.
    if force:
        logger.info(
            "DIQ_RUNNER_FORCE_INSTALL set - running install bootstrap (probe: %s)",
            _probe_msg,
        )
    else:
        logger.info(
            "DIQ deps probe failed (%s) - auto-installing crewai + pydantic v2 "
            "+ lightgbm via pip.  Set DIQ_RUNNER_SKIP_INSTALL=1 to opt out.",
            _probe_msg,
        )

    requirements_path = _os.path.join(project_root, "requirements.txt")
    if not _os.path.exists(requirements_path):
        logger.error("requirements.txt not found at: %s", requirements_path)
        return False

    logger.info("=" * 60)
    logger.info("DIQ runner: installing Python dependencies")
    logger.info("  requirements: %s", requirements_path)
    logger.info("  python:       %s", _sys.executable)
    logger.info("  (this typically takes 5-10 minutes on a fresh cluster)")
    logger.info("=" * 60)

    def _run_pip(args, label):
        """Run a pip subcommand and surface stdout+stderr to the logger.

        We deliberately do NOT use --quiet so the user actually sees what
        is happening on Databricks.  When we can't see pip's output, a
        compile failure (cmake / xgboost / pyarrow / ...) looks identical
        to a clean success that didn't actually install anything, and the
        downstream `import crewai` failure leaves the user stuck.
        """
        cmd = [_sys.executable, "-m", "pip", "install", *args]
        logger.info("[pip][%s] %s", label, " ".join(cmd))
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired as _exc:
            logger.error("[pip][%s] TIMEOUT after 30 minutes", label)
            return False, "TIMEOUT"
        except Exception as _exc:
            logger.error("[pip][%s] subprocess error: %s", label, _exc)
            return False, str(_exc)
        # Always log a tail of stdout + stderr so the user can see what
        # actually happened.  pip prints package downloads to stdout and
        # build errors to stderr.
        if res.stdout:
            logger.info("[pip][%s] stdout (last 2000 chars):\n%s",
                         label, res.stdout[-2000:])
        if res.stderr:
            level = logger.error if res.returncode != 0 else logger.warning
            level("[pip][%s] stderr (last 2000 chars):\n%s",
                   label, res.stderr[-2000:])
        if res.returncode == 0:
            logger.info("[pip][%s] return code 0 ✓", label)
            return True, ""
        logger.error("[pip][%s] return code %s ✗", label, res.returncode)
        return False, (res.stderr or res.stdout or "(no output)")[-2000:]

    def _import_check():
        """Return (ok: bool, msg: str) for crewai + pydantic v2 import."""
        for m in list(_sys.modules.keys()):
            if any(s in m for s in ("crewai", "litellm", "opentelemetry", "pydantic")):
                _sys.modules.pop(m, None)
        importlib.invalidate_caches()
        try:
            import crewai  # noqa: F401
            import pydantic
            ok = str(getattr(pydantic, "VERSION", "1")).startswith("2")
            return ok, (
                f"crewai={getattr(crewai, '__version__', '?')}, "
                f"pydantic={pydantic.VERSION}"
                + ("" if ok else " (still v1!)")
            )
        except Exception as _e:
            return False, str(_e)

    # Strategy A: standard install from requirements.txt (matches
    # databricks_runner.install_dependencies).  Most common case.
    okA, errA = _run_pip(
        ["-r", requirements_path, "--upgrade", "--no-warn-script-location"],
        "A:requirements",
    )
    okI, info = _import_check()
    if okI:
        logger.info("Post-install verification OK: %s", info)
        return True

    # Strategy B: --user install.  Some Databricks Runtimes have a
    # read-only /databricks/python/lib site-packages and require user-site
    # installs to land somewhere importable.
    logger.info(
        "Strategy A did not fix the import (%s); retrying with --user install",
        info,
    )
    okB, errB = _run_pip(
        ["-r", requirements_path, "--upgrade", "--user", "--no-warn-script-location"],
        "B:--user",
    )
    okI, info = _import_check()
    if okI:
        logger.info("Post-install verification OK after --user: %s", info)
        return True

    # Strategy C: install JUST the critical packages explicitly.  If the
    # full requirements.txt has one bad package that aborts the whole
    # install, we still want crewai + pydantic v2 to land so the rest of
    # the pipeline can run.
    logger.info(
        "Strategy B did not fix the import (%s); retrying minimal-set install "
        "(crewai + pydantic + lightgbm)",
        info,
    )
    okC, errC = _run_pip(
        ["crewai", "pydantic>=2.0", "lightgbm", "--upgrade",
         "--no-warn-script-location"],
        "C:minimal",
    )
    okI, info = _import_check()
    if okI:
        logger.info("Post-install verification OK after minimal: %s", info)
        return True

    # Strategy D: same minimal set but --user.
    logger.info(
        "Strategy C did not fix the import (%s); retrying minimal --user",
        info,
    )
    okD, errD = _run_pip(
        ["crewai", "pydantic>=2.0", "lightgbm", "--upgrade", "--user",
         "--no-warn-script-location"],
        "D:minimal-user",
    )
    okI, info = _import_check()
    if okI:
        logger.info("Post-install verification OK after minimal --user: %s", info)
        return True

    # All four strategies failed - log a clear summary so the user can
    # see exactly which install commands ran and what each said.
    logger.error("=" * 60)
    logger.error("DIQ install failed across all four strategies.")
    logger.error("Final post-install import probe: %s", info)
    logger.error("Strategy A (requirements):       ok=%s", okA)
    logger.error("Strategy B (requirements --user): ok=%s", okB)
    logger.error("Strategy C (minimal):            ok=%s", okC)
    logger.error("Strategy D (minimal --user):     ok=%s", okD)
    logger.error(
        "Workaround: run this in a notebook cell BEFORE importing diq_runner:\n"
        "    %%pip install crewai pydantic>=2 lightgbm --quiet\n"
        "    dbutils.library.restartPython()\n"
        "Or set DIQ_RUNNER_SKIP_INSTALL=1 in the cluster env if your cluster "
        "is already pre-provisioned."
    )
    logger.error("=" * 60)
    return False


# ---------------------------------------------------------------------------
# Category name -> config YAML slug.  The widget can supply the category
# in many forms:
#
#   "DEODORANT_FRAGRANCE"      (snake)
#   "DEODORANT & FRAGRANCE"    (display, with " & ")
#   "DEODORANT FRAGRANCE"      (display, just space)
#   "Deodorants & Fragrances"  (mixed case + plurals)
#
# The matcher is therefore staged:
#   1. Normalise the input: uppercase, strip "&", collapse non-word chars
#      to single underscore, strip leading/trailing underscores.
#   2. Look up the normalised string in the market-specific override map
#      (`_UK_YAML_SLUG_OVERRIDES` / `_DE_YAML_SLUG_OVERRIDES`).
#   3. If still unmatched, fall back to lowercased normalised string.
# ---------------------------------------------------------------------------

import re as _re

_UK_YAML_SLUG_OVERRIDES: Dict[str, str] = {
    # Deodorants & Fragrances
    "DEODORANT_FRAGRANCE":     "deo_fragrance",
    "DEODORANT_FRAGRANCES":    "deo_fragrance",
    "DEODORANTS_FRAGRANCE":    "deo_fragrance",
    "DEODORANTS_FRAGRANCES":   "deo_fragrance",
    "DEO_FRAGRANCE":           "deo_fragrance",
    # Cooking Aids
    "COOKING_AID":             "cooking_aid",
    "COOKING_AIDS":            "cooking_aid",
    "SCRATCH_COOKING_AIDS":    "cooking_aid",
    # Condiments / Dressings
    "CONDIMENT":               "condiment",
    "CONDIMENTS":              "condiment",
    "DRESSING":                "condiment",
    "DRESSINGS":               "condiment",
    # Fabric
    "FABRIC_CLEANING":         "fabric_cleaning",
    "FABRIC_CLEANINGS":        "fabric_cleaning",
    "FABRIC_ENHANCER":         "fabric_enhancer",
    "FABRIC_ENHANCERS":        "fabric_enhancer",
    # Hair / Skin
    "HAIR_CARE":               "hair_care",
    "HAIRCARE":                "hair_care",
    "SKIN_CARE":               "skin_care",
    "SKINCARE":                "skin_care",
    "SKIN_CLEANSING":          "skin_cleansing",
    "SKINCLEANSING":           "skin_cleansing",
    # Home Hygiene
    "HOME_HYGIENE":            "home_hygiene",
    "HOME_HYGIENES":           "home_hygiene",
    "HOMECARE":                "home_hygiene",
    # Mini Meals / Healthy snacking
    "MINI_MEAL":               "mini_meal",
    "MINI_MEALS":              "mini_meal",
    "MINIMEAL":                "mini_meal",
    "MINIMEALS":               "mini_meal",
    "HEALTHY_SNACKING":        "mini_meal",
    "HEALTHY_SNACK":           "mini_meal",
    # Other foods
    "OTH_FOOD":                "oth_food",
    "OTHER_FOOD":              "oth_food",
    "OTHER_FOODS":             "oth_food",
}

# DE slug map — DE category folders follow different naming than UK
# (e.g. UK 'cooking_aid' vs DE 'cooking_aids', UK 'fabric_cleaning' vs
# DE 'fab_clean'). Mirrors the file names already present under
# config/config_de_*_databricks.yaml.
_DE_YAML_SLUG_OVERRIDES: Dict[str, str] = {
    # Cooking Aids
    "COOKING_AID":             "cooking_aids",
    "COOKING_AIDS":            "cooking_aids",
    "SCRATCH_COOKING_AIDS":    "cooking_aids",
    # Deodorants & Fragrances → DE config is named just 'deo'
    "DEODORANT_FRAGRANCE":     "deo",
    "DEODORANT_FRAGRANCES":    "deo",
    "DEODORANTS_FRAGRANCE":    "deo",
    "DEODORANTS_FRAGRANCES":   "deo",
    "DEO_FRAGRANCE":           "deo",
    "DEO":                     "deo",
    # Dressings (no UK ↔ DE 'condiment' equivalence — DE uses dressings)
    "CONDIMENT":               "dressings",
    "CONDIMENTS":              "dressings",
    "DRESSING":                "dressings",
    "DRESSINGS":               "dressings",
    # Fabric Cleaning → DE shortens to fab_clean
    "FABRIC_CLEANING":         "fab_clean",
    "FABRIC_CLEANINGS":        "fab_clean",
    "FAB_CLEAN":               "fab_clean",
    # Hair / Skin — DE drops the underscore
    "HAIR_CARE":               "haircare",
    "HAIRCARE":                "haircare",
    "SKIN_CARE":               "skincare",
    "SKINCARE":                "skincare",
    "SKIN_CLEANSING":          "skincleansing",
    "SKINCLEANSING":           "skincleansing",
    # Home Hygiene
    "HOME_HYGIENE":            "home_hygiene",
    "HOME_HYGIENES":           "home_hygiene",
    "HOMECARE":                "home_hygiene",
    # Healthy Snacking (DE equivalent of UK MINI_MEAL)
    "HEALTHY_SNACK":           "healthy_snack",
    "HEALTHY_SNACKING":        "healthy_snack",
    "MINI_MEAL":               "healthy_snack",
    "MINI_MEALS":              "healthy_snack",
    # Oral care (DE-only)
    "ORAL_CARE":               "oral_care",
    "ORALCARE":                "oral_care",
}

_SLUG_OVERRIDES_BY_MARKET: Dict[str, Dict[str, str]] = {
    "uk": _UK_YAML_SLUG_OVERRIDES,
    "de": _DE_YAML_SLUG_OVERRIDES,
}


def _normalise_category_token(category: str) -> str:
    """Uppercase + collapse '&' / whitespace / hyphens / dots to '_'."""
    s = str(category).upper().replace("&", "_")
    s = _re.sub(r"[^A-Z0-9]+", "_", s)
    return s.strip("_")


def _category_to_yaml_slug(category: str, market: str = "uk") -> str:
    """Map a category name (any form) to the YAML slug used in
    config/config_{market}_{slug}_databricks.yaml."""
    norm = _normalise_category_token(category)
    overrides = _SLUG_OVERRIDES_BY_MARKET.get(market.lower(), _UK_YAML_SLUG_OVERRIDES)
    if norm in overrides:
        return overrides[norm]
    # Fallback: lowercased normalised token (e.g. CONDIMENT -> condiment).
    return norm.lower()


def _yaml_template_for(
    category: str,
    project_root: str,
    market: str = "uk",
    config_root: Optional[str] = None,
) -> Path:
    """Resolve the per-category template YAML path.

    Parameters
    ----------
    category
        Category name (any form — gets normalised + slug-mapped).
    project_root
        Repo root (used for the in-repo fallback ``<project_root>/config``).
    market
        ``"uk"`` or ``"de"`` — picks the slug-override map and the
        ``config_{market}_*`` filename prefix.
    config_root
        Optional override directory containing the YAML templates. When
        provided (non-blank) the lookup uses ``<config_root>/config_{market}_{slug}_databricks.yaml``
        directly — this lets configs live on a UC Volume (e.g.
        ``/Volumes/pds_feu_931272_dev/feu_common/config/EU_UK/DIQ_configs/``)
        instead of being shipped with the repo. When ``None`` or blank
        the historic behaviour is preserved (``<project_root>/config/``).
    """
    market_lc = market.lower()
    slug = _category_to_yaml_slug(category, market=market_lc)
    # External config root takes precedence when supplied; otherwise fall
    # back to the in-repo location. ``Path`` handles ``/Volumes/...`` UC
    # paths natively on Databricks runtimes.
    if config_root and str(config_root).strip():
        config_dir = Path(str(config_root).strip())
    else:
        config_dir = Path(project_root) / "config"
    path = config_dir / f"config_{market_lc}_{slug}_databricks.yaml"
    if not path.exists():
        # Provide a helpful error listing all available config slugs for
        # this market so the user can quickly add an alias to the
        # market-specific slug-override map.
        prefix = f"config_{market_lc}_"
        try:
            available = sorted(
                p.name.replace(prefix, "").replace("_databricks.yaml", "")
                for p in config_dir.glob(f"{prefix}*_databricks.yaml")
            )
        except Exception:
            available = []
        raise FileNotFoundError(
            f"Template YAML not found for category '{category}' "
            f"(market='{market_lc}', normalised='{_normalise_category_token(category)}', "
            f"slug='{slug}', expected {path}). "
            f"Looked in config_dir={config_dir}. "
            f"Available {market_lc.upper()} databricks slugs: {available}. "
            f"Either pass one of those names, extend "
            f"_{market_lc.upper()}_YAML_SLUG_OVERRIDES in utils.diq_runner, "
            f"or point ``config_root`` at a directory that contains the YAML."
        )
    return path


# ---------------------------------------------------------------------------
# Input table loading — read FS + 01_all_keys_dr_adjusted, join on key+week
# ---------------------------------------------------------------------------

# Default subdirectory names for the two Parquet inputs. UK runs use
# the historic '01_all_keys_dr_adjusted' folder; DE runs pass
# cats_subdir='DT' through run_diq_forecast(). Both markets share the
# 'FS/' sibling.
_DEFAULT_CATS_SUBDIR = "01_all_keys_dr_adjusted"
_DEFAULT_FS_SUBDIR = "FS"
_JOIN_COLS = ["key", "year_week"]


def _normalise_category_token(s: str) -> str:
    """Same normalisation _filter_and_save_category uses, hoisted here so
    the Spark filter pushdown can use it."""
    import re
    s = str(s).upper().replace("&", " ")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# Column names that historically have date / timestamp semantics in DIQ
# data and MUST NOT be silently converted to Categorical (which would
# break inequality comparisons like `df[col] <= cfg.train_end`).
_TIME_LIKE_COLS = frozenset({
    "year_week", "year_month", "year_quarter", "year",
    "period_start_date", "period", "date", "timestamp",
    "datetime", "calendar_date",
})


def _optimize_dtypes_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Aggressively downcast numeric columns and convert low-cardinality
    string columns to category dtype.  Lossless — only narrows when the
    full value range fits the smaller type.

    Typical wins on a wide DIQ frame:
      * int64 -> int8/16/32 where possible (8x / 4x / 2x smaller)
      * float64 -> float32 (2x smaller, model code accepts both)
      * object string columns with few uniques -> category (10-100x
        smaller for things like category_name, region_id)

    This is safe to run on the full joined frame *before* feature
    engineering because feature engineering reads numerics into numpy
    arrays, where the downcast is invisible.

    **Critical safety rules** (added after we observed
    "Unordered Categoricals can only compare equality or not" at
    deterministic_pipeline.py:154 `source_df[DATE_COL] <= cfg.train_end`):

      1. Columns whose name matches a known time/date semantic
         (``year_week``, ``year_month``, ``period_start_date``, ...)
         are NEVER categorised.  Pandas Categorical is unordered by
         default, which breaks the `<=` / `>=` comparisons the
         downstream pipeline relies on.

      2. Object columns whose first non-null value is a digit-only
         string (``'202401'``) or contains a hyphen / slash
         (``'2024-01'``, ``'2024/01/01'``) are NEVER categorised
         either — these are heuristic indicators of date-likeness
         that a column-name-based filter would miss.

      3. Truly categorical low-cardinality strings (region, category_name,
         segment_id) ARE still safely categorised.  These are typically
         used for groupby/equality filters, never inequality, so the
         dtype change is invisible to the pipeline.
    """
    if df is None or df.empty:
        return df
    import numpy as _np
    n_rows = len(df)
    for col in df.columns:
        try:
            s = df[col]
            kind = s.dtype.kind
            if kind == "i":
                # signed int — downcast to the smallest integer dtype
                df[col] = pd.to_numeric(s, downcast="integer")
            elif kind == "u":
                df[col] = pd.to_numeric(s, downcast="unsigned")
            elif kind == "f":
                # float64 -> float32 is enough precision for ML; we never
                # downcast further than that because float16 silently
                # underflows on multiplication chains in feature engineering.
                if s.dtype == _np.float64:
                    df[col] = s.astype(_np.float32)
            elif kind == "O":
                # Skip rule 1: known time-like column name.
                if col.lower() in _TIME_LIKE_COLS:
                    continue
                # Skip rule 2: value-based heuristic — peek at the first
                # non-null and bail if it looks date-like.  This catches
                # custom column names the rule-1 set doesn't know about.
                non_null = s.dropna()
                if len(non_null) > 0:
                    sample = non_null.iloc[0]
                    if isinstance(sample, str):
                        s_clean = sample.strip()
                        # Digit-only ('202401') -> probably year_week / period
                        if s_clean.isdigit():
                            continue
                        # Hyphen / slash ('2024-01', '2024/01/01') -> date-like
                        if "-" in s_clean or "/" in s_clean:
                            continue
                # Truly categorical strings (region='UK', category_name='DEO').
                # Convert to category iff cardinality is low enough to win
                # on memory.  Pandas equality semantics on Categorical
                # match object/string, so groupby/filter still work.
                try:
                    n_unique = s.nunique(dropna=True)
                    if n_rows > 0 and n_unique / n_rows < 0.5:
                        df[col] = s.astype("category")
                except Exception:
                    pass
        except Exception:
            # Any column that resists optimisation stays as-is — we
            # never want a dtype hint to break the actual pipeline.
            continue
    return df


def _cast_spark_decimals_to_double(sdf):
    """Cast every ``DecimalType`` column in a Spark DataFrame to ``double``.

    Why this is critical (not just a perf nicety):

      * Spark's ``toPandas()`` materialises ``DecimalType`` columns as
        ``decimal.Decimal`` objects living inside an object-dtype pandas
        Series.  pandas already warns about this
        (``UserWarning: The conversion of DecimalType columns is
        inefficient...``).
      * Downstream, ``_optimize_dtypes_inplace`` looks at object-dtype
        columns with low cardinality and converts them to ``category``.
      * In DMH, ``_select_feature_columns`` then classifies them as
        categorical features, and ``_prepare_x`` re-categorises.
      * LightGBM's ``model_to_string()`` calls
        ``json.dumps(pandas_categorical, default=_json_default_with_numpy)``,
        and that helper returns *non-numpy* objects unchanged - so
        ``json.dumps`` re-encodes the same Decimal id forever, Python's
        json marker spots the repeat, and the call dies with
        ``ValueError: Circular reference detected``.  DMH then falls back
        to recursive forecasting, costing ~80 minutes of wall-clock per
        run.

    Casting Decimal to double here, on the Spark side, kills both birds:
    the warning goes away, and the columns arrive in pandas as
    ``np.float64`` (JSON-native + numeric, so they take the numeric path
    in ``_select_feature_columns`` and never become categoricals).
    """
    try:
        from pyspark.sql.types import DecimalType
        from pyspark.sql import functions as F
    except ImportError:
        return sdf
    decimal_cols = [
        f.name for f in sdf.schema.fields
        if isinstance(f.dataType, DecimalType)
    ]
    if not decimal_cols:
        return sdf
    for c in decimal_cols:
        sdf = sdf.withColumn(c, F.col(c).cast("double"))
    logger.info(
        "[Spark] cast %d DecimalType column(s) to double: %s",
        len(decimal_cols), decimal_cols,
    )
    return sdf


def _load_and_join_inputs(
    parent_dir: Path,
    category_list: Optional[List[str]] = None,
    category_col: str = "category_name",
    cats_subdir: str = _DEFAULT_CATS_SUBDIR,
    fs_subdir: str = _DEFAULT_FS_SUBDIR,
) -> pd.DataFrame:
    """Read the two required Parquet inputs from ``parent_dir`` and
    inner-join them on ``['key', 'year_week']``.

    Required folders directly under ``parent_dir`` (names overridable):
      * ``{cats_subdir}/`` — driver/cats frame. Defaults to
        ``01_all_keys_dr_adjusted`` (UK). DE runs pass ``"DT"``.
      * ``{fs_subdir}/``   — feature-store columns to enrich with.
        Defaults to ``FS`` (shared by UK and DE).

    FS contributes only columns NOT already in cats AND NOT ending with
    ``_dep`` (legacy duplicates).

    **Memory-critical optimisation**: when ``category_list`` is provided,
    the category filter is pushed down into Spark (where data is
    distributed) BEFORE the .toPandas() that brings everything onto the
    driver.  Without this, a multi-category Parquet (~11 categories x
    millions of rows each) loaded entirely onto a 64 GB driver causes
    swap-thrash / OOM kernel death.  The filter uses the same
    normalisation _filter_and_save_category uses
    ("DEODORANT & FRAGRANCE" -> "DEODORANT_FRAGRANCE") so any of the
    common spelling variants resolve to the same Spark predicate.
    """
    cats_path = parent_dir / cats_subdir
    fs_path = parent_dir / fs_subdir

    for p, label in [(cats_path, cats_subdir), (fs_path, fs_subdir)]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required input folder '{label}' not found at {p}. "
                f"Both '{fs_subdir}/' and '{cats_subdir}/' must exist "
                f"directly under parent_dir."
            )

    # Pre-compute the normalised tokens once so we can apply the same
    # logic on both Spark and pandas paths.
    target_tokens = (
        [_normalise_category_token(c) for c in category_list if c]
        if category_list else None
    )
    raw_targets = [str(c) for c in category_list] if category_list else None

    try:
        from pyspark.sql import SparkSession, functions as F
        spark = SparkSession.builder.getOrCreate()

        # Enable Arrow for toPandas — 5-30x faster than the legacy path,
        # which is critical when collecting millions of rows to the driver.
        # `fallback.enabled=true` means columns Arrow can't serialize
        # (rare nested types) silently fall back to the slow path instead
        # of failing the whole call.
        try:
            spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
            spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        except Exception:
            pass

        # A path is a Delta table when it carries a _delta_log/ log; FR's DT
        # panel is Delta while FS may still be plain parquet, so detect per path.
        def _fmt(p):
            return "delta" if (Path(p) / "_delta_log").exists() else "parquet"
        cats = spark.read.format(_fmt(cats_path)).load(str(cats_path))
        fs = spark.read.format(_fmt(fs_path)).load(str(fs_path))
        # Cast DecimalType -> double BEFORE any further work.  See
        # _cast_spark_decimals_to_double for the full rationale; the
        # short version is that Decimal-valued columns survive Spark's
        # toPandas() as decimal.Decimal objects, then end up as
        # categorical features whose categories LightGBM cannot JSON-
        # serialise ("Circular reference detected" inside
        # _dump_pandas_categorical).
        cats = _cast_spark_decimals_to_double(cats)
        fs = _cast_spark_decimals_to_double(fs)
        cats_cols = set(cats.columns)
        fs_extra = [
            c for c in fs.columns
            if c not in cats_cols and not c.endswith("_dep")
        ]

        # Filter pushdown - happens on the workers BEFORE we collect to
        # the driver, so the driver only ever holds the filtered subset.
        if target_tokens and category_col in cats_cols:
            # Mirror _normalise_category_token in Spark SQL: upper, replace
            # & -> space, collapse non-alphanumerics to '_', trim '_'.
            norm_expr = F.regexp_replace(
                F.regexp_replace(
                    F.regexp_replace(
                        F.upper(F.col(category_col).cast("string")),
                        r"&", " "
                    ),
                    r"[^A-Z0-9]+", "_"
                ),
                r"^_+|_+$", ""
            )
            mask = norm_expr.isin(target_tokens)
            # Direct-equality safety net for verbatim matches
            if raw_targets:
                mask = mask | F.col(category_col).cast("string").isin(raw_targets)
            cats = cats.filter(mask)
            logger.info(
                "[Spark filter pushdown] applying %s filter: %s",
                category_col, target_tokens,
            )
        elif target_tokens:
            logger.warning(
                "category_list provided but column %r not found in cats "
                "frame; falling back to load-all-then-filter (memory risk!)",
                category_col,
            )

        joined = cats.join(
            fs.select(*_JOIN_COLS, *fs_extra),
            on=_JOIN_COLS,
            how="inner",
        )
        logger.info(
            f"Joined cats ({len(cats.columns)} cols) + FS extras "
            f"({len(fs_extra)} cols) on {_JOIN_COLS} (inner) via Spark — "
            f"Arrow toPandas enabled"
        )
        # toPandas is the action that runs the plan. We deliberately
        # avoid `.count()` calls before/after the filter — each one is a
        # full Spark action that re-runs the load+join+filter plan, and
        # the row count is free from `len(df_pd)` after toPandas anyway.
        df_pd = joined.toPandas()
        # Help Linux release the Spark frames so toPandas's intermediate
        # allocations can be reused for downstream feature engineering.
        try:
            joined.unpersist()
            cats.unpersist()
            fs.unpersist()
        except Exception:
            pass
        import gc as _gc; _gc.collect()

        # Aggressive dtype downcast: turns the object/int64/float64 dump
        # Spark hands us into the smallest dtypes that still hold the
        # data losslessly.  On a typical FS frame this is a 30-60% RAM
        # cut — the difference between the driver swapping and not.
        df_pd = _optimize_dtypes_inplace(df_pd)

        logger.info(
            "[Spark->pandas] %s rows x %s cols on driver (%.1f MB after dtype opt)",
            len(df_pd), len(df_pd.columns),
            df_pd.memory_usage(deep=True).sum() / 1024 / 1024,
        )
        return df_pd
    except ImportError:
        logger.info("PySpark unavailable — using pandas Parquet read")

    def _read_pd(p):
        # Delta table (has _delta_log): try deltalake first so tombstoned/rewritten
        # files are excluded. Falls back to plain parquet if deltalake can't handle
        # newer features (e.g., deletion vectors). PySpark on Databricks handles
        # these natively — this fallback only affects local runs.
        if (Path(p) / "_delta_log").exists():
            try:
                from deltalake import DeltaTable
                return DeltaTable(str(p)).to_pandas()
            except Exception as e:
                logger.warning("[_read_pd] DeltaTable read failed (%s), falling back to parquet: %s", type(e).__name__, e)
        return pd.read_parquet(p)

    cats_pd = _read_pd(cats_path)
    if target_tokens and category_col in cats_pd.columns:
        norm = cats_pd[category_col].astype(str).map(_normalise_category_token)
        raw  = cats_pd[category_col].astype(str)
        n_before = len(cats_pd)
        cats_pd = cats_pd.loc[norm.isin(target_tokens) | raw.isin(raw_targets or [])]
        logger.info(
            "[pandas filter] %s in %s: %s -> %s rows",
            category_col, target_tokens, n_before, len(cats_pd),
        )
    fs_pd = _read_pd(fs_path)
    cats_cols = set(cats_pd.columns)
    fs_extra = [
        c for c in fs_pd.columns
        if c not in cats_cols and not c.endswith("_dep")
    ]
    joined_pd = cats_pd.merge(
        fs_pd[_JOIN_COLS + fs_extra], on=_JOIN_COLS, how="inner"
    )
    # Free the source frames before dtype opt — the merge already
    # copied everything it needed.
    del cats_pd, fs_pd
    import gc as _gc; _gc.collect()
    joined_pd = _optimize_dtypes_inplace(joined_pd)
    logger.info(
        f"Joined cats + FS extras ({len(fs_extra)} cols) on {_JOIN_COLS} "
        f"(inner) via pandas: {len(joined_pd)} rows x {len(joined_pd.columns)} "
        f"cols ({joined_pd.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB)"
    )
    return joined_pd


# ---------------------------------------------------------------------------
# Per-category filter + save (NO other transforms)
# ---------------------------------------------------------------------------

def _filter_and_save_category(
    full_df: pd.DataFrame,
    category: str,
    category_col: str,
    sourcedata_dir: Path,
    save: bool = True,
) -> Tuple[Path, pd.DataFrame]:
    """Filter the unified input table to one category and write it to
    ``sourcedata_dir/{CATEGORY}.csv``. No other preprocessing — the
    upstream DIQ feed already delivers data that's ready for the
    pipeline. Returns ``(csv_path, filtered_df)`` so callers can hand
    the DataFrame to ``cfg.auto_detect_splits`` without re-reading.

    ``save=False`` skips the parquet/CSV write and returns the would-be
    path unwritten. Used for UK-engine-routed categories: the engine
    trains from the pooled ``full_df`` in memory and never reads the
    per-category sourcedata file — its only consumer is the legacy GBM
    pipeline via ``cfg.input_data_path``. Skipping the write avoids
    10x 600-col parquet writes to the UC Volume per snapshot and stops
    publishing per-category data splits nobody consumes.

    Matching strategy: accept either the widget token ("COOKING_AID")
    or a permissive variant (case-insensitive, spaces↔underscores,
    "&"↔"AND") so the caller doesn't have to normalise their input
    table's category column first.
    """
    if category_col not in full_df.columns:
        raise ValueError(
            f"Input table has no '{category_col}' column (columns: "
            f"{list(full_df.columns)[:20]}…)"
        )

    # Normalisation strategy:
    #   * UPPERCASE
    #   * Remove "&" entirely (it's connective noise, not a token)
    #   * Replace any whitespace run with a single "_"
    #   * Collapse consecutive "_" into one
    #   * Strip leading/trailing "_"
    # This maps "DEODORANT & FRAGRANCE" ↔ "DEODORANT_FRAGRANCE",
    #          "HOME & HYGIENE"         ↔ "HOME_HYGIENE", etc.
    def _normalise(s: str) -> str:
        import re
        s = s.upper().replace("&", " ")
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    target_token = _normalise(category)
    norm = full_df[category_col].astype(str).map(_normalise)
    mask = norm == target_token
    # Direct-equality safety net for verbatim matches
    mask = mask | (full_df[category_col].astype(str) == category)

    sub = full_df.loc[mask].copy()
    if sub.empty:
        unique = full_df[category_col].astype(str).unique().tolist()[:20]
        raise ValueError(
            f"No rows match category='{category}' in column '{category_col}'. "
            f"Sample of available values: {unique}"
        )
    logger.info(f"[{category}] {len(sub)} rows after category filter")

    out_path = sourcedata_dir / f"{category.upper()}.parquet"
    if not save:
        # UK-engine path: no consumer for the per-category file — skip the write.
        logger.info(
            f"[{category}] sourcedata write SKIPPED (uk_engine path; "
            f"{len(sub)} rows kept in memory)"
        )
        return out_path, sub

    sourcedata_dir.mkdir(parents=True, exist_ok=True)
    # Parquet instead of CSV for the per-category checkpoint:
    #   * 10-50x faster to write (no string serialisation)
    #   * 3-5x smaller on disk (columnar + Snappy)
    #   * Preserves dtypes (so downstream auto_detect_splits doesn't have
    #     to guess year_week is int vs string)
    # `load_source_data` (utils/agent_utilities.py) auto-detects format
    # from the extension, so downstream consumers transparently handle
    # the switch.  We keep a thin CSV fallback if pyarrow is somehow
    # missing on the cluster.
    try:
        sub.to_parquet(out_path, index=False)
    except Exception as exc:
        logger.warning(
            "[%s] parquet save failed (%s); falling back to CSV", category, exc
        )
        out_path = sourcedata_dir / f"{category.upper()}.csv"
        sub.to_csv(out_path, index=False)
    logger.info(f"[{category}] saved {len(sub)} rows to {out_path}")
    return out_path, sub


# ---------------------------------------------------------------------------
# Runtime config factory
# ---------------------------------------------------------------------------

def _build_runtime_config(
    category: str,
    project_root: str,
    snapshot_dir: Path,
    input_csv_path: Path,
    history_till: str,
    snapshot_week: str,
    category_df: Optional[pd.DataFrame] = None,
    market: str = "uk",
    config_root: Optional[str] = None,
):
    """Load the template YAML, override paths + time splits, force
    ``run_mode='forecast_only'`` (skips backtest), disable per-category
    MOQ (stacked MOQ runs once at the end).

    The widget values determine **val_end / test_start / test_end**
    explicitly.  Train_start is left empty and filled by the schema's
    own ``cfg.auto_detect_splits(df)`` method - that's what the schema
    docstring at config/schema.py:2037-2054 promises and what
    ``DemandForecastConfig.auto_detect_splits`` (line 2243) implements.
    Calling that method is the right way to honour any explicit splits
    the user set in the YAML while filling in only the missing ones
    from the actual data.

    Pass ``category_df`` to skip re-reading the CSV; we already have
    it in memory from ``_filter_and_save_category``.
    """
    from config.schema import load_config_from_yaml
    from utils.backtesting import increment_period, decrement_period

    template = _yaml_template_for(
        category, project_root, market=market, config_root=config_root,
    )
    cfg = load_config_from_yaml(str(template))

    category_dir = snapshot_dir / category.upper()
    artifact_dir = category_dir / "artifacts"

    # Redirect every path at the snapshot-scoped tree
    cfg.databricks_base_path = str(category_dir) + "/"
    cfg.artifact_base_path = str(artifact_dir)
    cfg.output_folder_path = str(category_dir) + "/"
    cfg.input_data_path = str(input_csv_path)

    # Forward-forecast-only — the deterministic pipeline guards the
    # backtest stage on cfg.should_backtest which returns False here.
    cfg.run_mode = "forecast_only"

    # Time splits from widget values: val_end is the last observed
    # period (history_till), val_start is val_periods earlier,
    # train_end is one period before val_start, and test_start /
    # test_end frame the forward-forecast window.  train_start is
    # left empty so cfg.auto_detect_splits() can fill it from the
    # actual data minimum below.
    tf = cfg.time_format
    val_periods = 3 if tf == "year_month" else 13
    cfg.train_start = ""  # ← filled by auto_detect_splits(df) below
    cfg.val_end = str(history_till)
    cfg.val_start = decrement_period(cfg.val_end, val_periods - 1, tf)
    cfg.train_end = decrement_period(cfg.val_start, 1, tf)
    cfg.test_start = str(snapshot_week)
    cfg.test_end = increment_period(cfg.test_start, cfg.forecast_horizon - 1, tf)

    # Per-category MOQ off; run once after the stack in _stack_and_moq.
    cfg.design.apply_moq_postprocessing = False

    # Honour the schema's own auto-detect API.  It only fills splits
    # that are still empty (we already set train_end / val_* / test_*),
    # so this resolves train_start to the earliest period in the data
    # without disturbing the explicitly-set boundaries.
    df_for_autodetect = category_df
    if df_for_autodetect is None:
        try:
            # Format-agnostic read: the per-category save now defaults to
            # parquet (much faster + dtype-preserving) and only falls
            # back to CSV when pyarrow is missing.  Detect format from
            # the suffix so this path stays robust either way.
            input_csv_path_str = str(input_csv_path)
            if input_csv_path_str.endswith((".parquet", ".pq")):
                df_for_autodetect = pd.read_parquet(
                    input_csv_path_str,
                    columns=[cfg.timestamp_col, cfg.target_col],
                )
            else:
                df_for_autodetect = pd.read_csv(
                    input_csv_path_str,
                    usecols=[cfg.timestamp_col, cfg.target_col],
                    low_memory=False,
                )
        except Exception as _exc:
            logger.warning(
                "[%s] Could not read %s for auto-detect: %s; train_start will "
                "remain empty and the pipeline will fail with `int('')`.",
                category, input_csv_path, _exc,
            )
    if df_for_autodetect is not None and not df_for_autodetect.empty:
        try:
            cfg.auto_detect_splits(df_for_autodetect)
        except Exception as _exc:
            logger.warning(
                "[%s] auto_detect_splits failed: %s; train_start will remain "
                "empty.  Likely cause: timestamp_col '%s' or target_col '%s' "
                "isn't in the input data.",
                category, _exc, cfg.timestamp_col, cfg.target_col,
            )

    # ----------------------------------------------------------------
    # Coerce ALL 6 splits to match the data column's physical format.
    #
    # We arrive here with a possibly-mixed bag of formats because three
    # different code paths populate the splits:
    #
    #   - Widget values (history_till="2026-10", snapshot_week="2026-11")
    #     come in as hyphenated YYYY-WW
    #   - decrement_period / increment_period preserve the hyphenated
    #     format ("2025-50", "2026-23")
    #   - auto_detect_splits produces splits in whatever format the data
    #     column already has (typically digit-only "202301" for int64
    #     year_week columns)
    #
    # If the data column is int64 (typical for year_week CSVs), the
    # downstream `feature_engineering.run_leakage_free_feature_pipeline`
    # does `int(val_end)` and crashes on a hyphenated value with
    # `ValueError: invalid literal for int() with base 10: '2026-10'`.
    #
    # The fix: detect the data column's format ONCE and align every
    # split to match.  This is the single chokepoint where all splits
    # become consistent.
    # ----------------------------------------------------------------
    if df_for_autodetect is not None and not df_for_autodetect.empty:
        try:
            col_dtype = df_for_autodetect[cfg.timestamp_col].dtype
            sample = df_for_autodetect[cfg.timestamp_col].dropna().iloc[0]

            # Use pandas dtype for the integer check - np.int64 is NOT
            # a python int, so isinstance(sample, int) returns False.
            data_is_int = pd.api.types.is_integer_dtype(col_dtype) or (
                pd.api.types.is_float_dtype(col_dtype)
            )
            data_is_str_numeric = (
                pd.api.types.is_object_dtype(col_dtype)
                or pd.api.types.is_string_dtype(col_dtype)
            ) and isinstance(sample, str) and sample.replace(".", "").isdigit()
            data_is_hyphenated = isinstance(sample, str) and "-" in sample and not str(sample).replace("-", "").replace(".", "").isdigit()
            # Note: a hyphenated period like "2025-49" has digits-only after
            # stripping the hyphen, so the check above wouldn't fire.  Use a
            # simpler, more direct check:
            if isinstance(sample, str) and "-" in str(sample):
                # Hyphenated form (YYYY-WW or YYYY-MM-DD)
                data_is_hyphenated = True
                data_is_int = False
            else:
                data_is_hyphenated = False
            # data_is_numeric covers BOTH int dtype and digit-string ("202610")
            data_is_numeric = data_is_int or data_is_str_numeric

            logger.info(
                "[%s] data column '%s' format: dtype=%s, sample=%r, "
                "numeric=%s, hyphenated=%s",
                category, cfg.timestamp_col, col_dtype, sample,
                data_is_numeric, data_is_hyphenated,
            )

            for fld in ("train_start", "train_end", "val_start", "val_end",
                         "test_start", "test_end"):
                v = getattr(cfg, fld, "")
                if not v:
                    continue
                v_str = str(v)
                if data_is_numeric:
                    # Strip hyphens: "2026-10" -> "202610".
                    new_v = v_str.replace("-", "")
                    if new_v.isdigit():
                        setattr(cfg, fld, new_v)
                elif data_is_hyphenated:
                    # Add hyphen if missing: "202610" -> "2026-10".
                    if "-" not in v_str and v_str.isdigit() and len(v_str) >= 5:
                        # YYYYWW or YYYYMM (6 digits).  Hyphenate as YYYY-WW.
                        new_v = f"{v_str[:4]}-{v_str[4:].zfill(2)}"
                        setattr(cfg, fld, new_v)
                # else: datetime / unknown -> leave as-is, downstream picks up
        except Exception as _exc:
            logger.warning(
                "[%s] split format coercion failed: %s; pipeline may crash on "
                "type mismatch.", category, _exc,
            )

    logger.info(
        f"[{category}] runtime config: "
        f"train=[{cfg.train_start}..{cfg.train_end}], "
        f"val=[{cfg.val_start}..{cfg.val_end}], "
        f"test=[{cfg.test_start}..{cfg.test_end}], "
        f"run_mode={cfg.run_mode}"
    )
    return cfg


# ---------------------------------------------------------------------------
# Stack + MOQ + final write
# ---------------------------------------------------------------------------

def _stack_and_moq(
    category_paths: Dict[str, Path],
    full_df: pd.DataFrame,
    parent_dir: Path,
    snapshot_week: str,
    first_cfg,  # DemandForecastConfig — for target/key/date/train_end
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """Concatenate each category's inference_forecast.csv, run MOQ post-
    processing on the stacked frame using the unified history, and write
    the final output(s).

    Output layout:
      * When ``out_path`` is provided (non-blank), the parent directory of
        ``out_path`` is treated as the output directory and ONE Parquet per
        category is written there using the pattern
        ``{out_dir}/{CATEGORY}_inference_forecast.parquet``. The file name
        in ``out_path`` itself is ignored — only its parent matters.
      * When ``out_path`` is None or blank, the legacy default applies:
        a single stacked Parquet at
        ``{parent_dir}/DIQ/TF_DIQ/{snapshot_week}/inference_forecast.parquet``.
    """
    from utils.moq_postprocessing import apply_moq_postprocessing

    frames: List[pd.DataFrame] = []
    for cat, art_dir in category_paths.items():
        fcst_path = art_dir / "model_artifacts" / "inference_forecast.csv"
        if not fcst_path.exists():
            logger.warning(
                f"[{cat}] inference_forecast.csv missing at {fcst_path} — "
                "skipping in stacked output"
            )
            continue
        df = pd.read_csv(fcst_path)
        df["category"] = cat.upper()
        frames.append(df)
        logger.info(f"[{cat}] stacked {len(df)} forecast rows from {fcst_path}")

    if not frames:
        raise RuntimeError(
            "No per-category inference_forecast.csv files were produced — "
            "cannot build the stacked TF_DIQ output."
        )
    stacked = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Stacked forecast: {len(stacked)} rows across {len(frames)} categories"
    )

    # MOQ on the stacked frame. MOQ detection is per-key and keys are
    # category-scoped, so stacking first and detecting once is identical
    # to doing it per-category but simpler and faster.
    date_col = first_cfg.timestamp_col
    hist = full_df.copy()
    hist[date_col] = hist[date_col].astype(str).str.replace("-", "", regex=False)
    hist_cutoff = str(first_cfg.train_end) if first_cfg.train_end else None

    stacked = apply_moq_postprocessing(
        stacked,
        first_cfg,
        history_df=hist,
        forecast_col="predicted",
        output_col="prediction_post_moq",
        history_till=hist_cutoff,
    )

    out_path_norm = (out_path or "").strip() if out_path is not None else ""
    if out_path_norm:
        out_dir = Path(out_path_norm).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for cat_value, group in stacked.groupby("category", sort=False):
            cat_file = out_dir / f"{cat_value}_inference_forecast.parquet"
            group.to_parquet(cat_file, index=False)
            logger.info(
                f"Wrote per-category forecast Parquet: {cat_file} "
                f"({len(group)} rows)"
            )
    else:
        out_dir = parent_dir / "DIQ" / "TF_DIQ" / str(snapshot_week)
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = out_dir / "inference_forecast.parquet"
        stacked.to_parquet(parquet_path, index=False)
        logger.info(f"Wrote stacked forecast Parquet: {parquet_path}")

    return stacked


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def _load_market_io(market: str, project_root: Optional[str] = None) -> Dict[str, str]:
    """Resolve the market-specific data-IO from ``config/config_{market}_base_local.yaml``.

    This is the SINGLE source of UK-vs-DE differences: ``run_diq_forecast(market=...)``
    reads the ``market_io`` block so the exact same code runs both markets on a pure
    configuration switch. Returns ``{category_col, cats_subdir, fs_subdir,
    lego_pred_col, lego_subdir, ensemble_weights, base_config}`` (``base_config`` is
    the absolute path to that market's engine config). Missing keys → caller falls
    back to legacy defaults, so an un-migrated market still runs.
    """
    import os
    from utils.config_loader import load_market_config
    rel = f"config/config_{market.lower()}_base_local.yaml"   # base_config string; engine prefers the .py sibling
    root = project_root or str(Path(__file__).resolve().parents[1])
    mio: Dict[str, str] = {}
    try:
        cfg = load_market_config(market.lower(), os.path.join(root, "config"))
        mio = dict((cfg or {}).get("market_io") or {})
    except FileNotFoundError:
        pass
    # base_config stays RELATIVE — the engine resolves it via _resolve_repo_path,
    # same convention as its built-in default.
    mio["base_config"] = rel
    return mio


def run_diq_forecast(
    parent_dir: str,
    history_till: str,
    snapshot_week: str,
    category_list: List[str],
    *,
    project_root: Optional[str] = None,
    category_col: Optional[str] = None,        # None → resolved from config_{market}_base_local.yaml
    continue_on_category_failure: bool = True,
    out_path: Optional[str] = None,
    market: str = "uk",
    cats_subdir: Optional[str] = None,         # None → market_io
    fs_subdir: Optional[str] = None,           # None → market_io
    config_root: Optional[str] = None,
    # --- stacked-ensemble engine (Pattern B - replaces GBM for routed cats; default off).
    #     Every market specific (config, weights, lego column + folder) resolves from
    #     config_{market}_base_local.yaml, so UK vs DE is a pure `market=` switch. ---
    enable_uk_engine: bool = False,
    uk_route_categories: Optional[List[str]] = None,
    uk_weights_path: Optional[str] = None,     # None → market_io ensemble_weights
    uk_lego_source: Optional[str] = None,      # None → {parent_dir}/{market_io lego_subdir}
    lego_pred_col: Optional[str] = None,       # None → market_io (UK prediction_rf, DE prediction_xgb)
    uk_num_threads: int = 1,
) -> pd.DataFrame:
    """End-to-end DIQ forward-forecast run.

    Parameters
    ----------
    parent_dir
        Root Databricks volume path. Two Parquet inputs are required
        directly under it: ``{cats_subdir}/`` and ``{fs_subdir}/``
        (joined inner on ``['key', 'year_week']``). All outputs land
        under ``{parent_dir}/DIQ/``.
    history_till
        Last period of known data, e.g. ``"202614"``. Becomes the
        ``val_end`` for every category's runtime config.
    snapshot_week
        First week to forecast, e.g. ``"202615"``. Used as the folder
        name for this run's artifacts and as ``test_start`` in the
        runtime configs.
    category_list
        Widget tokens (uppercase with underscores, e.g. ``"COOKING_AID"``,
        ``"DEODORANT_FRAGRANCE"``).
    project_root
        Path to the repository root. Auto-detected from this module's
        location when None.
    category_col
        Column in the joined input frame carrying the category label.
        Default ``"category_name"`` (UK). DE runs pass ``"Category"``.
    continue_on_category_failure
        When True (default), a failure on one category logs the error and
        moves on to the next. Set False during development for fail-fast.
    out_path
        Optional output path. When provided (non-blank), the parent of
        ``out_path`` is used as the output directory and one Parquet per
        category is written there as
        ``{out_dir}/{CATEGORY}_inference_forecast.parquet``. When None
        or blank, falls back to the default
        ``{parent_dir}/DIQ/TF_DIQ/{snapshot_week}/inference_forecast.parquet``
        single-file layout.
    market
        Country slug for the per-category YAML template lookup. ``"uk"``
        loads ``config/config_uk_{slug}_databricks.yaml``; ``"de"`` loads
        ``config/config_de_{slug}_databricks.yaml``. Default ``"uk"``.
    cats_subdir
        Folder name (under ``parent_dir``) containing the cats / driver
        Parquet. Defaults to UK's ``01_all_keys_dr_adjusted``; DE runs
        pass ``"DT"``.
    fs_subdir
        Folder name (under ``parent_dir``) containing the feature-store
        Parquet. Defaults to ``FS`` (shared by UK and DE).
    config_root
        Optional directory containing the per-category YAML templates.
        When ``None`` or blank (default) the runner reads templates from
        ``<project_root>/config/`` (the historic in-repo layout). When set
        to a path such as
        ``"/Volumes/pds_feu_931272_dev/feu_common/config/EU_UK/DIQ_configs/"``
        the templates are read from that UC Volume directory instead.
        Filename convention is unchanged: ``config_{market}_{slug}_databricks.yaml``.

    Returns
    -------
    pd.DataFrame
        The stacked forecast DataFrame.
    """
    if project_root is None:
        project_root = str(Path(__file__).resolve().parents[1])
    parent_dir_p = Path(parent_dir)
    snapshot_dir = parent_dir_p / "DIQ" / str(snapshot_week)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # --- single config switch: resolve ALL market specifics from
    #     config/config_{market}_base_local.yaml. Explicit args still override. ---
    _mio = _load_market_io(market, project_root)
    category_col = category_col or _mio.get("category_col") or "category_name"
    cats_subdir  = cats_subdir  or _mio.get("cats_subdir")  or _DEFAULT_CATS_SUBDIR
    fs_subdir    = fs_subdir    or _mio.get("fs_subdir")    or _DEFAULT_FS_SUBDIR
    lego_pred_col   = lego_pred_col   or _mio.get("lego_pred_col")   or "prediction_rf"
    uk_weights_path = uk_weights_path or _mio.get("ensemble_weights") or "config/uk_ensemble_weights.yaml"
    _engine_config_path = _mio.get("base_config")
    if enable_uk_engine and not uk_lego_source:
        uk_lego_source = f"{str(parent_dir).rstrip('/')}/{_mio.get('lego_subdir') or 'TF_modelling_pristine'}"

    logger.info("=" * 70)
    logger.info("DIQ FORWARD-FORECAST ORCHESTRATOR")
    logger.info("=" * 70)
    logger.info(f"  parent_dir:    {parent_dir_p}")
    logger.info(f"  market:        {market}")
    logger.info(f"  cats_subdir:   {cats_subdir}")
    logger.info(f"  fs_subdir:     {fs_subdir}")
    logger.info(f"  category_col:  {category_col}")
    logger.info(f"  snapshot_week: {snapshot_week}")
    logger.info(f"  history_till:  {history_till}")
    logger.info(f"  categories:    {category_list}")
    logger.info(f"  project_root:  {project_root}")
    logger.info(f"  out_path:      {out_path or '(blank — default TF_DIQ layout)'}")
    logger.info(
        f"  config_root:   "
        f"{config_root or '(blank — using <project_root>/config/)'}"
    )

    # CRITICAL for runtime: kill MLflow autologging before any model fit.
    # Databricks turns it on by default, which otherwise creates a fresh
    # MLflow Run for every lgb.train() and balloons runtime by orders of
    # magnitude.  Set DIQ_RUNNER_KEEP_MLFLOW_AUTOLOG=1 to override.
    _disable_mlflow_autolog_unless_requested()

    # Bootstrap: install requirements.txt into the active interpreter when
    # crewai / pydantic v2 isn't already importable.  databricks_runner.py
    # does this at its main(); diq_runner is a notebook entry point so it
    # has to run the same step itself or fail with ModuleNotFoundError on
    # `from crewai import Agent`.
    if not _ensure_dependencies(project_root):
        raise RuntimeError(
            "DIQ runner could not install Python dependencies from "
            "requirements.txt.  Set DIQ_RUNNER_SKIP_INSTALL=1 in the cluster "
            "environment to bypass this check if your cluster is already "
            "pre-provisioned."
        )

    # Push the category filter down to Spark BEFORE toPandas so we don't
    # materialise rows for categories we'll throw away in the pandas filter.
    # On a 64 GB driver running 11 categories this is the difference between
    # the kernel staying alive and an OOM (load average 398, swap saturated).
    full_df = _load_and_join_inputs(
        parent_dir_p,
        category_list=category_list,
        category_col=category_col,
        cats_subdir=cats_subdir,
        fs_subdir=fs_subdir,
    )
    logger.info(
        f"Loaded joined input (filtered to {len(category_list)} categories): "
        f"{len(full_df)} rows × {len(full_df.columns)} cols"
    )

    # --- SEGMENT-ENGINE LIVE path (config-gated by market_io.segment_col; DE). Isolated from
    #     the per-category / uk_engine flow → early return, zero UK impact. Applies the frozen
    #     per-(category × lego_segment) weights to the segment×category×LEGO stacked ensemble. ---
    if _mio.get("segment_col"):
        from types import SimpleNamespace as _SN
        from pathlib import Path as _P
        import yaml as _yaml
        from utils import segment_engine as _se
        _croot = config_root or str(_P(project_root) / "config")
        _cfgp = str(_P(_croot) / f"config_{market}_base_local.yaml")   # compute_segment_components→load_config prefers the .py sibling
        from utils.config_loader import load_market_config as _lmc
        _hz = int(_lmc(market, _croot).get("horizons", 13))
        _legosrc = f"{str(parent_dir_p).rstrip('/')}/{_mio.get('lego_subdir')}"
        comp = _se.compute_segment_components(
            _SN(forecast_horizon=_hz), full_df, snapshot_week=snapshot_week,
            route_categories=category_list, segment_col=_mio["segment_col"],
            lego_source=_legosrc, config_path=_cfgp, lego_pred_col=lego_pred_col, horizons=_hz)
        _wp = (uk_weights_path if str(uk_weights_path).startswith(("/", "dbfs"))
               else str(_P(project_root) / uk_weights_path))
        _w, _d = _se.load_segment_weights(_wp)
        comp["predicted"] = _se.blend_with_weights(comp, _w, default=_d)
        comp["prediction_post_moq"] = comp["predicted"]
        logger.info("[segment_engine] LIVE: %d forecast rows, %d (cat×seg) weights from %s",
                    len(comp), len(_w), _wp)
        if out_path:
            _P(out_path).parent.mkdir(parents=True, exist_ok=True)
            comp.to_parquet(out_path, index=False)
            logger.info("[segment_engine] wrote forecast -> %s", out_path)
        return comp

    # ── Production build is SEGMENT-ONLY (DE + UK both set market_io.segment_col). The legacy
    #    per-category "deterministic pipeline" path below is retired/archived; guard so a missing
    #    segment_col fails loudly instead of importing now-archived modules. ──
    raise RuntimeError(
        "segment-only production build: market_io.segment_col must be set in "
        "config_{market}_base_local.py (DE and UK both do). The legacy non-segment path is archived.")

    from utils.deterministic_pipeline import run_full_deterministic_pipeline

    # Pre-normalise the UK route set so the per-cat dispatch is a cheap membership check.
    # The set holds widget-token form ("DEODORANT_FRAGRANCE") so it matches whatever
    # the caller passed via uk_route_categories.
    if enable_uk_engine:
        from utils.uk_engine import _normalise_token as _uk_norm
        uk_route_set = {_uk_norm(c) for c in (uk_route_categories or []) if str(c).strip()}
        if not uk_route_set:
            logger.warning(
                "[uk_engine] enable_uk_engine=True but uk_route_categories is empty; "
                "no categories will use the new engine — falling back to GBM for all."
            )
    else:
        uk_route_set = set()

    category_artifact_paths: Dict[str, Path] = {}
    first_cfg = None
    failures: List[str] = []

    for raw_category in category_list:
        category = raw_category.strip()
        if not category:
            continue
        category_dir = snapshot_dir / category.upper()
        sourcedata_dir = category_dir / "sourcedata"
        artifact_dir = category_dir / "artifacts"
        logger.info("-" * 70)
        logger.info(f"CATEGORY: {category} → {category_dir}")
        logger.info("-" * 70)
        try:
            _is_uk_routed = False
            if enable_uk_engine and uk_route_set:
                from utils.uk_engine import _normalise_token as _uk_norm_pre
                _is_uk_routed = _uk_norm_pre(category) in uk_route_set
            csv_path, category_df = _filter_and_save_category(
                full_df=full_df,
                category=category,
                category_col=category_col,
                sourcedata_dir=sourcedata_dir,
                save=not _is_uk_routed,    # UK engine trains from full_df; file has no consumer
            )

            cfg = _build_runtime_config(
                category=category,
                project_root=project_root,
                snapshot_dir=snapshot_dir,
                input_csv_path=csv_path,
                history_till=history_till,
                snapshot_week=snapshot_week,
                category_df=category_df,
                market=market,
                config_root=config_root,
            )
            if first_cfg is None:
                first_cfg = cfg

            # Dispatch: UK stacked-ensemble engine REPLACES the GBM pipeline for
            # routed categories (Pattern B - engine replacement, not overlay).
            # The engine's heavy work runs ONCE (on first routed call within this
            # run_diq_forecast invocation) and produces forecasts for every routed
            # category up-front; this per-cat call just writes the matching CSV.
            if _is_uk_routed:
                from utils.uk_engine import run_uk_engine
                run_uk_engine(
                    cfg, full_df,
                    category=category,
                    snapshot_week=snapshot_week,
                    route_categories=uk_route_categories,
                    weights_path=uk_weights_path,
                    lego_source=uk_lego_source,
                    config_path=_engine_config_path,
                    lego_pred_col=lego_pred_col,
                    num_threads=uk_num_threads,
                )
                logger.info(f"[{category}] uk_engine path — artifacts at {artifact_dir}")
            else:
                run_full_deterministic_pipeline(cfg, clean_artifacts=True)
                logger.info(f"[{category}] GBM pipeline path — artifacts at {artifact_dir}")
            category_artifact_paths[category] = artifact_dir

        except Exception as exc:
            failures.append(category)
            logger.error(f"[{category}] FAILED: {exc}", exc_info=True)
            if not continue_on_category_failure:
                raise

    if not category_artifact_paths:
        raise RuntimeError(
            f"Every requested category failed ({failures}); no forecasts to stack."
        )

    if failures:
        logger.warning(
            f"Completed with {len(failures)} category failure(s): {failures}"
        )

    # The per-category loop is done; we no longer need the full FS+cats
    # bag.  MOQ only consumes [key, timestamp, target] (see
    # moq_postprocessing.apply_moq_postprocessing line 327), so project
    # full_df down to those columns + the keys needed to derive 'key'
    # if it isn't already present.  On a 200-column DIQ frame this is a
    # 95%+ memory cut right before the stack — exactly the moment we
    # need that headroom for `pd.concat` of all category forecasts.
    if first_cfg is not None:
        moq_cols = set()
        moq_cols.add(first_cfg.timestamp_col)
        moq_cols.add(first_cfg.target_col)
        if "key" in full_df.columns:
            moq_cols.add("key")
        for k in getattr(first_cfg, "prediction_key_cols", []) or []:
            if k in full_df.columns:
                moq_cols.add(k)
        # Keep the category column too — useful for diagnostics and
        # cheap (low cardinality, already category dtype after our
        # dtype-opt step).
        if category_col in full_df.columns:
            moq_cols.add(category_col)
        keep = [c for c in full_df.columns if c in moq_cols]
        if keep and len(keep) < len(full_df.columns):
            mem_before = full_df.memory_usage(deep=True).sum() / 1024 / 1024
            full_df = full_df[keep].copy()
            mem_after = full_df.memory_usage(deep=True).sum() / 1024 / 1024
            import gc as _gc; _gc.collect()
            logger.info(
                "[memory] projected full_df %d cols -> %d cols for MOQ "
                "(%.1f MB -> %.1f MB, -%.0f%%)",
                len(keep) + (len(full_df.columns) - len(keep)),  # cosmetic
                len(keep), mem_before, mem_after,
                100 * (1 - mem_after / max(mem_before, 1e-9)),
            )

    logger.info("=" * 70)
    logger.info("STACKING + MOQ POST-PROCESSING")
    logger.info("=" * 70)
    stacked = _stack_and_moq(
        category_paths=category_artifact_paths,
        full_df=full_df,
        parent_dir=parent_dir_p,
        snapshot_week=str(snapshot_week),
        first_cfg=first_cfg,
        out_path=out_path,
    )

    logger.info("=" * 70)
    logger.info("DIQ RUN COMPLETE")
    logger.info(
        f"  Categories:        {len(category_artifact_paths)} succeeded, "
        f"{len(failures)} failed"
    )
    logger.info(f"  Stacked rows:      {len(stacked)}")
    logger.info(
        f"  MOQ-rewritten:     "
        f"{(stacked['prediction_post_moq'] != stacked['predicted']).sum()} rows"
    )
    final_loc = (
        str(Path(out_path).parent) if (out_path and out_path.strip())
        else str(parent_dir_p / 'DIQ' / 'TF_DIQ' / str(snapshot_week))
    )
    logger.info(f"  Final output at:   {final_loc}")
    logger.info("=" * 70)

    return stacked


__all__ = ["run_diq_forecast"]
