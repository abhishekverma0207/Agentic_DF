"""Single entry point for loading the per-market base config.

The Databricks pipeline cannot carry .yaml/.json config, so DE and UK now ship their config
as ``config/config_{market}_base_local.py`` (a module exposing a ``CONFIG`` dict). This loader
PREFERS that .py and falls back to the legacy ``.yaml`` (still used by Thailand / others), so
nothing breaks while markets migrate. ``.py`` is executed (not imported) so ``config/`` need not
be an importable package.
"""
from pathlib import Path
from typing import Optional


def _exec_config(py_path) -> dict:
    ns: dict = {}
    exec(compile(Path(py_path).read_text(), str(py_path), "exec"), ns)
    cfg = ns.get("CONFIG")
    if not isinstance(cfg, dict):
        raise ValueError(f"{py_path} must define a top-level CONFIG dict")
    return dict(cfg)


def load_market_config(market: str, config_root: Optional[str] = "config") -> dict:
    """Load ``config_{market}_base_local`` — prefers the .py module, falls back to .yaml.
    Searches ``config_root`` then a plain ``config/`` (mirrors the old lookup)."""
    roots = [Path(config_root or "config"), Path("config")]
    for r in roots:
        py = r / f"config_{market}_base_local.py"
        if py.exists():
            return _exec_config(py)
    for r in roots:
        yml = r / f"config_{market}_base_local.yaml"
        if yml.exists():
            import yaml
            return yaml.safe_load(open(yml)) or {}
    raise FileNotFoundError(
        f"no config_{market}_base_local.(py|yaml) under {config_root!r} or 'config/'")


def load_config_file(path) -> dict:
    """Load a base config given an explicit (possibly .yaml) path: prefer the sibling .py."""
    p = Path(path)
    py = p.with_suffix(".py")
    if py.exists():
        return _exec_config(py)
    if p.exists():
        import yaml
        return yaml.safe_load(open(p)) or {}
    # last resort: try the .py even if the .yaml path doesn't exist
    if py.exists():
        return _exec_config(py)
    raise FileNotFoundError(f"config not found: {path} (and no sibling {py.name})")
