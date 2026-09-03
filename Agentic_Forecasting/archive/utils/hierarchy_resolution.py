"""Single-source-of-truth hierarchy resolver.

Every stage of the forecasting pipeline that needs to know "which columns
are hierarchy levels for this run" must go through this module. That
guarantees feature engineering, multi-level model training, hierarchical
reconciliation, and recursive inference all agree — previously they all
called ``detect_hierarchy_columns`` independently and could drift.

Resolution precedence (first match wins)
----------------------------------------

1. **Explicit override in config (`design.hierarchy_cols`)**

   Accepts either a flat list (legacy form, promoted to a
   single-product chain) or a dict of named chains::

       hierarchy_cols:
         product:  [market_name, brand_name, forecast_familygroup]
         customer: [APG_code]

2. **Previously persisted artifact
   (`seg_output/hierarchy_detection.json`)**

   Reading this artifact is what makes sure every stage sees the same
   hierarchies even when stages run in separate processes or different
   job steps. Segmentation writes it; everyone else reads it.

3. **Fresh detection against
   ``design.hierarchy_detection.candidate_cols``**

   Runs :func:`utils.segmentation_enhanced.detect_hierarchy_chains`
   against the curated candidate list. This is the recommended
   production path: the user writes down which columns are real
   business hierarchies in the config and the code figures out nesting,
   aliases, parallel hierarchies, etc.

4. **Legacy full auto-detect**

   Only used when both ``hierarchy_cols`` and ``candidate_cols`` are
   empty. Runs the detector against every non-key/date/target column
   with ``include_binary=False``. Preserved for configs that have not
   yet been migrated — a warning is logged every time this path runs.

The result is a :class:`HierarchyResolution` dataclass with the chains,
a ``flat`` list for legacy single-chain consumers, diagnostic info, and
a ``source`` field saying which precedence tier was used.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


ARTIFACT_FILENAME = "hierarchy_detection.json"


@dataclass
class HierarchyResolution:
    """Result of resolving hierarchies for a run.

    Attributes
    ----------
    product : List[str]
        Ordered coarse→fine column names for the product chain.
    customer : List[str]
        Ordered coarse→fine column names for the customer / distribution
        chain. Empty when none was detected.
    other : List[str]
        Additional parallel / unclassified hierarchy columns that did
        not fit into the primary chains.
    flat : List[str]
        Convenience list ``product + customer + other`` deduplicated. Used
        by legacy callers that expect a single flat list.
    primary_product_col : Optional[str]
        The single most-useful column for single-col consumers (e.g.
        :func:`train_multi_level_ensemble` wants one ``hierarchy_col``).
        Picked as the finest-level of the product chain, falling back to
        customer, falling back to other.
    source : str
        One of ``"config_explicit"`` / ``"artifact"`` / ``"candidate_list"``
        / ``"auto_detect"`` / ``"empty"`` — which precedence tier produced
        this result.
    diagnostics : Dict[str, Any]
        Full detector output + metadata for audit. Written to the artifact.
    """

    product: List[str] = field(default_factory=list)
    customer: List[str] = field(default_factory=list)
    other: List[str] = field(default_factory=list)
    flat: List[str] = field(default_factory=list)
    primary_product_col: Optional[str] = None
    source: str = "empty"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.product or self.customer or self.other)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product": list(self.product),
            "customer": list(self.customer),
            "other": list(self.other),
            "flat": list(self.flat),
            "primary_product_col": self.primary_product_col,
            "source": self.source,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HierarchyResolution":
        return cls(
            product=list(d.get("product", [])),
            customer=list(d.get("customer", [])),
            other=list(d.get("other", [])),
            flat=list(d.get("flat", [])),
            primary_product_col=d.get("primary_product_col"),
            source=d.get("source", "artifact"),
            diagnostics=d.get("diagnostics", {}),
        )


def _coerce_hierarchy_cols_override(value: Any) -> Optional[Dict[str, List[str]]]:
    """Normalise ``design.hierarchy_cols`` to a dict of named chains.

    * ``None`` / empty → returns None (no explicit override).
    * List[str] (legacy form) → ``{"product": list}``.
    * Dict[str, List[str]] → passed through (filtered to known buckets).
    """
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        return {"product": list(value)}
    if isinstance(value, dict):
        normalised = {
            k: list(v) for k, v in value.items()
            if isinstance(v, (list, tuple)) and v
        }
        return normalised or None
    logger.warning(
        "design.hierarchy_cols has unsupported type %s — ignoring override.",
        type(value).__name__,
    )
    return None


def _build_flat(product: List[str], customer: List[str], other: List[str]) -> List[str]:
    seen: set = set()
    flat: List[str] = []
    for col in product + customer + other:
        if col and col not in seen:
            seen.add(col)
            flat.append(col)
    return flat


def _primary_product(product: List[str], customer: List[str], other: List[str]) -> Optional[str]:
    """Legacy signature kept for back-compat.

    New callers should use :func:`_primary_from_detail` which considers
    the per-column score so multi-level-ensemble training picks a
    mid-cardinality grouping (e.g. brand with 28 groups × 165 keys/group)
    instead of the finest leaf (e.g. gtin with 237 groups × 19 keys/group
    — too thin for reliable per-group parameters).
    """
    if product:
        return product[-1]
    if customer:
        return customer[-1]
    if other:
        return other[-1]
    return None


def _primary_from_detail(detail: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    """Pick the single best hierarchy column from the detector detail.

    Prefers, in order:
    1. The highest-scored column in the product chain.
    2. The highest-scored column in the customer chain.
    3. The highest-scored column across all other detected hierarchies.

    Score combines cardinality + keys-per-group balance + name match —
    see ``detect_hierarchy_chains``. Picking by score means a brand-level
    chain (~30 groups × ~150 keys/group) is preferred over a gtin-level
    chain (~200 groups × ~20 keys/group), which is the sweet spot for
    multi-level ensemble training and MinT reconciliation.
    """
    for bucket_name in ("product", "customer", "other"):
        bucket = detail.get(bucket_name) or []
        if bucket:
            best = max(bucket, key=lambda it: it.get("score", 0.0))
            return best["column"]
    return None


def _load_artifact(seg_dir: Optional[str]) -> Optional[HierarchyResolution]:
    if not seg_dir:
        return None
    path = os.path.join(seg_dir, ARTIFACT_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        res = HierarchyResolution.from_dict(payload)
        logger.info("Loaded hierarchy resolution from artifact: %s", path)
        return res
    except Exception as exc:
        logger.warning("Failed to load %s: %s — ignoring.", path, exc)
        return None


def save_resolution(res: HierarchyResolution, seg_dir: str) -> str:
    """Write the resolution to ``<seg_dir>/hierarchy_detection.json``.

    Called by the segmentation stage so all downstream stages read the
    same answer.
    """
    os.makedirs(seg_dir, exist_ok=True)
    path = os.path.join(seg_dir, ARTIFACT_FILENAME)
    with open(path, "w") as f:
        json.dump(res.to_dict(), f, indent=2, default=str)
    logger.info("Persisted hierarchy resolution → %s (source=%s)", path, res.source)
    return path


def resolve_hierarchies(
    config: Any,
    source_df: Optional[pd.DataFrame] = None,
    seg_dir: Optional[str] = None,
    force_redetect: bool = False,
) -> HierarchyResolution:
    """Resolve the hierarchies that every downstream stage should use.

    Parameters
    ----------
    config : DemandForecastConfig
        Loaded config — the resolver reads
        ``config.design.hierarchy_cols`` (override),
        ``config.design.hierarchy_detection`` (candidate list + thresholds),
        plus ``config.prediction_key_cols``,
        ``config.timestamp_col``, ``config.target_col``.
    source_df : pd.DataFrame, optional
        Source data required for tiers 3 (candidate list) and 4 (auto).
        When ``None`` and neither override nor artifact is available, the
        function returns an empty resolution and logs a warning.
    seg_dir : str, optional
        Segmentation output directory used for reading / writing the
        artifact. When ``None`` the artifact path is not consulted and
        detection is re-run every call.
    force_redetect : bool
        When True, skip the artifact and run detection fresh. Useful for
        segmentation itself, which produces the artifact.

    Returns
    -------
    HierarchyResolution
    """
    design = getattr(config, "design", None)

    # --- Tier 1: explicit override in config ---------------------------
    override = _coerce_hierarchy_cols_override(getattr(design, "hierarchy_cols", None))
    if override:
        product = list(override.get("product", []))
        customer = list(override.get("customer", []))
        other = list(override.get("other", []))
        flat = _build_flat(product, customer, other)
        primary = _primary_product(product, customer, other)
        logger.info(
            "Hierarchy source = config_explicit. product=%s customer=%s other=%s",
            product, customer, other,
        )
        return HierarchyResolution(
            product=product, customer=customer, other=other, flat=flat,
            primary_product_col=primary, source="config_explicit",
            diagnostics={"override": override},
        )

    # --- Tier 2: persisted artifact -----------------------------------
    if not force_redetect:
        existing = _load_artifact(seg_dir)
        if existing and not existing.is_empty():
            # Make sure the source is stable in logs
            existing.source = "artifact"
            logger.info(
                "Hierarchy source = artifact. product=%s customer=%s other=%s",
                existing.product, existing.customer, existing.other,
            )
            return existing

    # --- Tier 3 & 4: run detector -------------------------------------
    if source_df is None:
        logger.warning(
            "resolve_hierarchies called with no source_df and no artifact — "
            "returning empty resolution.",
        )
        return HierarchyResolution(source="empty")

    from utils.segmentation_enhanced import detect_hierarchy_chains

    key_cols = list(getattr(config, "prediction_key_cols", []))
    date_col = getattr(config, "timestamp_col", None)
    target_col = getattr(config, "target_col", None)
    if not (key_cols and date_col and target_col):
        logger.warning(
            "resolve_hierarchies: missing key/date/target in config — returning empty.",
        )
        return HierarchyResolution(source="empty")

    hd = getattr(design, "hierarchy_detection", None)
    if hd is not None and hd.candidate_cols:
        result = detect_hierarchy_chains(
            source_df=source_df,
            key_cols=key_cols,
            date_col=date_col,
            target_col=target_col,
            candidate_cols=list(hd.candidate_cols),
            min_keys_per_group=hd.min_keys_per_group,
            min_cardinality=hd.min_cardinality,
            max_cardinality=hd.max_cardinality,
            require_coverage=hd.require_coverage,
            include_binary=hd.include_binary,
            product_name_patterns=list(hd.product_name_patterns),
            customer_name_patterns=list(hd.customer_name_patterns),
            collapse_duplicate_columns=hd.collapse_duplicate_columns,
            max_depth_per_chain=hd.max_depth_per_chain,
        )
        source_tag = "candidate_list"
    else:
        logger.warning(
            "resolve_hierarchies: no candidate_cols set — falling back to "
            "full auto-detection. Configure "
            "design.hierarchy_detection.candidate_cols with the business "
            "hierarchy columns (brand, segment, APG_code, ...) for "
            "production runs.",
        )
        # Use the same detector but without a candidate list and with
        # binary exclusion ON. This is stricter than the historical
        # behaviour and prevents the indicator-column bug.
        result = detect_hierarchy_chains(
            source_df=source_df,
            key_cols=key_cols,
            date_col=date_col,
            target_col=target_col,
            candidate_cols=None,
            min_keys_per_group=(hd.min_keys_per_group if hd else 5),
            min_cardinality=(hd.min_cardinality if hd else 3),
            max_cardinality=(hd.max_cardinality if hd else 2000),
            require_coverage=(hd.require_coverage if hd else 0.95),
            include_binary=(hd.include_binary if hd else False),
            product_name_patterns=(list(hd.product_name_patterns) if hd else None),
            customer_name_patterns=(list(hd.customer_name_patterns) if hd else None),
            collapse_duplicate_columns=(hd.collapse_duplicate_columns if hd else True),
            max_depth_per_chain=(hd.max_depth_per_chain if hd else 3),
        )
        source_tag = "auto_detect"

    product = [it["column"] for it in result.get("product", [])]
    customer = [it["column"] for it in result.get("customer", [])]
    other = [it["column"] for it in result.get("other", [])]
    flat = _build_flat(product, customer, other)
    # Score-based primary picks the best training-level grouping (usually
    # mid-cardinality like brand) rather than the deepest leaf.
    primary = _primary_from_detail({
        "product":  result.get("product", []),
        "customer": result.get("customer", []),
        "other":    result.get("other", []),
    })
    if primary is None:
        primary = _primary_product(product, customer, other)

    res = HierarchyResolution(
        product=product,
        customer=customer,
        other=other,
        flat=flat,
        primary_product_col=primary,
        source=source_tag,
        diagnostics={
            "candidates_evaluated": result.get("candidates_evaluated", []),
            "candidates_rejected":  result.get("candidates_rejected", []),
            "duplicates_collapsed": result.get("duplicates_collapsed", {}),
            "detail": {
                "product":  result.get("product", []),
                "customer": result.get("customer", []),
                "other":    result.get("other", []),
            },
        },
    )

    logger.info(
        "Hierarchy source = %s. product=%s customer=%s other=%s (primary=%s)",
        source_tag, product, customer, other, primary,
    )
    return res


__all__ = [
    "HierarchyResolution",
    "resolve_hierarchies",
    "save_resolution",
    "ARTIFACT_FILENAME",
]
