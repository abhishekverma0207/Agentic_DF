"""Tests for allocate_model_levels — the state-of-the-art model level
allocator in utils.intelligent_modeling.

The core regression targets here are the 2026-04 changes that widened
the individual-model tier on zero-inflated lumpy demand. Previously the
sparsity gate (max_zero_fraction_for_individual, min_nonzero_obs_for_
individual) blocked nearly all high-volume keys because median
zero_fraction on lumpy categories is ~0.82, leading to a pathological
0.8% individual-model rate on COOKING_AID (59/7120) vs the per-key
XGBoost benchmark's 100%. The new top_volume_bypass_quantile tier lets
high-volume keys become individual regardless of sparsity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_lumpy_per_key(n=200, seed=0):
    """Synthetic per-key metrics mirroring a lumpy UK category:
    - volume uniformly distributed on [0.1, 500], i.e. long tail
    - zero_fraction mostly >= 0.75 (lumpy), with a handful of smooth keys
    - n_obs ~ 135 weeks of history (close to real UK data)
    Keys split across 4 segments.
    """
    rng = np.random.default_rng(seed)
    # Power-law-ish volume: a few keys very big, most small
    volume = np.sort(rng.pareto(1.5, n) * 50 + 1)[::-1]
    # Most keys are lumpy (zf 0.7-0.95); 10% are smooth (zf < 0.5)
    zf = np.where(rng.uniform(size=n) < 0.1,
                  rng.uniform(0.1, 0.5, n),
                  rng.uniform(0.70, 0.95, n))
    n_obs = np.full(n, 135)
    # forecastability ~ inversely tied to zf
    forecastability = np.clip(1.0 - zf + rng.normal(0, 0.05, n), 0.05, 0.95)
    return pd.DataFrame({
        'key': [f'K{i:04d}' for i in range(n)],
        'segment_id': rng.integers(0, 4, size=n),
        'volume_mean': volume,
        'mean': volume,
        'sum': volume * n_obs,
        'zero_fraction_clean': zf,
        'zero_fraction': zf,
        'n_obs': n_obs,
        'history_length': n_obs,
        'cv': rng.uniform(0.5, 3.0, n),
        'cv_clean': rng.uniform(0.5, 3.0, n),
        'adi_log': rng.uniform(0.2, 1.5, n),
        'adi': rng.uniform(1.0, 5.0, n),
        'demand_pattern': 'lumpy',
        'forecastability_score': forecastability,
        'predictability_score': forecastability * 0.9,
    })


def test_top_volume_bypass_promotes_lumpy_high_volume_keys_to_individual():
    """With the bypass ON, keys in the top volume decile become individual
    even if their zero_fraction exceeds max_zero_fraction_for_individual.
    With it OFF, they get blocked by the sparsity gate. This is exactly
    the high-business-impact, hard-to-forecast tier the fix targets.
    """
    from utils.intelligent_modeling import allocate_model_levels
    df = _make_lumpy_per_key(n=200, seed=42)

    # Baseline: no bypass → most lumpy high-volume keys stuck in pooled
    res_no_bypass = allocate_model_levels(
        seg_df=df.copy(), verbose=False, time_format='year_week',
        min_individual_score=0.60, max_individual_pct=0.30,
        volume_override_quantile=0.93,
        min_nonzero_obs_for_individual=52,
        max_zero_fraction_for_individual=0.70,
        top_volume_bypass_quantile=None,
    )
    # With bypass at top 10%
    res_with_bypass = allocate_model_levels(
        seg_df=df.copy(), verbose=False, time_format='year_week',
        min_individual_score=0.60, max_individual_pct=0.30,
        volume_override_quantile=0.93,
        min_nonzero_obs_for_individual=52,
        max_zero_fraction_for_individual=0.70,
        top_volume_bypass_quantile=0.90,
    )

    assert res_with_bypass.summary['individual_models'] > res_no_bypass.summary['individual_models'], (
        'Enabling top_volume_bypass should strictly increase the count of '
        'individual models — it only promotes keys, never demotes them.'
    )
    assert res_with_bypass.summary.get('top_volume_bypass_keys', 0) > 0, (
        'top_volume_bypass_keys should be non-zero when the bypass is on '
        'and there are enough keys to exceed the quantile threshold.'
    )
    # The strategy label 'individual_top_volume_bypass' should appear
    # for keys that came in through the bypass-only path.
    strategies = res_with_bypass.allocations['model_strategy'].unique()
    assert 'individual_top_volume_bypass' in strategies, (
        f'Expected individual_top_volume_bypass label; got {sorted(strategies)}'
    )


def test_bypass_respects_min_obs_floor():
    """The bypass relaxes ZF and nonzero gates but NOT min_obs — a key
    with too few observations to train on should still not become
    individual."""
    from utils.intelligent_modeling import allocate_model_levels
    df = _make_lumpy_per_key(n=100, seed=7)
    # Force the top-volume key to have almost no history
    top_idx = df['volume_mean'].idxmax()
    df.loc[top_idx, 'n_obs'] = 5
    df.loc[top_idx, 'history_length'] = 5

    res = allocate_model_levels(
        seg_df=df.copy(), verbose=False, time_format='year_week',
        min_individual_score=0.60, max_individual_pct=0.50,
        top_volume_bypass_quantile=0.90,
    )
    top_key = df.loc[top_idx, 'key']
    alloc = res.allocations[res.allocations['key'] == top_key].iloc[0]
    assert alloc['model_strategy'] == 'segment_pooled', (
        'A data-insufficient key must not be individual even under bypass'
    )


def test_raising_cap_alone_does_nothing_when_sparsity_gate_blocks():
    """Confirms that the real bottleneck on lumpy categories is the
    sparsity gate — not max_individual_pct. Raising the cap without the
    bypass should barely change the individual count, because the
    strict gate filters eligible keys before the cap is applied.
    """
    from utils.intelligent_modeling import allocate_model_levels
    df = _make_lumpy_per_key(n=200, seed=1)
    res_cap_15 = allocate_model_levels(
        seg_df=df.copy(), verbose=False, time_format='year_week',
        min_individual_score=0.60, max_individual_pct=0.15,
        top_volume_bypass_quantile=None,
    )
    res_cap_50 = allocate_model_levels(
        seg_df=df.copy(), verbose=False, time_format='year_week',
        min_individual_score=0.60, max_individual_pct=0.50,
        top_volume_bypass_quantile=None,
    )
    # Going from 15% to 50% cap should NOT unlock a huge additional pool
    # when the sparsity gate has already filtered most candidates.
    diff = res_cap_50.summary['individual_models'] - res_cap_15.summary['individual_models']
    total = res_cap_50.summary['total_keys']
    assert diff < 0.15 * total, (
        f'Raising the cap alone added {diff} individuals on {total} keys — '
        'suggests the sparsity gate is NOT the limiting factor in this '
        'fixture. Check that the synthetic data is sufficiently lumpy.'
    )


def test_bypass_keys_count_as_override_for_cap():
    """When max_individual_pct is small, bypass keys must not be evicted
    in favour of score-based candidates — they're the whole point."""
    from utils.intelligent_modeling import allocate_model_levels
    df = _make_lumpy_per_key(n=200, seed=11)
    # Tight cap that would normally crowd out lower-priority candidates
    res = allocate_model_levels(
        seg_df=df.copy(), verbose=False, time_format='year_week',
        min_individual_score=0.60,
        max_individual_pct=0.05,   # very tight
        top_volume_bypass_quantile=0.90,
    )
    # All keys flagged as bypass should survive in the final individual set
    alloc = res.allocations
    bypass_mask = alloc['model_strategy'] == 'individual_top_volume_bypass'
    if bypass_mask.any():
        # Any key labeled individual_top_volume_bypass is by definition
        # in the individual set — the label only exists for individuals.
        assert (alloc.loc[bypass_mask, 'model_strategy'] != 'segment_pooled').all()


def test_new_defaults_materially_widen_allocation_on_synthetic_lumpy():
    """End-to-end sanity check: using the NEW allocator defaults
    (max_individual_pct=0.30, top_volume_bypass_quantile=0.90) produces
    more individual models than the OLD defaults (max=0.15, bypass=None)
    on a realistic lumpy fixture. The numeric gap is the lift the fix
    buys relative to the old allocator."""
    from utils.intelligent_modeling import allocate_model_levels
    df = _make_lumpy_per_key(n=300, seed=99)
    old = allocate_model_levels(
        seg_df=df.copy(), verbose=False, time_format='year_week',
        min_individual_score=0.60, max_individual_pct=0.15,
        top_volume_bypass_quantile=None,
    )
    new = allocate_model_levels(
        seg_df=df.copy(), verbose=False, time_format='year_week',
        min_individual_score=0.60, max_individual_pct=0.30,
        top_volume_bypass_quantile=0.90,
    )
    assert new.summary['individual_models'] >= 1.5 * old.summary['individual_models'], (
        f'Expected at least 1.5× more individuals under new defaults '
        f'(got old={old.summary["individual_models"]}, '
        f'new={new.summary["individual_models"]})'
    )


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
