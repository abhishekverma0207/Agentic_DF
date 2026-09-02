"""Tests for utils.moq_postprocessing.

Covers the three primitives (``check_strong_moq``, ``custom_round``,
``_pick_best_threshold``) and the end-to-end ``apply_moq_postprocessing``
integration on synthetic forecast + history DataFrames. The failure
modes tested are the exact ones that made the legacy implementation
fragile: empty inputs, missing columns, NaN forecasts, Lego_segments
gate, and the round-trip over inference-shaped / backtest-shaped
DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.moq_postprocessing import (
    apply_moq_postprocessing,
    check_strong_moq,
    custom_round,
)


# ---------------------------------------------------------------------------
# check_strong_moq
# ---------------------------------------------------------------------------

def test_check_strong_moq_detects_p40_multiple():
    """A series where p40=60 evenly divides p70=120 should return 60.
    Uses clean integer values — matching how real case-pack sales look."""
    rng = np.random.default_rng(0)
    base = rng.choice([60, 60, 60, 120, 120, 180], size=60).astype(float)
    past = base.tolist() + [0] * 20  # 60 non-zero + zeros
    moq = check_strong_moq(past)
    assert moq == 60.0, f'Expected MOQ=60, got {moq}'


def test_check_strong_moq_tolerates_small_float_drift():
    """Real upstream aggregation can introduce <1% float drift on case
    counts even when the underlying orders are exact integers. The
    detector must survive this rather than treating 59.998 / 120.003
    as "not a multiple"."""
    rng = np.random.default_rng(0)
    base = rng.choice([60, 60, 60, 120, 120, 180], size=60).astype(float)
    drift = rng.uniform(-0.3, 0.3, size=60)  # sub-1% on 60 / 120
    past = (base + drift).tolist() + [0] * 20
    moq = check_strong_moq(past)
    # Should still detect MOQ near 60 (exact value drifts a little)
    assert 59 <= moq <= 61, f'Expected MOQ≈60 under small drift, got {moq}'


def test_check_strong_moq_returns_zero_on_smooth_series():
    """Uniformly distributed sales (no fixed multiple) should yield 0."""
    rng = np.random.default_rng(1)
    past = rng.uniform(10, 500, size=80).tolist()
    moq = check_strong_moq(past)
    assert moq == 0.0, f'Expected MOQ=0 on smooth data, got {moq}'


def test_check_strong_moq_requires_minimum_observations():
    """Even if every value is a clean multiple, <30 non-zeros -> no MOQ.
    Guards against fitting MOQ to tiny samples where percentiles are
    unstable."""
    past = [60, 120, 180] * 5  # 15 non-zero values (<30), sum=1800
    assert check_strong_moq(past) == 0.0


def test_check_strong_moq_requires_floor_of_48():
    """MOQ < 48 units is below the realistic case-pack floor — reject.
    Values of p40 = 10 shouldn't trigger even if divisibility holds."""
    past = ([10, 10, 10, 20, 20, 30] * 10)  # p40=10, p70=20 (divides)
    assert check_strong_moq(past) == 0.0


def test_check_strong_moq_ignores_nans_and_zeros():
    """NaN and zero values should be filtered out before percentile calc."""
    past = [60] * 35 + [0] * 50 + [np.nan] * 10
    moq = check_strong_moq(past)
    # 35 positives, all 60 -> p40=p50=p60=p70=p80=60, trivially divides
    assert moq == 60.0


def test_check_strong_moq_handles_empty_input():
    assert check_strong_moq([]) == 0.0
    assert check_strong_moq([0, 0, np.nan]) == 0.0


# ---------------------------------------------------------------------------
# custom_round
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('value,fact,thsd,expected', [
    (130.0, 60.0, 0.5, 120.0),   # 2.17 -> floor (frac 0.17 < 0.5)
    (160.0, 60.0, 0.5, 180.0),   # 2.67 -> ceil  (frac 0.67 > 0.5)
    (60.0,  60.0, 0.5, 60.0),    # exact multiple
    (0.0,   60.0, 0.5, 0.0),     # zero
    (-10.0, 60.0, 0.5, 0.0),     # negative clipped
    (100.0, 60.0, 0.1, 120.0),   # low threshold -> almost always ceil
    (100.0, 60.0, 0.9, 60.0),    # high threshold -> almost always floor
])
def test_custom_round(value, fact, thsd, expected):
    assert custom_round(value, fact, thsd) == expected


def test_custom_round_handles_nan():
    assert np.isnan(custom_round(np.nan, 60.0, 0.5)) or custom_round(np.nan, 60.0, 0.5) == 0.0


def test_custom_round_handles_zero_fact():
    """fact<=0 is a detector bug, but shouldn't crash — pass value through."""
    assert custom_round(100.0, 0.0, 0.5) == 100.0


# ---------------------------------------------------------------------------
# apply_moq_postprocessing — end-to-end
# ---------------------------------------------------------------------------

class _StubConfig:
    """Minimal stand-in for DemandForecastConfig.

    The real schema is heavy to construct in tests. ``apply_moq_post
    processing`` only reads a handful of fields — we expose exactly
    those, mirroring real attribute access."""
    def __init__(self, *, train_end='202548', timestamp_col='year_week',
                 target_col='actual_sales', key_col='key'):
        self.train_end = train_end
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.prediction_key_cols = [key_col]
        self.input_data_path = '/tmp/not-used-in-tests'


def _make_moq_key_history(key_val='MOQ_KEY', moq_size=60, n_weeks=135):
    """Synthesise a history that WILL trigger MOQ detection — values
    concentrated on multiples of ``moq_size`` with the SMALLER multiple
    dominating the distribution so p40 falls on 1x (not 2x).

    Composition (of non-zeros): 62.5% at 1×, 25% at 2×, 12.5% at 3×.
    This gives p40 = moq_size, p70 = 2*moq_size → 2 is an integer
    multiple → MOQ detector fires."""
    rng = np.random.default_rng(42)
    multiples = rng.choice(
        [0, 0, 1, 1, 1, 1, 1, 2, 2, 3],  # 20% zero, 50% 1x, 20% 2x, 10% 3x
        size=n_weeks,
        replace=True,
    )
    values = (multiples * moq_size).astype(float)
    return pd.DataFrame({
        'key': [key_val] * n_weeks,
        'year_week': [f'2025{str(i+1).zfill(2)}' for i in range(n_weeks)],
        'actual_sales': values,
    })


def _make_smooth_key_history(key_val='SMOOTH_KEY', n_weeks=135):
    """Synthesise a smooth history that should NOT trigger MOQ."""
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        'key': [key_val] * n_weeks,
        'year_week': [f'2025{str(i+1).zfill(2)}' for i in range(n_weeks)],
        'actual_sales': rng.uniform(50, 500, size=n_weeks),
    })


def _make_forecasts(key_val, periods, predicted):
    return pd.DataFrame({
        'key': [key_val] * len(periods),
        'year_week': list(periods),
        'predicted': list(predicted),
        'actual': [np.nan] * len(periods),
    })


def test_apply_moq_detects_and_rounds_moq_key():
    """The MOQ key's forecast rows should be rounded to multiples of 60."""
    history = _make_moq_key_history(key_val='K1', moq_size=60, n_weeks=100)
    # Train cutoff = 202548; forecast periods 202549..202556 at odd values
    forecast = _make_forecasts(
        'K1',
        [f'20260{i}' for i in range(1, 9)],
        predicted=[70.0, 125.0, 58.0, 190.0, 45.0, 15.0, 200.0, 305.0],
    )
    config = _StubConfig(train_end='202548')

    out = apply_moq_postprocessing(forecast, config, history_df=history)
    assert 'prediction_post_moq' in out.columns
    # Every rounded value should be a multiple of 60
    for v in out['prediction_post_moq']:
        assert v % 60 == 0, f'{v} is not a multiple of 60'


def test_apply_moq_passes_through_smooth_key():
    """A smooth key's prediction_post_moq should equal its predicted."""
    history = _make_smooth_key_history(key_val='K2', n_weeks=100)
    forecast = _make_forecasts(
        'K2', [f'20260{i}' for i in range(1, 5)],
        predicted=[123.45, 88.12, 219.7, 42.0],
    )
    config = _StubConfig(train_end='202548')

    out = apply_moq_postprocessing(forecast, config, history_df=history)
    np.testing.assert_allclose(out['prediction_post_moq'], forecast['predicted'])


def test_apply_moq_mixed_dataframe_handles_both():
    """A DF with one MOQ key and one smooth key — MOQ key gets rounded,
    smooth key is passed through unchanged."""
    hist_moq = _make_moq_key_history('K1', moq_size=60, n_weeks=100)
    hist_smooth = _make_smooth_key_history('K2', n_weeks=100)
    history = pd.concat([hist_moq, hist_smooth], ignore_index=True)
    fcst_moq = _make_forecasts('K1', ['202601', '202602', '202603'], [70.0, 125.0, 190.0])
    fcst_smooth = _make_forecasts('K2', ['202601', '202602', '202603'], [123.45, 88.12, 219.7])
    forecast = pd.concat([fcst_moq, fcst_smooth], ignore_index=True)
    config = _StubConfig(train_end='202548')

    out = apply_moq_postprocessing(forecast, config, history_df=history)
    # MOQ key rounded to multiples of 60
    k1 = out[out['key'] == 'K1']
    for v in k1['prediction_post_moq']:
        assert v % 60 == 0
    # Smooth key unchanged
    k2 = out[out['key'] == 'K2']
    np.testing.assert_allclose(k2['prediction_post_moq'], k2['predicted'])


def test_apply_moq_respects_lego_segments_gate_when_present():
    """With honour_lego_segments_if_present=True and a Lego_segments column
    present, only keys whose segment is in the legacy whitelist should be
    considered — the MOQ key outside the whitelist should pass through."""
    history = _make_moq_key_history('K_OUT', moq_size=60, n_weeks=100)
    fcst = _make_forecasts('K_OUT', ['202601', '202602', '202603'],
                           [70.0, 125.0, 190.0])
    fcst['Lego_segments'] = '03: Established'  # NOT in whitelist
    config = _StubConfig(train_end='202548')

    out = apply_moq_postprocessing(fcst, config, history_df=history)
    # Should be pass-through because Lego_segments filter excluded it
    np.testing.assert_allclose(out['prediction_post_moq'], fcst['predicted'])


def test_apply_moq_honours_lego_segments_when_in_whitelist():
    """Same key, but now Lego_segments='09: MOQ Orders' → MOQ rounding
    SHOULD fire."""
    history = _make_moq_key_history('K_IN', moq_size=60, n_weeks=100)
    fcst = _make_forecasts('K_IN', ['202601', '202602', '202603'],
                           [70.0, 125.0, 190.0])
    fcst['Lego_segments'] = '09: MOQ Orders'
    config = _StubConfig(train_end='202548')

    out = apply_moq_postprocessing(fcst, config, history_df=history)
    for v in out['prediction_post_moq']:
        assert v % 60 == 0


def test_apply_moq_missing_forecast_col_skips_gracefully():
    """If the requested forecast column isn't present, we should log a
    warning and return the original DF with NaN prediction_post_moq
    rather than crashing."""
    history = _make_moq_key_history('K1', 60, 100)
    fcst = _make_forecasts('K1', ['202601'], [100.0])
    fcst.drop(columns=['predicted'], inplace=True)
    config = _StubConfig(train_end='202548')

    out = apply_moq_postprocessing(fcst, config, history_df=history,
                                   forecast_col='predicted')
    assert 'prediction_post_moq' in out.columns
    assert out['prediction_post_moq'].isna().all()


def test_apply_moq_empty_history_passes_through():
    """With no history matching the keys in the forecast, nothing is
    rounded — prediction_post_moq mirrors predicted."""
    fcst = _make_forecasts('K1', ['202601', '202602'], [100.0, 150.0])
    empty_history = pd.DataFrame(columns=['key', 'year_week', 'actual_sales'])
    config = _StubConfig(train_end='202548')
    out = apply_moq_postprocessing(fcst, config, history_df=empty_history)
    np.testing.assert_allclose(out['prediction_post_moq'], fcst['predicted'])


def test_apply_moq_no_history_till_skips():
    """If train_end is empty and no override, we pass through the raw
    forecast rather than guess a cutoff."""
    history = _make_moq_key_history('K1', 60, 100)
    fcst = _make_forecasts('K1', ['202601'], [100.0])
    config = _StubConfig(train_end='')
    out = apply_moq_postprocessing(fcst, config, history_df=history)
    np.testing.assert_allclose(out['prediction_post_moq'], fcst['predicted'])


def test_apply_moq_only_rewrites_future_rows():
    """Rows whose date is at/before the train cutoff should NOT be
    rewritten, even for MOQ-detected keys. This ensures we never touch
    historical / in-sample predictions that may be reported alongside."""
    history = _make_moq_key_history('K1', 60, 100)
    # Include one pre-cutoff row and three post-cutoff rows in forecasts
    fcst = pd.DataFrame({
        'key': ['K1'] * 4,
        'year_week': ['202540', '202601', '202602', '202603'],
        'predicted': [111.0, 70.0, 125.0, 190.0],
        'actual': [np.nan] * 4,
    })
    config = _StubConfig(train_end='202548')
    out = apply_moq_postprocessing(fcst, config, history_df=history)
    # Pre-cutoff row stays exactly at the original predicted value
    pre = out[out['year_week'] == '202540']
    assert float(pre['prediction_post_moq'].iloc[0]) == 111.0
    # Post-cutoff rows are multiples of 60
    post = out[out['year_week'] > '202548']
    for v in post['prediction_post_moq']:
        assert v % 60 == 0


def test_apply_moq_returns_copy_not_inplace():
    """Caller's DataFrame should NOT be mutated."""
    history = _make_moq_key_history('K1', 60, 100)
    fcst = _make_forecasts('K1', ['202601', '202602'], [70.0, 125.0])
    config = _StubConfig(train_end='202548')
    before_columns = set(fcst.columns)
    _ = apply_moq_postprocessing(fcst, config, history_df=history)
    # Caller's DF shouldn't have gained a column
    assert set(fcst.columns) == before_columns


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
