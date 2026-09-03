"""segment_engine: guardrail keep/snap, the calibrate path, blend application, weight round-trip."""
import numpy as np
import pandas as pd

from utils import segment_engine as se


def _w(fitted, cat, seg):
    return fitted[(cat, seg)]["weights"]


def test_guardrail_keeps_stack_where_it_beats_lego(synth_components):
    fitted = se.fit_segment_weights(synth_components, shrink=False)
    # 'good': s==c==actual, rf over-forecasts -> stack must carry real weight on s/c
    wg = _w(fitted, "A", "good")
    assert wg[0] + wg[1] > 0.5, f"expected stack weight on s/c for 'good', got {wg}"
    # 'legobest': rf==actual -> optimum is ~pure LEGO
    wl = _w(fitted, "A", "legobest")
    assert wl[2] > 0.9 and wl[0] + wl[1] < 0.1, f"expected ~LEGO for 'legobest', got {wl}"


def test_low_volume_floors_to_lego(synth_components):
    import os
    os.environ["SEG_MIN_VOLUME"] = "10_000_000"        # force every segment below the floor
    try:
        fitted = se.fit_segment_weights(synth_components, shrink=True)
        assert all(v["weights"] == [0.0, 0.0, 1.0] for v in fitted.values())
    finally:
        os.environ["SEG_MIN_VOLUME"] = "0"


def test_calibrate_path_runs_and_is_well_formed(synth_components):
    fitted = se.fit_segment_weights(synth_components, calibrate=True)
    assert fitted, "calibrate path returned no groups"
    for (cat, seg), v in fitted.items():
        assert v["use"] in ("CAL", "LEGO")
        assert len(v["weights"]) == 3 and all(w >= 0 for w in v["weights"])
        assert {"acc_lego", "acc_stack", "bias_lego", "bias_stack"} <= set(v)


def test_blend_with_weights_and_default_fallback():
    comp = pd.DataFrame({
        "category": ["A", "A", "A"], "segment": ["good", "good", "unknown"],
        "s": [10.0, 20.0, 5.0], "c": [10.0, 20.0, 5.0], "rf": [8.0, 8.0, 8.0],
    })
    weights = {("A", "good"): [1.0, 0.0, 0.0]}        # use s only
    default = [0.0, 0.0, 1.0]                          # unknown -> LEGO
    out = np.asarray(se.blend_with_weights(comp, weights, default=default), dtype=float)
    assert out[0] == 10.0 and out[1] == 20.0          # matched -> s
    assert out[2] == 8.0                              # unknown (cat,seg) -> default -> rf
    assert (out >= 0).all()                           # clipped non-negative


def test_load_segment_weights_from_dict_round_trip():
    d = {"mode": "segment", "default_weights": [0.0, 0.0, 1.0],
         "segment_weights": {"A": {"good": [0.6, 0.1, 0.3], "_default": [0.0, 0.0, 1.0]}}}
    weights, default = se.load_segment_weights(d)
    assert default == [0.0, 0.0, 1.0]
    assert weights[("A", "good")] == [0.6, 0.1, 0.3]
