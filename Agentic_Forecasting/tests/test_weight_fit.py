"""weight_fit.fit_per_category_weights: asymmetric-WAPE optimizer + LEGO guardrail at GTIN grain."""
from utils.weight_fit import fit_per_category_weights


def _fit(df, **kw):
    d = df.assign(g=df["s"], l=df["c"])           # engine expects g (=s) and l (=c)
    base = dict(grain="gtin", objective="wape_asym", bias_tolerance=0.02,
                lego_col="rf", gtin_col="cs_gtin", week_col="year_week")
    base.update(kw)
    return fit_per_category_weights(d, cat_col="segment", **base)


def test_guardrail_on_keeps_winner_snaps_loser(synth_components):
    fitted = _fit(synth_components, guardrail=True)
    good, lego = fitted["good"], fitted["legobest"]
    assert good["use"] == "STACK" and good["acc_stack"] >= good["acc_lego"]
    assert good["weights"][0] + good["weights"][1] > 0.5         # real stack
    assert lego["weights"][2] > 0.9 and lego["weights"][0] + lego["weights"][1] < 0.1   # ~LEGO


def test_guardrail_off_keeps_raw_coefficients(synth_components):
    fitted = _fit(synth_components, guardrail=False)
    # with guardrail off no group is snapped to LEGO by the guardrail (use != 'LEGO')
    assert all(v["use"] != "LEGO" for v in fitted.values())


def test_acc_margin_hardens_guardrail(synth_components):
    # a huge margin makes even the winner fail the bias/acc test -> falls back to LEGO
    fitted = _fit(synth_components, guardrail=True, acc_margin=0.99)
    assert fitted["good"]["weights"] == [0.0, 0.0, 1.0]


def test_metrics_present_and_sane(synth_components):
    fitted = _fit(synth_components, guardrail=True)
    for v in fitted.values():
        assert 0.0 <= v["acc_lego"] <= 1.0 and v["acc_stack"] <= 1.0
        assert len(v["weights"]) == 3 and all(w >= 0 for w in v["weights"])
