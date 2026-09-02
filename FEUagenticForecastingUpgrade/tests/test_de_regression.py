"""Regression lock: the validated DE out-of-sample numbers must reproduce on the cached components.

prod guardrail  ~ +1.4pp vs LEGO @ 9/10 cats ;  E (calibrate+guardrail) ~ +5.8pp @ 9/10, clean bias.
Uses the local datacache/de (gitignored) — skipped where absent. Catches any silent engine/cleanup drift.
"""
import pandas as pd
import pytest


@pytest.mark.regression
def test_de_prod_and_calibrated_E_reproduce(de_cache_ready):
    import scripts.de_stacker_compare as C
    import scripts.de_bias_correct as B
    import scripts.de_segment_research as R

    data = C.load_all()
    train = pd.concat([data[s] for s in C.TRAIN], ignore_index=True)
    ev = data[C.EVAL]

    lego = R._score(ev.assign(blend=ev["rf"]), "blend")[0]
    assert abs(lego - 0.5647) < 0.010, f"LEGO baseline drifted: {lego:.4f}"

    # prod guardrail
    prod = C.fit_guardrail_on(train)
    e = ev.copy(); e["blend"] = R.apply_blend(e, prod)
    a_prod = R._score(e, "blend")[0]
    assert abs(a_prod - 0.5784) < 0.006, f"prod acc drifted: {a_prod:.4f}"

    # E = calibrate-then-guardrail on the lasso-hier blend
    lasso = C.fit_hier(train, 0.2, l1=True)
    r, _ = B.cal_then_guardrail(train, ev, lasso, 0.0)
    assert abs(r["acc"] - 0.6228) < 0.008, f"E acc drifted: {r['acc']:.4f}"
    assert r["dacc"] > 0.04, f"E gain over LEGO shrank: {r['dacc']:+.4f}"
    assert r["wins"] >= 8, f"E category wins dropped: {r['wins']}/{r['ncats']}"
    assert abs(r["bias"]) < 0.03, f"E bias no longer clean: {r['bias']:+.4f}"
