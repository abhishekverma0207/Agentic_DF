#!/usr/bin/env python
"""
Generic entry point for the two-mode stacked-DIQ pipeline. Market-agnostic —
UK vs DE is a pure ``--market`` switch (everything resolves from
``config/config_{market}_base_local.yaml``).

  FIT  — estimate per-category ensemble weights from the configured rolling-origin
         training snapshots and write ``config/{market}_ensemble_weights.yaml``:
             python run_stack.py --mode fit  --market de
  LIVE — apply the frozen weights on the configured live snapshot (production-style
         forward forecast), delegating to the unchanged ``run_diq_forecast``:
             python run_stack.py --mode live --market de

Both modes read a ``weight_fit:`` block from the market config. LIVE does NOT touch
the weights — it just consumes whatever ``{market}_ensemble_weights.yaml`` holds, so
running FIT then LIVE reproduces the production behaviour.
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stacked-DIQ FIT / LIVE runner (generic, config-driven)")
    ap.add_argument("--mode", choices=["fit", "live"], required=True)
    ap.add_argument("--market", required=True, help="uk | de | ... (selects config_{market}_base_local.yaml)")
    ap.add_argument("--config-root", default="config")
    ap.add_argument("--out", default=None, help="FIT: weights output path (default config/{market}_ensemble_weights.yaml)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                        format="%(asctime)s %(levelname).1s %(name)s | %(message)s")

    if args.mode == "fit":
        from utils.weight_fit import fit_weights
        out = fit_weights(args.market, config_root=args.config_root, out_path=args.out)
        print(f"[run_stack] FIT complete -> {out}")
        return

    # LIVE — derive params from the weight_fit block and delegate to the unchanged runner
    from utils.diq_runner import run_diq_forecast
    from utils.config_loader import load_market_config
    cfg = load_market_config(args.market, args.config_root)   # prefers .py, falls back to .yaml
    mio = cfg.get("market_io", {})
    wf = cfg["weight_fit"]
    snap = wf["live_snapshot"]
    parent_dir = wf["live_parent_dir"]
    cats = wf.get("in_scope_categories")     # None => engine routes all available
    stacked = run_diq_forecast(
        parent_dir=parent_dir,
        history_till=wf.get("live_history_till"),
        snapshot_week=snap,
        category_list=cats or [],
        out_path=wf.get("live_out_path"),
        market=args.market,
        config_root=args.config_root,
        enable_uk_engine=True,
        uk_route_categories=cats,
        uk_lego_source=f"{parent_dir.rstrip('/')}/{mio.get('lego_subdir')}",
    )
    print(f"[run_stack] LIVE complete: {len(stacked)} rows, snapshot={snap}")


if __name__ == "__main__":
    main()
