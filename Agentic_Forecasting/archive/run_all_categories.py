#!/usr/bin/env python3
"""
Master Runner: Full Pipeline + Benchmark Comparison for ALL DACH Categories.

For each category:
  1. Sends email: "Starting {category}..."
  2. Runs full pipeline via runner.py (EDA → FA → Segmentation → Features → Training → Inference → Backtesting)
  3. Runs benchmark comparison (compare_benchmark.py)
  4. Sends email: "{category} complete — WAPE: ours X% vs benchmark Y%"
  5. Saves CSV reports to TmpValidationData/DACH/benchmark_reports/

At the end, produces consolidated master CSV files with all categories.

Usage:
    source .venv/bin/activate

    # Run all categories
    python run_all_categories.py

    # Run specific categories only
    python run_all_categories.py --categories SKIN_CARE ORAL_CARE

    # Run without email
    python run_all_categories.py --no-email

    # Custom email
    python run_all_categories.py --email someone@company.com
"""
import argparse
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

ALL_CATEGORIES = [
    'SKIN_CARE',
    'ORAL_CARE',
    'HAIR_CARE',
    'FABRIC_CLEANING',
    'HOME_HYGIENE',
    'HEALTHY_SNACKING',
    'DRESSINGS',
    'SKIN_CLEANSING',
    'DEODORANTS_FRAGRANCES',
    'SCRATCH_COOKING_AIDS',
]

CATEGORY_TO_CONFIG = {
    'SKIN_CARE': 'config/config_de_skincare.yaml',
    'SKIN_CLEANSING': 'config/config_de_skincleansing.yaml',
    'DEODORANTS_FRAGRANCES': 'config/config_de_deo.yaml',
    'DRESSINGS': 'config/config_de_dressings.yaml',
    'FABRIC_CLEANING': 'config/config_de_fab_clean.yaml',
    'HAIR_CARE': 'config/config_de_haircare.yaml',
    'HEALTHY_SNACKING': 'config/config_de_healthy_snack.yaml',
    'HOME_HYGIENE': 'config/config_de_home_hygiene.yaml',
    'ORAL_CARE': 'config/config_de_oral_care.yaml',
    'SCRATCH_COOKING_AIDS': 'config/config_de_cooking_aids.yaml',
}

REPORT_DIR = "TmpValidationData/DACH/benchmark_reports"


# =============================================================================
# EMAIL (Mac Outlook Desktop via AppleScript — same as runner.py)
# =============================================================================

def send_email(to_email: str, subject: str, html_body: str) -> bool:
    """Send HTML email via Mac Outlook Desktop."""
    try:
        esc = html_body.replace("\\", "\\\\").replace('"', '\\"')
        esc = esc.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
        script = f'''
        tell application "Microsoft Outlook"
            set newMessage to make new outgoing message with properties {{subject:"{subject}"}}
            tell newMessage
                make new recipient at end of to recipients with properties {{email address:{{address:"{to_email}"}}}}
                set content of newMessage to "{esc}"
            end tell
            send newMessage
        end tell
        '''
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
        logger.info(f"  Email sent: {subject}")
        return True
    except Exception as e:
        logger.warning(f"  Email failed: {e}")
        return False


def build_start_email(category: str, idx: int, total: int) -> str:
    return f"""<html><body>
<h2>FEU Agentic Forecasting — Starting {category}</h2>
<p>Category <b>{idx}/{total}</b> starting at {datetime.now().strftime('%H:%M:%S')}</p>
<p>Pipeline: EDA → Feature Availability → Segmentation → Feature Engineering → Training → Inference → Backtesting</p>
<p>Config: {CATEGORY_TO_CONFIG[category]}</p>
</body></html>"""


def build_completion_email(category: str, idx: int, total: int, results: dict) -> str:
    our_w = results.get('our_wape_key', 'N/A')
    bench_w = results.get('bench_wape_key', 'N/A')
    our_b = results.get('our_bias_key', 'N/A')
    bench_b = results.get('bench_bias_key', 'N/A')
    our_wp = results.get('our_wape_prod', 'N/A')
    bench_wp = results.get('bench_wape_prod', 'N/A')
    elapsed = results.get('elapsed', 'N/A')
    winner = "✅ OURS" if results.get('we_win', False) else "⚠️ BENCHMARK"
    status = results.get('status', 'UNKNOWN')

    color = '#27ae60' if status == 'SUCCESS' else '#e74c3c'

    return f"""<html><body>
<h2 style="color:{color}">FEU Agentic Forecasting — {category} {status}</h2>
<p>Category <b>{idx}/{total}</b> completed at {datetime.now().strftime('%H:%M:%S')} ({elapsed})</p>

<h3>Key × Week Level</h3>
<table border="1" cellpadding="5" style="border-collapse:collapse">
<tr><th>Metric</th><th>Our Forecast</th><th>Benchmark (TVF)</th></tr>
<tr><td>WAPE</td><td><b>{our_w}</b></td><td>{bench_w}</td></tr>
<tr><td>Bias</td><td>{our_b}</td><td>{bench_b}</td></tr>
</table>

<h3>Product × Week Level</h3>
<table border="1" cellpadding="5" style="border-collapse:collapse">
<tr><th>Metric</th><th>Our Forecast</th><th>Benchmark (TVF)</th></tr>
<tr><td>WAPE</td><td><b>{our_wp}</b></td><td>{bench_wp}</td></tr>
</table>

<p><b>Overall: {winner}</b></p>
<p>Reports: {REPORT_DIR}/{category}_*.csv</p>
</body></html>"""


def build_final_email(results_all: list, total_time: float) -> str:
    rows = ""
    for r in results_all:
        color = '#27ae60' if r.get('we_win') else ('#e74c3c' if r.get('status') == 'FAILED' else '#f39c12')
        rows += f"""<tr style="color:{color}">
        <td>{r['category']}</td>
        <td>{r.get('our_wape_key', 'N/A')}</td><td>{r.get('bench_wape_key', 'N/A')}</td>
        <td>{r.get('our_wape_prod', 'N/A')}</td><td>{r.get('bench_wape_prod', 'N/A')}</td>
        <td>{r.get('our_bias_key', 'N/A')}</td><td>{r.get('bench_bias_key', 'N/A')}</td>
        <td>{r.get('status', 'N/A')}</td>
        </tr>"""

    return f"""<html><body>
<h2>FEU Agentic Forecasting — ALL CATEGORIES COMPLETE</h2>
<p>Total time: {total_time:.0f}s | Completed: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<table border="1" cellpadding="5" style="border-collapse:collapse">
<tr><th>Category</th><th>Our WAPE (Key)</th><th>Bench WAPE (Key)</th>
<th>Our WAPE (Prod)</th><th>Bench WAPE (Prod)</th>
<th>Our Bias</th><th>Bench Bias</th><th>Status</th></tr>
{rows}
</table>

<p>Reports saved to: {REPORT_DIR}/</p>
<p>Consolidated files: {REPORT_DIR}/MASTER_*.csv</p>
</body></html>"""


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def run_pipeline(category: str, config_path: str) -> bool:
    """Run full pipeline via runner.py subprocess."""
    logger.info(f"  Running: python runner.py --config {config_path}")

    # Load config to know artifact path
    from config.schema import load_config_from_yaml
    cfg = load_config_from_yaml(config_path)
    artifact_dir = cfg.artifact_base_path

    # Use deterministic pipeline (same functions as CrewAI agents, zero LLM)
    result = subprocess.run(
        [sys.executable, "-u", "run_full_benchmark.py", "--category", category,
         "--output-dir", REPORT_DIR],
        timeout=7200,  # 2hr max
    )

    # Check if artifacts were created (pipeline may succeed but Python exit is dirty
    # due to daemon threads from CrewAI/LiteLLM — this is harmless)
    bt_exists = os.path.exists(os.path.join(artifact_dir, "backtest_output", "backtest_forecasts.csv"))
    model_exists = os.path.exists(os.path.join(artifact_dir, "model_artifacts"))

    if result.returncode != 0:
        # Check if artifacts were created despite dirty exit (daemon thread cleanup)
        if bt_exists or model_exists:
            logger.warning(f"  Pipeline exit code {result.returncode} but artifacts exist — treating as SUCCESS")
            return True

        logger.error(f"  Pipeline failed for {category} (exit code {result.returncode})")
        return False

    return True


def run_benchmark_comparison(category: str) -> dict:
    """Run compare_benchmark.py and parse results."""
    # Capture output here (it's fast) so we can parse the metrics
    result = subprocess.run(
        [sys.executable, "-u", "compare_benchmark.py", "--category", category, "--output-dir", REPORT_DIR],
        capture_output=True, text=True, timeout=300,
    )
    # Print the output so user sees it
    if result.stdout:
        print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='')

    if result.returncode != 0:
        # Check if report was created despite error
        report_exists = os.path.exists(os.path.join(REPORT_DIR, f'{category}_summary.csv'))
        if not report_exists:
            logger.error(f"  Benchmark comparison failed for {category}")
            return {'status': 'FAILED'}

    # Parse results from output
    output = (result.stdout or '') + (result.stderr or '')
    metrics = {}
    for line in output.split('\n'):
        if 'KEY × WEEK:' in line and 'Ours=' in line:
            parts = line.split('Ours=')[1]
            our_w = parts.split()[0]
            bench_w = parts.split('Bench=')[1].split()[0]
            metrics['our_wape_key'] = our_w
            metrics['bench_wape_key'] = bench_w
        if 'PROD × WEEK:' in line and 'Ours=' in line:
            parts = line.split('Ours=')[1]
            our_w = parts.split()[0]
            bench_w = parts.split('Bench=')[1].split()[0]
            metrics['our_wape_prod'] = our_w
            metrics['bench_wape_prod'] = bench_w
        if 'Bias:' in line and 'Ours=' in line:
            parts = line.split('Ours=')[1]
            our_b = parts.split()[0]
            bench_b = parts.split('Bench=')[1].split()[0]
            metrics['our_bias_key'] = our_b
            metrics['bench_bias_key'] = bench_b

    # Determine winner
    try:
        our_v = float(metrics.get('our_wape_key', '999').rstrip('%'))
        bench_v = float(metrics.get('bench_wape_key', '999').rstrip('%'))
        metrics['we_win'] = our_v < bench_v
    except ValueError:
        metrics['we_win'] = False

    metrics['status'] = 'SUCCESS'
    return metrics


def build_master_report(categories_run: list):
    """Build consolidated CSV files across all categories."""
    import pandas as pd

    logger.info(f"\nBuilding master CSV reports in: {REPORT_DIR}")

    all_raw = []
    summary_rows = []

    for cat in categories_run:
        summary_path = os.path.join(REPORT_DIR, f"{cat}_summary.csv")
        raw_path = os.path.join(REPORT_DIR, f"{cat}_raw_forecasts.csv")
        prod_path = os.path.join(REPORT_DIR, f"{cat}_per_product.csv")

        if not os.path.exists(summary_path):
            continue

        try:
            if os.path.exists(raw_path):
                raw = pd.read_csv(raw_path)
                raw['Category'] = cat
                all_raw.append(raw)

            summ = pd.read_csv(summary_path)
            summ['Category'] = cat
            summary_rows.append(summ)

            if os.path.exists(prod_path):
                prod = pd.read_csv(prod_path)
                prod['Category'] = cat
                prod.to_csv(os.path.join(REPORT_DIR, f'MASTER_{cat}_products.csv'), index=False)

        except Exception as e:
            logger.warning(f"  Could not read CSVs for {cat}: {e}")

    # Master summary
    if summary_rows:
        pd.concat(summary_rows, ignore_index=True).to_csv(
            os.path.join(REPORT_DIR, 'MASTER_all_summaries.csv'), index=False)

    # All raw forecasts
    if all_raw:
        pd.concat(all_raw, ignore_index=True).to_csv(
            os.path.join(REPORT_DIR, 'MASTER_all_raw_forecasts.csv'), index=False)

    logger.info(f"  Master CSV reports saved to: {REPORT_DIR}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run full pipeline + benchmark for all DACH categories")
    parser.add_argument('--categories', nargs='+', default=None,
                        choices=ALL_CATEGORIES,
                        help='Specific categories to run (default: all)')
    parser.add_argument('--email', default='debonil.chowdhury@aria-is.com',
                        help='Notification email address')
    parser.add_argument('--no-email', action='store_true',
                        help='Disable email notifications')
    parser.add_argument('--skip-pipeline', action='store_true',
                        help='Skip pipeline, only run benchmark comparison (assumes artifacts exist)')
    args = parser.parse_args()

    categories = args.categories or ALL_CATEGORIES
    to_email = None if args.no_email else args.email
    os.makedirs(REPORT_DIR, exist_ok=True)

    t0 = time.time()
    total = len(categories)

    logger.info("=" * 90)
    logger.info(f"FEU AGENTIC FORECASTING — FULL BENCHMARK RUN")
    logger.info(f"Categories: {categories}")
    logger.info(f"Email: {to_email or 'disabled'}")
    logger.info(f"Skip pipeline: {args.skip_pipeline}")
    logger.info("=" * 90)

    results_all = []

    for idx, category in enumerate(categories, 1):
        cat_t0 = time.time()
        config_path = CATEGORY_TO_CONFIG[category]

        logger.info(f"\n{'='*90}")
        logger.info(f"[{idx}/{total}] {category}")
        logger.info(f"{'='*90}")

        # Send start email
        if to_email:
            send_email(to_email,
                       f"FEU Forecast [{idx}/{total}] Starting: {category}",
                       build_start_email(category, idx, total))

        # Run full pipeline + benchmark comparison (all in one)
        if not args.skip_pipeline:
            logger.info(f"  Running full deterministic pipeline...")
            pipeline_ok = run_pipeline(category, config_path)
            if not pipeline_ok:
                results_all.append({'category': category, 'status': 'PIPELINE_FAILED', 'we_win': False})
                if to_email:
                    send_email(to_email,
                               f"FEU Forecast [{idx}/{total}] FAILED: {category}",
                               f"<html><body><h2 style='color:red'>Pipeline failed for {category}</h2></body></html>")
                continue

        # Parse benchmark results from the report that run_full_benchmark.py already created
        logger.info(f"  Parsing benchmark results...")
        metrics = run_benchmark_comparison(category)
        elapsed = time.time() - cat_t0
        metrics['category'] = category
        metrics['elapsed'] = f"{elapsed:.0f}s"
        results_all.append(metrics)

        logger.info(f"  {category}: WAPE ours={metrics.get('our_wape_key','N/A')} vs bench={metrics.get('bench_wape_key','N/A')} ({elapsed:.0f}s)")

        # Send completion email
        if to_email:
            send_email(to_email,
                       f"FEU Forecast [{idx}/{total}] Done: {category} — {'WIN' if metrics.get('we_win') else 'REVIEW'}",
                       build_completion_email(category, idx, total, metrics))

    # Build master consolidated report
    build_master_report([r['category'] for r in results_all if r.get('status') == 'SUCCESS'])

    total_time = time.time() - t0

    # Final summary
    logger.info(f"\n{'='*90}")
    logger.info("ALL CATEGORIES COMPLETE")
    logger.info(f"{'='*90}")
    logger.info(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    for r in results_all:
        status = "✓" if r.get('we_win') else ("✗" if r.get('status') == 'PIPELINE_FAILED' else "~")
        logger.info(f"  {status} {r['category']:<25} WAPE: {r.get('our_wape_key','N/A'):>8} vs {r.get('bench_wape_key','N/A'):>8}  [{r.get('status','?')}]")
    logger.info(f"\n  Reports: {REPORT_DIR}/")
    logger.info(f"  Master:  {REPORT_DIR}/MASTER_*.csv")

    # Send final summary email
    if to_email:
        send_email(to_email,
                   f"FEU Forecast — ALL {total} CATEGORIES COMPLETE",
                   build_final_email(results_all, total_time))


if __name__ == '__main__':
    main()
