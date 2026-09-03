#!/usr/bin/env python3
"""
FEU-Agentic-Forecasting Rolling-Origin Backtesting Pipeline Runner

This script performs rolling-origin (walk-forward) backtesting by iteratively
running the inference pipeline, each time rolling the training cutoff forward
by one week into the test period.

Rolling-Origin Backtesting Methodology:
- Origin 0: Train up to val_end, forecast from test_start
- Origin 1: Train up to test_start (inclusive), forecast from test_start+1
- Origin 2: Train up to test_start+1 (inclusive), forecast from test_start+2
- ... continues until no more viable forecast periods remain

This produces multi-origin forecasts that enable:
1. Assessment of forecast accuracy at different forecast horizons
2. Evaluation of model stability over time
3. Generation of more robust accuracy metrics
4. Analysis of forecast degradation as horizon increases

Usage:
    python run_backtesting.py --config config/config.yaml
    python run_backtesting.py --config config/config.yaml --output-dir ./backtest_results
    python run_backtesting.py --config config/config.yaml --verbose

Example:
    # Standard run with default config
    python run_backtesting.py --config config/config_us_pc.yaml

    # Run with custom output directory
    python run_backtesting.py --config config/config_us_pc.yaml --output-dir ./my_backtest

    # Run with verbose logging
    python run_backtesting.py --config config/config_us_pc.yaml --verbose

    # Run with email notifications after each origin
    python run_backtesting.py --config config/config_us_pc.yaml --email
    python run_backtesting.py --config config/config_us_pc.yaml --email --to your.email@aria-is.com
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def setup_logging(verbose: bool = False, log_file: str = None) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []  # Clear existing handlers
    for handler in handlers:
        root_logger.addHandler(handler)

    # Suppress noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


# =============================================================================
# Email Notification (Mac Outlook Desktop via AppleScript)
# =============================================================================

def send_outlook_desktop_email(to_email: str, subject: str, html_body: str) -> bool:
    """
    Sends HTML email using Microsoft Outlook (Mac desktop) via AppleScript.

    Requirements:
        - Microsoft Outlook installed on Mac
        - Outlook logged in
        - Allowed to automate (macOS will prompt on first use)
    """
    try:
        # Escape HTML for AppleScript string literal
        esc = html_body.replace("\\", "\\\\").replace('"', '\\"')
        esc = esc.replace("\r\n", "\n").replace("\r", "\n")
        esc = esc.replace("\n", "\\n")

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
        logging.info(f"✉️  Email sent to {to_email}: {subject}")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Failed to send email via Outlook: {e}")
        if e.stderr:
            logging.error(f"   stderr: {e.stderr.decode()}")
        return False
    except Exception as e:
        logging.error(f"❌ Unexpected error sending email: {e}")
        return False


def send_origin_completion_email(
    to_email: Optional[str],
    origin_idx: int,
    total_origins: int,
    origin_result: Dict[str, Any],
    metrics: Dict[str, float],
) -> None:
    """Send email notification when a backtest origin completes."""
    if not to_email:
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    duration_sec = origin_result.get('elapsed_seconds', 0)
    duration_min = duration_sec / 60.0

    # Extract metrics
    wape = metrics.get('wape', 0) * 100  # Convert to percentage
    bias_pct = metrics.get('bias_pct', 0)
    mae = metrics.get('mae', 0)
    n_forecasts = metrics.get('n_forecasts', 0)

    # Determine status color based on WAPE
    if wape < 15:
        status_emoji = "🟢"
        status_color = "#4caf50"
        performance = "Excellent"
    elif wape < 25:
        status_emoji = "🟡"
        status_color = "#ff9800"
        performance = "Good"
    else:
        status_emoji = "🔴"
        status_color = "#f44336"
        performance = "Needs Improvement"

    subject = f"[FEU-Agentic-Forecasting Backtest] {status_emoji} Origin {origin_idx}/{total_origins} Complete - WAPE: {wape:.1f}%"

    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; color: #333; }}
            .header {{ background-color: {status_color}; color: white; padding: 20px; border-radius: 5px; }}
            .content {{ padding: 20px; background-color: #f5f5f5; border-radius: 5px; margin-top: 10px; }}
            .metrics-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            .metrics-table th, .metrics-table td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            .metrics-table th {{ background-color: #e3f2fd; font-weight: bold; }}
            .metric-value {{ font-size: 18px; font-weight: bold; color: {status_color}; }}
            .info-row {{ padding: 8px 0; border-bottom: 1px solid #ddd; }}
            .label {{ font-weight: bold; color: #555; }}
            .footer {{ margin-top: 20px; padding: 15px; background-color: #e3f2fd; border-radius: 5px; }}
            .progress-bar {{ background-color: #ddd; border-radius: 10px; height: 20px; margin: 10px 0; }}
            .progress-fill {{ background-color: {status_color}; height: 20px; border-radius: 10px; text-align: center; color: white; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>{status_emoji} Backtest Origin {origin_idx} Complete</h2>
            <p>Performance: {performance}</p>
        </div>
        <div class="content">
            <h3>Progress</h3>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {(origin_idx/total_origins)*100:.0f}%;">
                    {origin_idx}/{total_origins}
                </div>
            </div>

            <h3>Origin Details</h3>
            <div class="info-row">
                <span class="label">Origin Period:</span>
                <span>{origin_result.get('origin_period', 'N/A')}</span>
            </div>
            <div class="info-row">
                <span class="label">Train Cutoff:</span>
                <span>{origin_result.get('train_cutoff', 'N/A')}</span>
            </div>
            <div class="info-row">
                <span class="label">Forecast Window:</span>
                <span>{origin_result.get('forecast_start', 'N/A')} to {origin_result.get('forecast_end', 'N/A')}</span>
            </div>
            <div class="info-row">
                <span class="label">Duration:</span>
                <span>{duration_min:.2f} minutes</span>
            </div>

            <h3>Key Metrics</h3>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Interpretation</th>
                </tr>
                <tr>
                    <td>WAPE</td>
                    <td class="metric-value">{wape:.2f}%</td>
                    <td>Weighted Absolute Percentage Error</td>
                </tr>
                <tr>
                    <td>Bias</td>
                    <td class="metric-value">{bias_pct:+.2f}%</td>
                    <td>{'Over-forecasting' if bias_pct > 0 else 'Under-forecasting' if bias_pct < 0 else 'Balanced'}</td>
                </tr>
                <tr>
                    <td>MAE</td>
                    <td class="metric-value">{mae:.2f}</td>
                    <td>Mean Absolute Error</td>
                </tr>
                <tr>
                    <td>Forecasts</td>
                    <td class="metric-value">{n_forecasts:,}</td>
                    <td>Number of predictions</td>
                </tr>
            </table>
        </div>
        <div class="footer">
            <p><strong>Timestamp:</strong> {now}</p>
            <p>{total_origins - origin_idx} origins remaining in backtest</p>
        </div>
    </body>
    </html>
    """

    send_outlook_desktop_email(to_email, subject, html_body)


def send_backtest_start_email(
    to_email: Optional[str],
    total_origins: int,
    config,
) -> None:
    """Send email notification when backtest starts."""
    if not to_email:
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    subject = f"[FEU-Agentic-Forecasting Backtest] 🚀 Started - {total_origins} Origins"

    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; color: #333; }}
            .header {{ background-color: #1976d2; color: white; padding: 20px; border-radius: 5px; }}
            .content {{ padding: 20px; background-color: #f5f5f5; border-radius: 5px; margin-top: 10px; }}
            .info-row {{ padding: 8px 0; border-bottom: 1px solid #ddd; }}
            .label {{ font-weight: bold; color: #555; }}
            .value {{ color: #1976d2; }}
            .footer {{ margin-top: 20px; padding: 15px; background-color: #e3f2fd; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🚀 Rolling-Origin Backtest Started</h2>
        </div>
        <div class="content">
            <h3>Backtest Configuration</h3>
            <div class="info-row">
                <span class="label">Total Origins:</span>
                <span class="value">{total_origins}</span>
            </div>
            <div class="info-row">
                <span class="label">Test Period:</span>
                <span class="value">{config.test_start} to {config.test_end}</span>
            </div>
            <div class="info-row">
                <span class="label">Forecast Horizon:</span>
                <span class="value">{config.forecast_horizon}</span>
            </div>
            <div class="info-row">
                <span class="label">Validation End:</span>
                <span class="value">{config.val_end}</span>
            </div>
            <div class="info-row">
                <span class="label">Input Data:</span>
                <span class="value">{config.input_data_path}</span>
            </div>
            <div class="info-row">
                <span class="label">Artifacts:</span>
                <span class="value">{config.artifact_base_path}</span>
            </div>
        </div>
        <div class="footer">
            <p><strong>Start Time:</strong> {now}</p>
            <p>You will receive email updates after each origin completes.</p>
        </div>
    </body>
    </html>
    """

    send_outlook_desktop_email(to_email, subject, html_body)


def send_backtest_completion_email(
    to_email: Optional[str],
    result,
    overall_metrics: Dict[str, float],
) -> None:
    """Send email notification when entire backtest completes."""
    if not to_email:
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    duration_min = result.elapsed_seconds / 60.0

    # Extract overall metrics
    wape = overall_metrics.get('wape', 0) * 100
    bias_pct = overall_metrics.get('bias_pct', 0)
    mae = overall_metrics.get('mae', 0)

    # Status color
    if wape < 15:
        status_color = "#4caf50"
    elif wape < 25:
        status_color = "#ff9800"
    else:
        status_color = "#f44336"

    subject = f"[FEU-Agentic-Forecasting Backtest] ✅ Complete - Overall WAPE: {wape:.1f}%"

    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; color: #333; }}
            .header {{ background-color: #4caf50; color: white; padding: 20px; border-radius: 5px; }}
            .content {{ padding: 20px; background-color: #f5f5f5; border-radius: 5px; margin-top: 10px; }}
            .metrics-box {{ background-color: white; padding: 20px; border-radius: 5px; border-left: 4px solid {status_color}; margin: 15px 0; }}
            .big-metric {{ font-size: 36px; font-weight: bold; color: {status_color}; }}
            .metric-label {{ font-size: 14px; color: #666; }}
            .info-row {{ padding: 8px 0; border-bottom: 1px solid #ddd; }}
            .label {{ font-weight: bold; color: #555; }}
            .footer {{ margin-top: 20px; padding: 15px; background-color: #e3f2fd; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🎉 Rolling-Origin Backtest Complete!</h2>
        </div>
        <div class="content">
            <div class="metrics-box" style="text-align: center;">
                <div class="metric-label">Overall WAPE</div>
                <div class="big-metric">{wape:.1f}%</div>
            </div>

            <h3>Summary</h3>
            <div class="info-row">
                <span class="label">Total Origins:</span>
                <span>{result.total_origins}</span>
            </div>
            <div class="info-row">
                <span class="label">Total Forecasts:</span>
                <span>{result.total_forecasts:,}</span>
            </div>
            <div class="info-row">
                <span class="label">Forecast Horizon:</span>
                <span>{result.forecast_horizon}</span>
            </div>
            <div class="info-row">
                <span class="label">Total Duration:</span>
                <span>{duration_min:.1f} minutes</span>
            </div>

            <h3>Overall Metrics</h3>
            <div class="info-row">
                <span class="label">WAPE:</span>
                <span>{wape:.2f}%</span>
            </div>
            <div class="info-row">
                <span class="label">Bias:</span>
                <span>{bias_pct:+.2f}%</span>
            </div>
            <div class="info-row">
                <span class="label">MAE:</span>
                <span>{mae:.2f}</span>
            </div>

            <h3>Output Files</h3>
            <div class="info-row">
                <span class="label">Forecasts:</span>
                <span>{result.forecasts_path}</span>
            </div>
            <div class="info-row">
                <span class="label">Metrics:</span>
                <span>{result.metrics_path}</span>
            </div>
            <div class="info-row">
                <span class="label">Summary:</span>
                <span>{result.summary_path}</span>
            </div>
        </div>
        <div class="footer">
            <p><strong>Completed:</strong> {now}</p>
        </div>
    </body>
    </html>
    """

    send_outlook_desktop_email(to_email, subject, html_body)


def create_origin_callback(to_email: Optional[str], total_origins: int) -> Callable:
    """Create a callback function to be called after each origin completes."""
    def origin_callback(origin_idx: int, origin_result: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """Callback called after each origin completes."""
        send_origin_completion_email(
            to_email=to_email,
            origin_idx=origin_idx,
            total_origins=total_origins,
            origin_result=origin_result,
            metrics=metrics,
        )
    return origin_callback


def print_banner() -> None:
    """Print startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     ██╗  ██╗ █████╗ ██████╗ ███╗   ███╗ ██████╗ ███╗   ██╗██╗ ██████╗        ║
║     ██║  ██║██╔══██╗██╔══██╗████╗ ████║██╔═══██╗████╗  ██║██║██╔═══██╗       ║
║     ███████║███████║██████╔╝██╔████╔██║██║   ██║██╔██╗ ██║██║██║   ██║       ║
║     ██╔══██║██╔══██║██╔══██╗██║╚██╔╝██║██║   ██║██║╚██╗██║██║██║▄▄ ██║       ║
║     ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║╚██████╔╝██║ ╚████║██║╚██████╔╝       ║
║     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝ ╚══▀▀═╝        ║
║                                                                               ║
║               ROLLING-ORIGIN BACKTESTING PIPELINE                             ║
║                                                                               ║
║   Walk-forward validation with multiple forecast origins                      ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_config_summary(config) -> None:
    """Print configuration summary."""
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"  Input Data:       {config.input_data_path}")
    print(f"  Artifact Base:    {config.artifact_base_path}")
    print(f"  Key Column(s):    {config.prediction_key_cols}")
    print(f"  Target Column:    {config.target_col}")
    print(f"  Date Column:      {config.timestamp_col}")
    print(f"  Forecast Horizon: {config.forecast_horizon}")
    print(f"  Validation End:   {config.val_end}")
    print(f"  Test Period:      {config.test_start} to {config.test_end}")
    print(f"  Bias Calibration: {'Enabled' if config.design.apply_bias_calibration else 'Disabled'}")
    print("=" * 70 + "\n")


def print_backtest_plan(val_end: str, test_start: str, test_end: str, forecast_horizon: int) -> None:
    """Print the backtest plan showing all origins."""
    from utils.backtesting import generate_backtest_origins

    origins = generate_backtest_origins(val_end, test_start, test_end, forecast_horizon)

    print("\n" + "=" * 70)
    print("BACKTEST PLAN")
    print("=" * 70)
    print(f"  Total Origins: {len(origins)}")
    print(f"  Forecast Horizon: {forecast_horizon}")
    print()
    print(f"  {'Origin':<8} {'Train Cutoff':<14} {'Forecast Window':<30}")
    print(f"  {'-'*8} {'-'*14} {'-'*30}")

    for origin in origins:
        origin_idx = origin['origin_idx']
        train_cutoff = origin['train_cutoff']
        forecast_start = origin['forecast_start']
        forecast_end = origin['forecast_end']
        print(f"  {origin_idx:<8} {train_cutoff:<14} {forecast_start} to {forecast_end}")

    print("=" * 70 + "\n")


def print_result_summary(result) -> None:
    """Print result summary."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    if result.success:
        print(f"  Status:           ✅ SUCCESS")
        print(f"  Total Origins:    {result.total_origins}")
        print(f"  Total Forecasts:  {result.total_forecasts}")
        print(f"  Forecast Horizon: {result.forecast_horizon}")
        print(f"  Elapsed Time:     {result.elapsed_seconds:.1f}s ({result.elapsed_seconds/60:.1f} min)")

        # Print per-origin summary
        if result.origin_details:
            print(f"\n  Per-Origin Summary:")
            print(f"  {'Origin':<8} {'Forecasts':<12} {'Time (s)':<10} {'Status':<10}")
            print(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*10}")
            for origin_idx, details in sorted(result.origin_details.items(), key=lambda x: int(x[0])):
                status = "✓" if details['success'] else "✗"
                print(f"  {origin_idx:<8} {details['num_forecasts']:<12} {details['elapsed_seconds']:<10.1f} {status:<10}")

        print(f"\n  Output Files:")
        print(f"    Forecasts: {result.forecasts_path}")
        print(f"    Metrics:   {result.metrics_path}")
        print(f"    Summary:   {result.summary_path}")
    else:
        print(f"  Status:           ❌ FAILED")
        print(f"  Error:            {result.error_message}")

    print("=" * 70 + "\n")


def run_backtesting_only(config_path: str, enable_trace: bool = True, output_dir: str = None, verbose: bool = False) -> None:
    """
    Run backtesting pipeline programmatically (for use by databricks_runner.py and other scripts).

    Args:
        config_path: Path to config YAML file
        enable_trace: Whether to enable trace logging (default True)
        output_dir: Override output directory (optional)
        verbose: Enable verbose logging

    Raises:
        SystemExit: If pipeline fails
    """
    setup_logging(verbose=verbose)
    logger = logging.getLogger(__name__)

    print_banner()

    # Validate config path
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Load config
    logger.info(f"Loading config from: {config_path}")

    try:
        from config.schema import load_config_from_yaml
        config = load_config_from_yaml(config_path)
        # Bootstrap: auto-detect splits + normalise periods
        from utils.period_utils import bootstrap_config as _bootstrap
        config, _src_df = _bootstrap(config_path if "config_path" in dir() else args.config)
        del _src_df  # Free memory; crews load data themselves

    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    print_config_summary(config)

    # Print backtest plan
    print_backtest_plan(
        val_end=config.val_end,
        test_start=config.test_start,
        test_end=config.test_end,
        forecast_horizon=config.forecast_horizon,
    )

    # Validate required artifacts exist
    artifact_base = config.artifact_base_path
    required_dirs = ['seg_output', 'feature_output', 'model_artifacts']
    required_files = [
        ('seg_output', 'cluster_model.joblib'),
        ('seg_output', 'scaler.joblib'),
        ('seg_output', 'clustering_metrics.json'),
        ('feature_output', 'training_manifest.csv'),
        ('feature_output', 'train_features.csv'),
        ('feature_output', 'val_features.csv'),
        ('feature_output', 'test_features.csv'),
        ('model_artifacts', 'final_model_specs.json'),
    ]

    missing = []
    for subdir in required_dirs:
        path = os.path.join(artifact_base, subdir)
        if not os.path.exists(path):
            missing.append(path)

    # Format-agnostic check for train/val/test_features.csv: parquet OR
    # csv at the same base name counts as "present".
    from utils.feature_io import features_intermediate_exists
    _features_bases = {'train_features.csv', 'val_features.csv', 'test_features.csv'}
    for subdir, filename in required_files:
        if filename in _features_bases:
            base = filename[:-4]
            if not features_intermediate_exists(os.path.join(artifact_base, subdir), base):
                missing.append(os.path.join(artifact_base, subdir, f"{base}.[parquet|csv]"))
            continue
        path = os.path.join(artifact_base, subdir, filename)
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        logger.error("Missing required artifacts:")
        for m in missing:
            logger.error(f"  - {m}")
        logger.error("\nPlease run the full training pipeline first.")
        sys.exit(1)

    # Run backtesting pipeline
    logger.info("Starting rolling-origin backtesting pipeline...")
    start_time = datetime.now()

    try:
        from utils.backtesting import run_rolling_origin_backtest

        result = run_rolling_origin_backtest(
            config=config,
            output_dir=output_dir,
            verbose=verbose,
        )

    except Exception as e:
        import traceback
        logger.error(f"Pipeline failed with exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Print result summary
    print_result_summary(result)

    # Calculate elapsed time
    elapsed = datetime.now() - start_time
    logger.info(f"Total elapsed time: {elapsed}")

    # Exit with appropriate code
    if result.success:
        logger.info("Backtesting pipeline completed successfully!")
    else:
        logger.error("Backtesting pipeline failed!")
        sys.exit(1)


def main():
    """Main entry point for backtesting pipeline."""
    parser = argparse.ArgumentParser(
        description='Run rolling-origin backtesting with walk-forward validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtesting.py --config config/config_us_pc.yaml
  python run_backtesting.py --config config/config_us_pc.yaml --verbose
  python run_backtesting.py --config config/config_us_pc.yaml --output-dir ./backtest
  python run_backtesting.py --config config/config_us_pc.yaml --plan-only
        """
    )

    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to config YAML file'
    )

    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Override output directory (default: {artifact_base}/backtest_output)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (DEBUG level) logging'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress banner and summary output'
    )

    parser.add_argument(
        '--plan-only',
        action='store_true',
        help='Only show the backtest plan without running'
    )

    parser.add_argument(
        '--log-file',
        default=None,
        help='Path to log file (optional)'
    )

    parser.add_argument(
        '--email',
        action='store_true',
        help='Enable email notifications after each origin (uses Mac Outlook Desktop)'
    )

    parser.add_argument(
        '--to',
        type=str,
        default='debonil.chowdhury@aria-is.com',
        help='Email recipient address (default: debonil.chowdhury@aria-is.com)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose, log_file=args.log_file)
    logger = logging.getLogger(__name__)

    # Print banner
    if not args.quiet:
        print_banner()

    # Validate config path
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Load config
    logger.info(f"Loading config from: {args.config}")

    try:
        from config.schema import load_config_from_yaml
        config = load_config_from_yaml(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Print config summary
    if not args.quiet:
        print_config_summary(config)

    # Print backtest plan
    if not args.quiet:
        print_backtest_plan(
            val_end=config.val_end,
            test_start=config.test_start,
            test_end=config.test_end,
            forecast_horizon=config.forecast_horizon,
        )

    # If plan-only mode, exit here
    if args.plan_only:
        logger.info("Plan-only mode: exiting without running backtest")
        sys.exit(0)

    # Validate required artifacts exist
    artifact_base = config.artifact_base_path
    required_dirs = ['seg_output', 'feature_output', 'model_artifacts']
    required_files = [
        ('seg_output', 'cluster_model.joblib'),
        ('seg_output', 'scaler.joblib'),
        ('seg_output', 'clustering_metrics.json'),
        ('feature_output', 'training_manifest.csv'),
        ('feature_output', 'train_features.csv'),
        ('feature_output', 'val_features.csv'),
        ('feature_output', 'test_features.csv'),
        ('model_artifacts', 'final_model_specs.json'),
    ]

    missing = []
    for subdir in required_dirs:
        path = os.path.join(artifact_base, subdir)
        if not os.path.exists(path):
            missing.append(path)

    # Format-agnostic check for train/val/test_features.csv: parquet OR
    # csv at the same base name counts as "present".
    from utils.feature_io import features_intermediate_exists
    _features_bases = {'train_features.csv', 'val_features.csv', 'test_features.csv'}
    for subdir, filename in required_files:
        if filename in _features_bases:
            base = filename[:-4]
            if not features_intermediate_exists(os.path.join(artifact_base, subdir), base):
                missing.append(os.path.join(artifact_base, subdir, f"{base}.[parquet|csv]"))
            continue
        path = os.path.join(artifact_base, subdir, filename)
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        logger.error("Missing required artifacts:")
        for m in missing:
            logger.error(f"  - {m}")
        logger.error("\nPlease run the full training pipeline first.")
        sys.exit(1)

    # Setup email notifications if enabled
    to_email = args.to if args.email else None
    origin_callback = None

    if to_email:
        logger.info(f"Email notifications enabled - sending to: {to_email}")
        # Generate origins to get total count for callback
        from utils.backtesting import generate_backtest_origins
        origins = generate_backtest_origins(
            val_end=config.val_end,
            test_start=config.test_start,
            test_end=config.test_end,
            forecast_horizon=config.forecast_horizon,
        )
        total_origins = len(origins)
        origin_callback = create_origin_callback(to_email, total_origins)

        # Send start email
        send_backtest_start_email(to_email, total_origins, config)

    # Run backtesting pipeline
    logger.info("Starting rolling-origin backtesting pipeline...")
    start_time = datetime.now()

    try:
        from utils.backtesting import run_rolling_origin_backtest

        result = run_rolling_origin_backtest(
            config=config,
            output_dir=args.output_dir,
            verbose=args.verbose,
            origin_callback=origin_callback,
        )

    except Exception as e:
        import traceback
        logger.error(f"Pipeline failed with exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Print result summary
    if not args.quiet:
        print_result_summary(result)

    # Send final completion email
    if to_email and result.success:
        # Read overall metrics from the metrics file
        import pandas as pd
        try:
            metrics_df = pd.read_csv(result.metrics_path)
            overall_row = metrics_df[metrics_df['group_type'] == 'overall'].iloc[0]
            overall_metrics = {
                'wape': overall_row.get('wape', 0),
                'bias_pct': overall_row.get('bias_pct', 0),
                'mae': overall_row.get('mae', 0),
                'n_forecasts': overall_row.get('n_forecasts', 0),
            }
        except Exception:
            overall_metrics = {'wape': 0, 'bias_pct': 0, 'mae': 0, 'n_forecasts': 0}

        send_backtest_completion_email(to_email, result, overall_metrics)

    # Calculate elapsed time
    elapsed = datetime.now() - start_time
    logger.info(f"Total elapsed time: {elapsed}")

    # Exit with appropriate code
    if result.success:
        logger.info("Backtesting pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("Backtesting pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
