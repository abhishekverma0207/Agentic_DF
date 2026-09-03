#!/usr/bin/env python3
"""
FEU-Agentic-Forecasting Demand Forecasting - Unit Test Runner
===============================================

Runs the complete agentic forecasting pipeline on a SAMPLED subset of prediction keys
for fast unit testing and validation.

Usage:
    # Run with 5% sample (default):
    python unit_test_runner.py --config config/config.yaml

    # Run with custom sample percentage:
    python unit_test_runner.py --config config/config.yaml --sample-pct 10

    # With email notifications (Mac Outlook Desktop):
    python unit_test_runner.py --config config/config.yaml --email

    # Specify recipient email (optional):
    python unit_test_runner.py --config config/config.yaml --email --to your.email@aria-is.com

    # Set minimum number of keys (useful when sample % would give too few):
    python unit_test_runner.py --config config/config.yaml --sample-pct 5 --min-keys 10

The pipeline executes in sequence:
    1. EDA Crew
    2. Segmentation Crew
    3. Feature Engineering Crew
    4. Model Training Crew
    5. Diagnostic Crew

This runner:
    - Samples prediction keys randomly at the specified percentage
    - Creates a temporary sampled dataset
    - Creates a temporary config pointing to the sampled data
    - Runs the full pipeline on the sample
    - Cleans up temporary files (optional)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

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

    Parameters
    ----------
    to_email : str
        Recipient email address
    subject : str
        Email subject
    html_body : str
        Email body (HTML format)

    Returns
    -------
    bool
        True if email sent successfully, False otherwise
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


def send_crew_start_notification(
    to_email: Optional[str],
    crew_name: str,
    crew_number: int,
    total_crews: int,
) -> None:
    """Send notification when a crew starts."""
    if not to_email:
        return

    subject = f"[FEU-Agentic-Forecasting UNIT TEST] {crew_name} Started ({crew_number}/{total_crews})"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; color: #333; }}
            .header {{ background-color: #ff9800; color: white; padding: 20px; border-radius: 5px; }}
            .content {{ padding: 20px; background-color: #f5f5f5; border-radius: 5px; margin-top: 10px; }}
            .info-row {{ padding: 8px 0; border-bottom: 1px solid #ddd; }}
            .label {{ font-weight: bold; color: #555; }}
            .value {{ color: #ff9800; }}
            .footer {{ margin-top: 20px; padding: 15px; background-color: #fff3e0; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🧪 [UNIT TEST] {crew_name} Started</h2>
        </div>
        <div class="content">
            <div class="info-row">
                <span class="label">Progress:</span>
                <span class="value">{crew_number} of {total_crews} crews</span>
            </div>
            <div class="info-row">
                <span class="label">Start Time:</span>
                <span class="value">{now}</span>
            </div>
        </div>
        <div class="footer">
            <p>⚠️ This is a UNIT TEST run on sampled data. The crew is now processing.</p>
        </div>
    </body>
    </html>
    """

    send_outlook_desktop_email(to_email, subject, html_body)


def send_crew_completion_notification(
    to_email: Optional[str],
    crew_name: str,
    crew_number: int,
    total_crews: int,
    duration_seconds: float,
    summary: str,
) -> None:
    """Send notification when a crew completes successfully."""
    if not to_email:
        return

    subject = f"[FEU-Agentic-Forecasting UNIT TEST] ✅ {crew_name} Completed ({crew_number}/{total_crews})"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    duration_min = duration_seconds / 60.0

    status_emoji = "🎉" if crew_number == total_crews else "⏭️"
    status_text = "Unit Test Complete!" if crew_number == total_crews else "Proceeding to next crew..."
    status_color = "#4caf50" if crew_number == total_crews else "#2196f3"

    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; color: #333; }}
            .header {{ background-color: #4caf50; color: white; padding: 20px; border-radius: 5px; }}
            .content {{ padding: 20px; background-color: #f5f5f5; border-radius: 5px; margin-top: 10px; }}
            .info-row {{ padding: 8px 0; border-bottom: 1px solid #ddd; }}
            .label {{ font-weight: bold; color: #555; }}
            .value {{ color: #4caf50; }}
            .summary {{ background-color: white; padding: 15px; border-left: 4px solid #4caf50; margin: 15px 0; }}
            .status {{ margin-top: 20px; padding: 15px; background-color: {status_color}; color: white; border-radius: 5px; text-align: center; }}
            .test-badge {{ background-color: #ff9800; color: white; padding: 5px 10px; border-radius: 3px; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>✅ <span class="test-badge">UNIT TEST</span> {crew_name} Completed</h2>
        </div>
        <div class="content">
            <div class="info-row">
                <span class="label">Progress:</span>
                <span class="value">{crew_number} of {total_crews} crews</span>
            </div>
            <div class="info-row">
                <span class="label">Duration:</span>
                <span class="value">{duration_min:.2f} minutes</span>
            </div>
            <div class="info-row">
                <span class="label">Completion Time:</span>
                <span class="value">{now}</span>
            </div>
            <div class="summary">
                <h3>Summary:</h3>
                <pre style="font-family: monospace; white-space: pre-wrap;">{summary}</pre>
            </div>
        </div>
        <div class="status">
            <h3>{status_emoji} {status_text}</h3>
        </div>
    </body>
    </html>
    """

    send_outlook_desktop_email(to_email, subject, html_body)


def send_crew_failure_notification(
    to_email: Optional[str],
    crew_name: str,
    crew_number: int,
    total_crews: int,
    error_message: str,
    traceback_str: str,
) -> None:
    """Send notification when a crew fails."""
    if not to_email:
        return

    subject = f"[FEU-Agentic-Forecasting UNIT TEST] ❌ {crew_name} FAILED ({crew_number}/{total_crews})"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; color: #333; }}
            .header {{ background-color: #f44336; color: white; padding: 20px; border-radius: 5px; }}
            .content {{ padding: 20px; background-color: #f5f5f5; border-radius: 5px; margin-top: 10px; }}
            .info-row {{ padding: 8px 0; border-bottom: 1px solid #ddd; }}
            .label {{ font-weight: bold; color: #555; }}
            .value {{ color: #f44336; }}
            .error-box {{ background-color: #ffebee; padding: 15px; border-left: 4px solid #f44336; margin: 15px 0; }}
            .traceback {{ background-color: #fafafa; padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin: 15px 0; overflow-x: auto; }}
            .warning {{ margin-top: 20px; padding: 15px; background-color: #ff9800; color: white; border-radius: 5px; }}
            .test-badge {{ background-color: #ff9800; color: white; padding: 5px 10px; border-radius: 3px; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>❌ <span class="test-badge">UNIT TEST</span> {crew_name} Failed</h2>
        </div>
        <div class="content">
            <div class="info-row">
                <span class="label">Crew:</span>
                <span class="value">{crew_name}</span>
            </div>
            <div class="info-row">
                <span class="label">Progress:</span>
                <span class="value">{crew_number} of {total_crews} crews</span>
            </div>
            <div class="info-row">
                <span class="label">Failure Time:</span>
                <span class="value">{now}</span>
            </div>
            <div class="error-box">
                <h3 style="margin-top: 0; color: #f44336;">Error Message:</h3>
                <p><strong>{error_message}</strong></p>
            </div>
            <div class="traceback">
                <h3>Traceback:</h3>
                <pre style="font-family: monospace; white-space: pre-wrap; font-size: 12px;">{traceback_str}</pre>
            </div>
        </div>
        <div class="warning">
            <h3 style="margin: 0;">⚠️ Unit Test Execution Halted</h3>
            <p style="margin: 10px 0 0 0;">Please investigate and resolve the issue above.</p>
        </div>
    </body>
    </html>
    """

    send_outlook_desktop_email(to_email, subject, html_body)


def send_pipeline_completion_notification(
    to_email: Optional[str],
    total_duration_seconds: float,
    crew_summaries: List[Dict[str, Any]],
    sample_info: Dict[str, Any],
) -> None:
    """Send final notification when entire pipeline completes."""
    if not to_email:
        return

    subject = "[FEU-Agentic-Forecasting UNIT TEST] 🎉 Complete Pipeline Finished Successfully"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total_duration_min = total_duration_seconds / 60.0

    crew_rows = "\n".join([
        f"""
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;">{s['name']}</td>
            <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">{s['duration']:.2f} min</td>
            <td style="padding: 10px; border: 1px solid #ddd; text-align: center; color: #4caf50;">✅ Success</td>
        </tr>
        """
        for s in crew_summaries
    ])

    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; color: #333; }}
            .header {{ background-color: #4caf50; color: white; padding: 25px; border-radius: 5px; text-align: center; }}
            .content {{ padding: 20px; background-color: #f5f5f5; border-radius: 5px; margin-top: 10px; }}
            .sample-info {{ background-color: #fff3e0; padding: 15px; border-left: 4px solid #ff9800; margin: 15px 0; border-radius: 5px; }}
            .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .stat-box {{ background-color: white; padding: 20px; border-radius: 5px; text-align: center; flex: 1; margin: 0 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .stat-value {{ font-size: 32px; font-weight: bold; color: #4caf50; }}
            .stat-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
            table {{ width: 100%; border-collapse: collapse; background-color: white; margin: 20px 0; }}
            th {{ background-color: #4caf50; color: white; padding: 12px; text-align: left; }}
            .footer {{ margin-top: 20px; padding: 20px; background-color: #e8f5e9; border-radius: 5px; border-left: 4px solid #4caf50; }}
            .test-badge {{ background-color: #ff9800; color: white; padding: 5px 10px; border-radius: 3px; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1 style="margin: 0;">🎉 <span class="test-badge">UNIT TEST</span> Pipeline Complete!</h1>
            <p style="margin: 10px 0 0 0; font-size: 18px;">All crews executed successfully on sampled data</p>
        </div>
        <div class="content">
            <div class="sample-info">
                <h3 style="margin-top: 0; color: #ff9800;">🧪 Sample Information</h3>
                <p><strong>Total prediction keys:</strong> {sample_info['total_keys']}</p>
                <p><strong>Sampled keys:</strong> {sample_info['sampled_keys']} ({sample_info['sample_pct']:.1f}%)</p>
                <p><strong>Total rows (original):</strong> {sample_info['total_rows']}</p>
                <p><strong>Sampled rows:</strong> {sample_info['sampled_rows']}</p>
            </div>

            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">{total_duration_min:.1f}</div>
                    <div class="stat-label">Total Minutes</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len(crew_summaries)}</div>
                    <div class="stat-label">Crews Completed</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">100%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
            </div>

            <h3>Crew Execution Summary</h3>
            <table>
                <thead>
                    <tr>
                        <th>Crew Name</th>
                        <th style="text-align: center;">Duration</th>
                        <th style="text-align: center;">Status</th>
                    </tr>
                </thead>
                <tbody>
                    {crew_rows}
                </tbody>
            </table>

            <p style="margin-top: 15px;"><strong>Completion Time:</strong> {now}</p>
        </div>
        <div class="footer">
            <h3 style="margin-top: 0; color: #4caf50;">✅ Unit Test Passed</h3>
            <p style="margin: 5px 0;">
                • The pipeline executed successfully on the sampled data subset<br>
                • Review artifacts in the test output directory<br>
                • Run the full pipeline with <code>runner.py</code> for production
            </p>
        </div>
    </body>
    </html>
    """

    send_outlook_desktop_email(to_email, subject, html_body)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: str = "unit_test_runner.log"):
    """Configure logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


# =============================================================================
# Data Sampling
# =============================================================================

def sample_prediction_keys(
    config_yaml_path: str,
    sample_pct: float = 5.0,
    min_keys: int = 3,
    random_seed: int = 42,
    output_dir: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Sample prediction keys from the dataset and create config/data files for unit testing.

    Parameters
    ----------
    config_yaml_path : str
        Path to the original config YAML file
    sample_pct : float
        Percentage of prediction keys to sample (default: 5%)
    min_keys : int
        Minimum number of keys to sample (default: 3)
    random_seed : int
        Random seed for reproducibility (default: 42)
    output_dir : Optional[str]
        Directory to save test artifacts. If None, creates 'test_artifacts/' in project root.

    Returns
    -------
    Tuple[str, str, Dict[str, Any]]
        - Path to sampled config YAML
        - Path to test output directory
        - Sample info dictionary with statistics
    """
    logging.info(f"Loading original config from: {config_yaml_path}")

    # Load original config
    yaml_path = Path(config_yaml_path).expanduser().resolve()
    with yaml_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # Resolve paths relative to config directory
    config_dir = yaml_path.parent
    if config_dir.name.lower() in {"config", "configs"}:
        project_root = config_dir.parent
    else:
        project_root = config_dir

    def resolve_path(p: str) -> str:
        pp = Path(p).expanduser()
        if pp.is_absolute():
            return str(pp.resolve())
        return str((project_root / pp).resolve())

    original_data_path = resolve_path(raw_config["input_data_path"])
    original_artifact_path = resolve_path(raw_config["artifact_base_path"])
    prediction_key_cols = raw_config["prediction_key_cols"]

    logging.info(f"Loading data from: {original_data_path}")
    logging.info(f"Prediction key columns: {prediction_key_cols}")

    # Load the data (supports csv, parquet, delta)
    if original_data_path.endswith(".parquet") or original_data_path.endswith(".pq"):
        df = pd.read_parquet(original_data_path)
    elif original_data_path.endswith(".csv"):
        df = pd.read_csv(original_data_path)
    elif os.path.isdir(original_data_path) and os.path.isdir(os.path.join(original_data_path, '_delta_log')):
        from deltalake import DeltaTable
        df = DeltaTable(original_data_path).to_pandas()
    elif os.path.isdir(original_data_path):
        df = pd.read_parquet(original_data_path)
    else:
        raise ValueError(f"Unsupported data format: {original_data_path}. Use .csv, .parquet, or delta directory.")

    total_rows = len(df)
    logging.info(f"Loaded {total_rows:,} rows")

    # Get unique prediction keys
    if len(prediction_key_cols) == 1:
        unique_keys = df[prediction_key_cols[0]].unique()
        key_df = pd.DataFrame({prediction_key_cols[0]: unique_keys})
    else:
        key_df = df[prediction_key_cols].drop_duplicates()
        unique_keys = key_df.values

    total_keys = len(key_df)
    logging.info(f"Found {total_keys:,} unique prediction keys")

    # Calculate sample size
    sample_size = max(min_keys, int(np.ceil(total_keys * sample_pct / 100.0)))
    sample_size = min(sample_size, total_keys)  # Don't exceed total

    logging.info(f"Sampling {sample_size} keys ({sample_pct}% of {total_keys}, min={min_keys})")

    # Random sample
    np.random.seed(random_seed)
    sample_indices = np.random.choice(len(key_df), size=sample_size, replace=False)
    sampled_key_df = key_df.iloc[sample_indices]

    # Filter data to sampled keys
    if len(prediction_key_cols) == 1:
        mask = df[prediction_key_cols[0]].isin(sampled_key_df[prediction_key_cols[0]])
    else:
        # Multi-column key: merge to filter
        df_with_marker = df.merge(
            sampled_key_df.assign(_sampled=True),
            on=prediction_key_cols,
            how="left"
        )
        mask = df_with_marker["_sampled"].fillna(False)

    sampled_df = df[mask].copy()
    sampled_rows = len(sampled_df)

    logging.info(f"Sampled data: {sampled_rows:,} rows ({sampled_rows / total_rows * 100:.1f}% of original)")

    # Create test output directory in project root (NOT temp directory)
    if output_dir:
        test_dir = Path(output_dir).expanduser().resolve()
    else:
        # Default: create test_artifacts/ in project root with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir = project_root / f"test_artifacts_{timestamp}"

    test_dir = Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    test_dir_str = str(test_dir)

    logging.info(f"Created test output directory: {test_dir_str}")

    # Save sampled data
    sampled_data_dir = test_dir / "sampled_data"
    sampled_data_dir.mkdir(exist_ok=True)

    sampled_data_filename = Path(original_data_path).name
    if sampled_data_filename.endswith(".parquet"):
        sampled_data_path = str(sampled_data_dir / sampled_data_filename)
        sampled_df.to_parquet(sampled_data_path, index=False)
    else:
        sampled_data_path = str(sampled_data_dir / sampled_data_filename.replace(".csv", "_sampled.csv"))
        sampled_df.to_csv(sampled_data_path, index=False)

    logging.info(f"Saved sampled data to: {sampled_data_path}")

    # Create artifact directory inside test directory
    sampled_artifact_path = str(test_dir / "artifacts")
    os.makedirs(sampled_artifact_path, exist_ok=True)

    # Create new config with updated paths
    sampled_config = raw_config.copy()
    sampled_config["input_data_path"] = sampled_data_path
    sampled_config["artifact_base_path"] = sampled_artifact_path

    # Save sampled config
    sampled_config_path = os.path.join(test_dir_str, "config_unit_test.yaml")
    with open(sampled_config_path, "w", encoding="utf-8") as f:
        yaml.dump(sampled_config, f, default_flow_style=False)

    logging.info(f"Saved sampled config to: {sampled_config_path}")

    # Save sampling info
    sample_info = {
        "total_keys": total_keys,
        "sampled_keys": sample_size,
        "sample_pct": sample_pct,
        "total_rows": total_rows,
        "sampled_rows": sampled_rows,
        "random_seed": random_seed,
        "test_dir": test_dir_str,
        "sampled_data_path": sampled_data_path,
        "sampled_config_path": sampled_config_path,
        "sampled_artifact_path": sampled_artifact_path,
    }

    # Save sample info to test dir
    with open(os.path.join(test_dir_str, "sample_info.json"), "w") as f:
        json.dump(sample_info, f, indent=2)

    return sampled_config_path, test_dir_str, sample_info


# =============================================================================
# Crew Runners
# =============================================================================

def run_eda_crew_with_notifications(
    llm,
    config,
    config_yaml_path: str,
    to_email: Optional[str],
) -> Dict[str, Any]:
    """Run EDA crew with email notifications.

    Uses DETERMINISTIC EDA for core analysis (no LLM for per_key_metrics.csv etc.)
    LLM is only used for creating context files with rationale.
    """
    from crews.eda_crew import run_eda_deterministic

    crew_name = "EDA Crew"
    crew_number = 1
    total_crews = 5

    logging.info(f"\n{'='*70}")
    logging.info(f"[{crew_number}/{total_crews}] Starting {crew_name}")
    logging.info(f"{'='*70}")

    send_crew_start_notification(to_email, crew_name, crew_number, total_crews)

    start_time = datetime.now()

    try:
        # Use DETERMINISTIC EDA - core analysis without LLM, only context files use LLM
        result = run_eda_deterministic(config=config, llm=llm)

        duration = (datetime.now() - start_time).total_seconds()

        summary = f"""EDA analysis complete.
  - Global summary: {result.global_eda_summary_path}
  - Per-key metrics: {result.per_key_metrics_path}
  - Report: {result.eda_report_markdown_path}
"""

        send_crew_completion_notification(
            to_email, crew_name, crew_number, total_crews, duration, summary
        )

        logging.info(f"✅ {crew_name} completed in {duration:.2f} seconds")
        return {"name": crew_name, "duration": duration / 60.0, "result": result}

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        tb = traceback.format_exc()

        send_crew_failure_notification(
            to_email, crew_name, crew_number, total_crews, error_msg, tb
        )

        logging.error(f"❌ {crew_name} failed after {duration:.2f} seconds")
        raise


def run_segmentation_crew_with_notifications(
    llm,
    config,
    config_yaml_path: str,
    to_email: Optional[str],
) -> Dict[str, Any]:
    """Run Segmentation crew with email notifications."""
    from crews.segmentation_crew import run_segmentation_crew

    crew_name = "Segmentation Crew"
    crew_number = 2
    total_crews = 5

    logging.info(f"\n{'='*70}")
    logging.info(f"[{crew_number}/{total_crews}] Starting {crew_name}")
    logging.info(f"{'='*70}")

    send_crew_start_notification(to_email, crew_name, crew_number, total_crews)

    start_time = datetime.now()

    try:
        result = run_segmentation_crew(llm=llm, config=config, config_yaml_path=config_yaml_path)

        duration = (datetime.now() - start_time).total_seconds()

        summary = f"""Segmentation complete.
  - Segments: {result.per_key_with_segments_path}
  - Modeling strategy: {result.modeling_strategy_path}
  - Feature recommendations: {result.feature_recommendations_path}
  - Report: {result.segmentation_report_markdown_path}
"""

        send_crew_completion_notification(
            to_email, crew_name, crew_number, total_crews, duration, summary
        )

        logging.info(f"✅ {crew_name} completed in {duration:.2f} seconds")
        return {"name": crew_name, "duration": duration / 60.0, "result": result}

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        tb = traceback.format_exc()

        send_crew_failure_notification(
            to_email, crew_name, crew_number, total_crews, error_msg, tb
        )

        logging.error(f"❌ {crew_name} failed after {duration:.2f} seconds")
        raise


def run_feature_crew_with_notifications(
    llm,
    config,
    config_yaml_path: str,
    to_email: Optional[str],
) -> Dict[str, Any]:
    """Run Feature Engineering crew with email notifications."""
    from crews.feature_crew import run_feature_crew

    crew_name = "Feature Engineering Crew"
    crew_number = 3
    total_crews = 5

    logging.info(f"\n{'='*70}")
    logging.info(f"[{crew_number}/{total_crews}] Starting {crew_name}")
    logging.info(f"{'='*70}")

    send_crew_start_notification(to_email, crew_name, crew_number, total_crews)

    start_time = datetime.now()

    try:
        result = run_feature_crew(llm=llm, config=config, config_yaml_path=config_yaml_path)

        duration = (datetime.now() - start_time).total_seconds()

        summary = f"""Feature engineering complete.
  - Metadata: {result.feature_metadata_path}
  - Quality summary: {result.feature_quality_summary_path}
  - Report: {result.feature_report_markdown_path}
"""

        send_crew_completion_notification(
            to_email, crew_name, crew_number, total_crews, duration, summary
        )

        logging.info(f"✅ {crew_name} completed in {duration:.2f} seconds")
        return {"name": crew_name, "duration": duration / 60.0, "result": result}

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        tb = traceback.format_exc()

        send_crew_failure_notification(
            to_email, crew_name, crew_number, total_crews, error_msg, tb
        )

        logging.error(f"❌ {crew_name} failed after {duration:.2f} seconds")
        raise


def run_training_crew_with_notifications(
    llm,
    config,
    config_yaml_path: str,
    to_email: Optional[str],
) -> Dict[str, Any]:
    """Run Model Training crew with email notifications."""
    from crews.training_crew import run_training_crew

    crew_name = "Model Training Crew"
    crew_number = 4
    total_crews = 5

    logging.info(f"\n{'='*70}")
    logging.info(f"[{crew_number}/{total_crews}] Starting {crew_name}")
    logging.info(f"{'='*70}")

    send_crew_start_notification(to_email, crew_name, crew_number, total_crews)

    start_time = datetime.now()

    try:
        result = run_training_crew(llm=llm, config=config, config_yaml_path=config_yaml_path)

        duration = (datetime.now() - start_time).total_seconds()

        summary = f"""Model training complete.
  - Model specs: {result.final_model_specs_path}
  - Val forecasts directory: {result.model_dir}
  - Test forecasts directory: {result.model_dir}
"""

        send_crew_completion_notification(
            to_email, crew_name, crew_number, total_crews, duration, summary
        )

        logging.info(f"✅ {crew_name} completed in {duration:.2f} seconds")
        return {"name": crew_name, "duration": duration / 60.0, "result": result}

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        tb = traceback.format_exc()

        send_crew_failure_notification(
            to_email, crew_name, crew_number, total_crews, error_msg, tb
        )

        logging.error(f"❌ {crew_name} failed after {duration:.2f} seconds")
        raise


def run_diagnostic_crew_with_notifications(
    llm,
    config,
    config_yaml_path: str,
    to_email: Optional[str],
) -> Dict[str, Any]:
    """Run Diagnostic crew with email notifications."""
    from crews.diagnostic_crew import run_diagnostic_crew

    crew_name = "Diagnostic Crew"
    crew_number = 5
    total_crews = 5

    logging.info(f"\n{'='*70}")
    logging.info(f"[{crew_number}/{total_crews}] Starting {crew_name}")
    logging.info(f"{'='*70}")

    send_crew_start_notification(to_email, crew_name, crew_number, total_crews)

    start_time = datetime.now()

    try:
        result = run_diagnostic_crew(llm=llm, config=config, config_yaml_path=config_yaml_path)

        duration = (datetime.now() - start_time).total_seconds()

        summary = f"""Diagnostic analysis complete.
  - Group diagnostics: {result.group_diagnostics_path}
  - Segment diagnostics: {result.segment_diagnostics_path}
  - Summary JSON: {result.diagnostics_summary_path}
  - Report: {result.diagnostics_report_markdown_path}
"""

        send_crew_completion_notification(
            to_email, crew_name, crew_number, total_crews, duration, summary
        )

        logging.info(f"✅ {crew_name} completed in {duration:.2f} seconds")
        return {"name": crew_name, "duration": duration / 60.0, "result": result}

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        tb = traceback.format_exc()

        send_crew_failure_notification(
            to_email, crew_name, crew_number, total_crews, error_msg, tb
        )

        logging.error(f"❌ {crew_name} failed after {duration:.2f} seconds")
        raise


# =============================================================================
# Main Pipeline
# =============================================================================

def run_complete_pipeline(
    config_yaml_path: str,
    sample_pct: float = 5.0,
    min_keys: int = 3,
    random_seed: int = 42,
    to_email: Optional[str] = None,
    output_dir: Optional[str] = None,
    cleanup: bool = False,
    enable_trace: bool = True,
) -> bool:
    """
    Run the complete forecasting pipeline on sampled data.

    Parameters
    ----------
    config_yaml_path : str
        Path to config YAML file
    sample_pct : float
        Percentage of prediction keys to sample (default: 5%)
    min_keys : int
        Minimum number of keys to sample (default: 3)
    random_seed : int
        Random seed for reproducibility (default: 42)
    to_email : Optional[str]
        Email address for notifications (if None, no emails sent)
    output_dir : Optional[str]
        Directory to save test artifacts (default: test_artifacts_<timestamp>/ in project root)
    cleanup : bool
        Whether to delete test artifacts after successful completion (default: False)
    enable_trace : bool
        Whether to enable detailed CrewAI trace logging (default: True)

    Returns
    -------
    bool
        True if pipeline completed successfully, False otherwise
    """
    from config.schema import load_config_from_yaml
    from config.llm_config import create_bedrock_claude_llm

    pipeline_start = datetime.now()
    test_dir = None

    logging.info("\n" + "="*70)
    logging.info("FEU-AGENTIC-FORECASTING DEMAND FORECASTING - UNIT TEST PIPELINE")
    logging.info("="*70)
    logging.info(f"Original config: {config_yaml_path}")
    logging.info(f"Sample percentage: {sample_pct}%")
    logging.info(f"Minimum keys: {min_keys}")
    logging.info(f"Random seed: {random_seed}")
    logging.info(f"Output directory: {output_dir or 'test_artifacts_<timestamp>/'}")
    logging.info(f"Email notifications: {'ENABLED' if to_email else 'DISABLED'}")
    if to_email:
        logging.info(f"Recipient: {to_email}")
    logging.info(f"Trace logging: {'ENABLED' if enable_trace else 'DISABLED'}")
    logging.info("="*70 + "\n")

    try:
        # Sample data and create test config
        logging.info("="*70)
        logging.info("STEP 0: Sampling prediction keys")
        logging.info("="*70)

        sampled_config_path, test_dir, sample_info = sample_prediction_keys(
            config_yaml_path=config_yaml_path,
            sample_pct=sample_pct,
            min_keys=min_keys,
            random_seed=random_seed,
            output_dir=output_dir,
        )

        logging.info("\n📊 Sample Summary:")
        logging.info(f"   Total prediction keys: {sample_info['total_keys']:,}")
        logging.info(f"   Sampled keys: {sample_info['sampled_keys']:,} ({sample_info['sample_pct']:.1f}%)")
        logging.info(f"   Total rows: {sample_info['total_rows']:,}")
        logging.info(f"   Sampled rows: {sample_info['sampled_rows']:,}")
        logging.info(f"   Test directory: {test_dir}")
        logging.info("")

        # Load sampled configuration
        logging.info("Loading sampled configuration...")
        cfg = load_config_from_yaml(sampled_config_path)
        llm = create_bedrock_claude_llm()
        logging.info("✓ Configuration loaded\n")

        # Set up trace logging after we have the artifact_base_path
        if enable_trace:
            try:
                from utils.trace_logging import (
                    setup_crewai_trace_logging,
                    set_crewai_logging_env_vars,
                    get_trace_logger,
                )
                set_crewai_logging_env_vars()
                trace_dir = setup_crewai_trace_logging(
                    artifact_base_path=cfg.artifact_base_path,
                    log_level="DEBUG",
                    also_log_to_console=False,
                )
                logging.info(f"✓ CrewAI trace logging enabled")
                logging.info(f"  Trace directory: {trace_dir}")
                logging.info(f"  Text log: {trace_dir}/crewai_trace.log")
                logging.info(f"  JSON trace: {trace_dir}/crewai_trace.jsonl\n")
            except ImportError as e:
                logging.warning(f"Could not enable trace logging: {e}")

        # Initialize cost tracking for the pipeline
        try:
            from utils.cost_tracking import get_cost_tracker
            cost_tracker = get_cost_tracker()
            cost_tracker.start_pipeline()
            logging.info("✓ Cost tracking enabled\n")
        except ImportError as e:
            cost_tracker = None
            logging.warning(f"Could not enable cost tracking: {e}")

        crew_summaries = []

        # 1. EDA Crew
        eda_summary = run_eda_crew_with_notifications(llm, cfg, sampled_config_path, to_email)
        crew_summaries.append(eda_summary)

        # 2. Segmentation Crew
        seg_summary = run_segmentation_crew_with_notifications(llm, cfg, sampled_config_path, to_email)
        crew_summaries.append(seg_summary)

        # 3. Feature Engineering Crew
        feat_summary = run_feature_crew_with_notifications(llm, cfg, sampled_config_path, to_email)
        crew_summaries.append(feat_summary)

        # 4. Model Training Crew
        train_summary = run_training_crew_with_notifications(llm, cfg, sampled_config_path, to_email)
        crew_summaries.append(train_summary)

        # 5. Diagnostic Crew
        diag_summary = run_diagnostic_crew_with_notifications(llm, cfg, sampled_config_path, to_email)
        crew_summaries.append(diag_summary)

        # Pipeline completed
        total_duration = (datetime.now() - pipeline_start).total_seconds()

        logging.info("\n" + "="*70)
        logging.info("🎉 UNIT TEST PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("="*70)
        logging.info(f"Total duration: {total_duration / 60.0:.2f} minutes")
        logging.info(f"\n📊 Sample Summary:")
        logging.info(f"   Prediction keys tested: {sample_info['sampled_keys']:,} of {sample_info['total_keys']:,}")
        logging.info(f"   Rows processed: {sample_info['sampled_rows']:,} of {sample_info['total_rows']:,}")
        logging.info("\nCrew Summary:")
        for summary in crew_summaries:
            logging.info(f"  ✅ {summary['name']}: {summary['duration']:.2f} min")

        # Log context flow summary
        logging.info("\nContext Flow Summary:")
        logging.info("-" * 40)
        logging.info("  EDA → Segmentation: eda_to_segmentation_context.json")
        logging.info("  EDA → Feature:      eda_to_feature_context.json")
        logging.info("  Seg → Feature:      segmentation_to_feature_context.json")
        logging.info("  Seg → Training:     segmentation_to_training_context.json")
        logging.info("  Feature → Training: feature_to_training_context.json")
        logging.info("  Training → Diag:    training_to_diagnostic_context.json")
        logging.info("-" * 40)

        # Print context sizes if available
        try:
            from utils.context_manager import print_context_flow_summary
            print_context_flow_summary(cfg.artifact_base_path)
        except Exception:
            pass  # Context manager not available or no contexts found

        # Save trace summary if trace logging was enabled
        if enable_trace:
            try:
                from utils.trace_logging import get_trace_logger, print_trace_summary
                trace_logger = get_trace_logger()
                summary_path = trace_logger.save_summary()
                logging.info(f"\nTrace Summary saved to: {summary_path}")
                print_trace_summary(cfg.artifact_base_path)
            except Exception as e:
                logging.warning(f"Could not save trace summary: {e}")

        # Save and print cost summary
        if cost_tracker:
            try:
                pipeline_cost_report = cost_tracker.get_pipeline_report(output_dir=cfg.artifact_base_path)
                logging.info(f"\nPipeline Cost Summary saved to: {cfg.artifact_base_path}/pipeline_cost_summary.json")
                cost_tracker.print_summary()
            except Exception as e:
                logging.warning(f"Could not save cost summary: {e}")

        logging.info("="*70 + "\n")

        send_pipeline_completion_notification(to_email, total_duration, crew_summaries, sample_info)

        # Cleanup (only if explicitly requested)
        if cleanup and test_dir:
            logging.info(f"Cleaning up test directory: {test_dir}")
            shutil.rmtree(test_dir)
            logging.info("✓ Cleanup complete")
        else:
            logging.info(f"\n📁 Test artifacts saved to: {test_dir}")
            logging.info(f"   - Sampled data: {sample_info['sampled_data_path']}")
            logging.info(f"   - Test config: {sample_info['sampled_config_path']}")
            logging.info(f"   - Artifacts: {sample_info['sampled_artifact_path']}")

        return True

    except Exception as e:
        total_duration = (datetime.now() - pipeline_start).total_seconds()

        logging.error("\n" + "="*70)
        logging.error("❌ UNIT TEST PIPELINE FAILED")
        logging.error("="*70)
        logging.error(f"Duration before failure: {total_duration / 60.0:.2f} minutes")
        logging.error(f"Error: {e}")
        logging.error("="*70 + "\n")

        # Always keep test dir on failure for debugging
        if test_dir:
            logging.info(f"Test artifacts preserved for debugging: {test_dir}")

        # Failure notification already sent by individual crew runner
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run FEU-Agentic-Forecasting demand forecasting pipeline on sampled data for unit testing'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file (e.g., config/config.yaml)'
    )
    parser.add_argument(
        '--sample-pct',
        type=float,
        default=5.0,
        help='Percentage of prediction keys to sample (default: 5%%)'
    )
    parser.add_argument(
        '--min-keys',
        type=int,
        default=3,
        help='Minimum number of prediction keys to sample (default: 3)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save test artifacts (default: test_artifacts_<timestamp>/ in project root)'
    )
    parser.add_argument(
        '--email',
        action='store_true',
        help='Enable email notifications via Mac Outlook Desktop'
    )
    parser.add_argument(
        '--to',
        type=str,
        default='debonil.chowdhury@aria-is.com',
        help='Email address for notifications (default: debonil.chowdhury@aria-is.com)'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Delete test artifacts after successful completion (default: keep artifacts)'
    )
    parser.add_argument(
        '--trace',
        action='store_true',
        default=True,
        help='Enable detailed CrewAI trace logging (default: enabled)'
    )
    parser.add_argument(
        '--no-trace',
        action='store_true',
        help='Disable detailed CrewAI trace logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Determine email recipient
    to_email = args.to if args.email else None

    # Determine if trace logging should be enabled
    enable_trace = args.trace and not args.no_trace

    # Run pipeline
    success = run_complete_pipeline(
        config_yaml_path=args.config,
        sample_pct=args.sample_pct,
        min_keys=args.min_keys,
        random_seed=args.seed,
        to_email=to_email,
        output_dir=args.output_dir,
        cleanup=args.cleanup,
        enable_trace=enable_trace,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
