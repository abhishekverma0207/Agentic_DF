#!/usr/bin/env python3
"""
FEU-Agentic-Forecasting Demand Forecasting - Production Runner
================================================

Runs the complete agentic forecasting pipeline with optional email notifications.

Usage:
    # With email notifications (Mac Outlook Desktop):
    python runner.py --config config/config.yaml --email

    # Without email notifications:
    python runner.py --config config/config.yaml

    # Specify recipient email (optional, default: debonil.chowdhury@aria-is.com):
    python runner.py --config config/config.yaml --email --to your.email@aria-is.com

The pipeline executes in sequence:
    1. EDA Crew
    2. Segmentation Crew
    3. Feature Engineering Crew
    4. Model Training Crew
    5. Inference Pipeline (generates forward forecasts for true model performance)
    6. Diagnostic Crew (analyzes inference results)
    7. Rolling-Origin Backtesting (validates model across multiple forecast origins)

Email notifications (if enabled) are sent at each stage:
    - Crew start
    - Crew completion (with summary)
    - Crew failure (with error details)
    - Pipeline completion
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

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

    subject = f"[FEU-Agentic-Forecasting] {crew_name} Started ({crew_number}/{total_crews})"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

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
            <h2>🚀 {crew_name} Started</h2>
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
            <p>The crew is now processing data and will notify you upon completion.</p>
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

    subject = f"[FEU-Agentic-Forecasting] ✅ {crew_name} Completed ({crew_number}/{total_crews})"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    duration_min = duration_seconds / 60.0

    status_emoji = "🎉" if crew_number == total_crews else "⏭️"
    status_text = "Pipeline Complete!" if crew_number == total_crews else "Proceeding to next crew..."
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
        </style>
    </head>
    <body>
        <div class="header">
            <h2>✅ {crew_name} Completed Successfully</h2>
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

    subject = f"[FEU-Agentic-Forecasting] ❌ {crew_name} FAILED ({crew_number}/{total_crews})"
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
        </style>
    </head>
    <body>
        <div class="header">
            <h2>❌ {crew_name} Failed</h2>
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
            <h3 style="margin: 0;">⚠️ Pipeline Execution Halted</h3>
            <p style="margin: 10px 0 0 0;">Please investigate and resolve the issue above.</p>
        </div>
    </body>
    </html>
    """

    send_outlook_desktop_email(to_email, subject, html_body)




def _load_test_performance_metrics(artifact_base_path: str) -> Dict[str, Any]:
    """
    Load test performance metrics from diagnostics output.

    Returns a dict with overall_wape, accuracy, and other key metrics.
    """
    metrics = {
        'overall_wape': None,
        'overall_accuracy': None,
        'total_keys': None,
        'keys_below_threshold': None,
        'deployment_ready': None,
        'overall_assessment': None,
    }

    # Try to load diagnostics_summary.json
    diag_dir = os.path.join(artifact_base_path, "diagnostics_output")
    summary_path = os.path.join(diag_dir, "diagnostics_summary.json")

    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)

            metrics['overall_wape'] = summary.get('overall_wape')
            metrics['total_keys'] = summary.get('total_keys')
            metrics['keys_below_threshold'] = summary.get('keys_below_threshold')

            # Calculate accuracy from WAPE: Accuracy = 100 - WAPE
            if metrics['overall_wape'] is not None:
                wape_pct = metrics['overall_wape'] * 100 if metrics['overall_wape'] < 1 else metrics['overall_wape']
                metrics['overall_accuracy'] = max(0, 100 - wape_pct)
        except Exception as e:
            logging.warning(f"Could not parse diagnostics_summary.json: {e}")

    # Try to load model_verdict.json for additional info
    verdict_path = os.path.join(diag_dir, "model_verdict.json")
    if os.path.exists(verdict_path):
        try:
            with open(verdict_path, 'r') as f:
                verdict = json.load(f)

            if metrics['overall_wape'] is None:
                metrics['overall_wape'] = verdict.get('overall_wape')
            metrics['deployment_ready'] = verdict.get('deployment_ready')
            metrics['overall_assessment'] = verdict.get('overall_assessment')

            # Recalculate accuracy if we got WAPE from verdict
            if metrics['overall_accuracy'] is None and metrics['overall_wape'] is not None:
                wape_pct = metrics['overall_wape'] * 100 if metrics['overall_wape'] < 1 else metrics['overall_wape']
                metrics['overall_accuracy'] = max(0, 100 - wape_pct)
        except Exception as e:
            logging.warning(f"Could not parse model_verdict.json: {e}")

    return metrics


def send_pipeline_completion_notification(
    to_email: Optional[str],
    total_duration_seconds: float,
    crew_summaries: List[Dict[str, Any]],
    artifact_base_path: Optional[str] = None,
) -> None:
    """Send final notification when entire pipeline completes."""
    if not to_email:
        return

    subject = "[FEU-Agentic-Forecasting] 🎉 Complete Pipeline Finished Successfully"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total_duration_min = total_duration_seconds / 60.0

    # Load test performance metrics if artifact_base_path is provided
    metrics = {}

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

    # Build metrics section for the email
    metrics_html = ""
    if metrics.get('overall_wape') is not None or metrics.get('overall_accuracy') is not None:
        wape_display = f"{metrics['overall_wape']:.2%}" if metrics.get('overall_wape') is not None else "N/A"
        accuracy_display = f"{metrics['overall_accuracy']:.1f}%" if metrics.get('overall_accuracy') is not None else "N/A"
        assessment = metrics.get('overall_assessment', 'N/A')
        deployment_status = "✅ Yes" if metrics.get('deployment_ready') else "⚠️ Review Required" if metrics.get('deployment_ready') is False else "N/A"

        # Color coding for WAPE
        wape_color = "#4caf50"  # green
        if metrics.get('overall_wape') is not None:
            if metrics['overall_wape'] > 0.3:
                wape_color = "#f44336"  # red
            elif metrics['overall_wape'] > 0.2:
                wape_color = "#ff9800"  # orange

        metrics_html = f"""
            <div style="margin: 20px 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
                <h2 style="margin: 0 0 15px 0; text-align: center;">📊 Test Data Performance</h2>
                <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                    <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 8px; margin: 5px; min-width: 120px;">
                        <div style="font-size: 36px; font-weight: bold; color: {wape_color};">{wape_display}</div>
                        <div style="font-size: 14px; opacity: 0.9;">Overall WAPE</div>
                    </div>
                    <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 8px; margin: 5px; min-width: 120px;">
                        <div style="font-size: 36px; font-weight: bold;">{accuracy_display}</div>
                        <div style="font-size: 14px; opacity: 0.9;">Forecast Accuracy</div>
                    </div>
                    <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 8px; margin: 5px; min-width: 120px;">
                        <div style="font-size: 24px; font-weight: bold;">{assessment}</div>
                        <div style="font-size: 14px; opacity: 0.9;">Assessment</div>
                    </div>
                    <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 8px; margin: 5px; min-width: 120px;">
                        <div style="font-size: 24px; font-weight: bold;">{deployment_status}</div>
                        <div style="font-size: 14px; opacity: 0.9;">Deploy Ready</div>
                    </div>
                </div>
            </div>
        """

    # Removed backtest metrics section

    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; color: #333; }}
            .header {{ background-color: #4caf50; color: white; padding: 25px; border-radius: 5px; text-align: center; }}
            .content {{ padding: 20px; background-color: #f5f5f5; border-radius: 5px; margin-top: 10px; }}
            .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .stat-box {{ background-color: white; padding: 20px; border-radius: 5px; text-align: center; flex: 1; margin: 0 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .stat-value {{ font-size: 32px; font-weight: bold; color: #4caf50; }}
            .stat-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
            table {{ width: 100%; border-collapse: collapse; background-color: white; margin: 20px 0; }}
            th {{ background-color: #4caf50; color: white; padding: 12px; text-align: left; }}
            .footer {{ margin-top: 20px; padding: 20px; background-color: #e8f5e9; border-radius: 5px; border-left: 4px solid #4caf50; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1 style="margin: 0;">🎉 Pipeline Complete!</h1>
            <p style="margin: 10px 0 0 0; font-size: 18px;">All crews executed successfully</p>
        </div>
        <div class="content">
            {metrics_html}
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
            <h3 style="margin-top: 0; color: #4caf50;">✅ All Forecasting Artifacts Generated</h3>
            <p style="margin: 5px 0;">
                • Check the <strong>diagnostics report</strong> for comprehensive performance analysis<br>
                • Review model specs and forecasts in the <strong>model_artifacts</strong> directory<br>
                • Examine diagnostic charts for visual insights
            </p>
        </div>
    </body>
    </html>
    """

    send_outlook_desktop_email(to_email, subject, html_body)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: str = "pipeline_runner.log", trace_dir: Optional[str] = None):
    """
    Configure logging to both file and console.

    If trace_dir is provided, also sets up comprehensive CrewAI trace logging.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Set up CrewAI trace logging if trace_dir is provided
    if trace_dir:
        try:
            from utils.trace_logging import setup_crewai_trace_logging, set_crewai_logging_env_vars
            set_crewai_logging_env_vars()
            setup_crewai_trace_logging(
                artifact_base_path=trace_dir,
                log_level="DEBUG",
                also_log_to_console=False,  # Avoid duplicate console output
            )
            logging.info(f"CrewAI trace logging enabled at: {trace_dir}/trace_logs/")
        except ImportError as e:
            logging.warning(f"Could not enable trace logging: {e}")


# =============================================================================
# Crew Runners
# =============================================================================

def run_eda_crew_with_notifications(
    llm,
    config,
    config_yaml_path: str,
    to_email: Optional[str],
) -> Dict[str, Any]:
    """Run EDA crew with email notifications and token tracking.

    Uses DETERMINISTIC EDA for core analysis (no LLM for per_key_metrics.csv etc.)
    LLM is only used for creating context files with rationale.
    """
    from crews.eda_crew import run_eda_deterministic

    crew_name = "EDA Crew"
    crew_number = 1
    total_crews = 8

    logging.info(f"\n{'='*70}")
    logging.info(f"[{crew_number}/{total_crews}] Starting {crew_name}")
    logging.info(f"{'='*70}")

    send_crew_start_notification(to_email, crew_name, crew_number, total_crews)

    # Start token tracking for this crew
    try:
        from utils.cost_tracking import get_cost_tracker
        cost_tracker = get_cost_tracker()
        cost_tracker.start_crew(crew_name)
    except ImportError:
        cost_tracker = None

    start_time = datetime.now()

    try:
        # Use DETERMINISTIC EDA - core analysis without LLM, only context files use LLM
        result = run_eda_deterministic(config=config, llm=llm)

        duration = (datetime.now() - start_time).total_seconds()

        # End token tracking and save crew cost report
        tokens_info = ""
        if cost_tracker:
            try:
                eda_output_dir = os.path.join(config.artifact_base_path, "eda_output")
                crew_report = cost_tracker.end_crew(crew_name, eda_output_dir)
                tokens_info = f"\n  - Tokens: {crew_report.total_input_tokens:,} input / {crew_report.total_output_tokens:,} output"
                tokens_info += f"\n  - Est. Cost: ${crew_report.total_cost_usd:.4f}"
            except Exception as e:
                logging.warning(f"Could not save crew cost report: {e}")

        summary = f"""EDA analysis complete.
  - Global summary: {result.global_eda_summary_path}
  - Per-key metrics: {result.per_key_metrics_path}
  - Report: {result.eda_report_markdown_path}{tokens_info}
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

        # Still end tracking even on failure
        if cost_tracker:
            try:
                eda_output_dir = os.path.join(config.artifact_base_path, "eda_output")
                cost_tracker.end_crew(crew_name, eda_output_dir)
            except Exception:
                pass

        send_crew_failure_notification(
            to_email, crew_name, crew_number, total_crews, error_msg, tb
        )

        logging.error(f"❌ {crew_name} failed after {duration:.2f} seconds")
        raise


def run_feature_availability_with_notifications(
    llm,
    config,
    config_yaml_path: str,
    to_email: Optional[str],
) -> Dict[str, Any]:
    """Run Feature Availability Detection with email notifications.

    This step auto-detects which features are available in future periods
    and generates frozen embedding specs for history-only features.
    Runs between EDA and Segmentation crews.
    """
    from crews.feature_availability_crew import run_feature_availability_crew

    crew_name = "Feature Availability Detection"
    crew_number = 2
    total_crews = 8  # Updated: 8 stages now

    logging.info(f"\n{'='*70}")
    logging.info(f"[{crew_number}/{total_crews}] Starting {crew_name}")
    logging.info(f"{'='*70}")

    send_crew_start_notification(to_email, crew_name, crew_number, total_crews)

    try:
        from utils.cost_tracking import get_cost_tracker
        cost_tracker = get_cost_tracker()
        cost_tracker.start_crew(crew_name)
    except ImportError:
        cost_tracker = None

    start_time = datetime.now()

    try:
        result = run_feature_availability_crew(
            llm=llm,
            config=config,
            config_yaml_path=config_yaml_path,
        )

        if not result.success:
            raise RuntimeError(f"Feature Availability Detection failed: {result.error}")

        duration = (datetime.now() - start_time).total_seconds()

        tokens_info = ""
        if cost_tracker:
            try:
                crew_report = cost_tracker.end_crew(crew_name, result.output_dir)
                tokens_info = f"\n  - Tokens: {crew_report.total_input_tokens:,} input / {crew_report.total_output_tokens:,} output"
                tokens_info += f"\n  - Est. Cost: ${crew_report.total_cost_usd:.4f}"
            except Exception as e:
                logging.warning(f"Could not save crew cost report: {e}")

        summary = f"""Feature Availability Detection complete.
  - Features analyzed: {result.n_features_analyzed}
  - Known in future: {result.n_known_in_future}
  - History only: {result.n_history_only}
  - Partially known: {result.n_partially_known}
  - Excluded: {result.n_excluded}
  - Detected cutoff: {result.detected_cutoff}{tokens_info}
"""

        send_crew_completion_notification(
            to_email, crew_name, crew_number, total_crews, duration, summary
        )

        logging.info(f"Feature Availability Detection completed in {duration:.2f} seconds")
        return {"name": crew_name, "duration": duration / 60.0, "result": result}

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        tb = traceback.format_exc()

        if cost_tracker:
            try:
                output_dir = os.path.join(config.artifact_base_path, "feature_availability_output")
                cost_tracker.end_crew(crew_name, output_dir)
            except Exception:
                pass

        send_crew_failure_notification(
            to_email, crew_name, crew_number, total_crews, error_msg, tb
        )

        logging.error(f"Feature Availability Detection failed after {duration:.2f} seconds")
        raise


def run_segmentation_crew_with_notifications(
    llm,
    config,
    config_yaml_path: str,
    to_email: Optional[str],
) -> Dict[str, Any]:
    """Run Segmentation crew with email notifications and token tracking."""
    from crews.segmentation_crew import run_segmentation_crew

    crew_name = "Segmentation Crew"
    crew_number = 3
    total_crews = 8

    logging.info(f"\n{'='*70}")
    logging.info(f"[{crew_number}/{total_crews}] Starting {crew_name}")
    logging.info(f"{'='*70}")

    send_crew_start_notification(to_email, crew_name, crew_number, total_crews)

    # Start token tracking for this crew
    try:
        from utils.cost_tracking import get_cost_tracker
        cost_tracker = get_cost_tracker()
        cost_tracker.start_crew(crew_name)
    except ImportError:
        cost_tracker = None

    start_time = datetime.now()

    try:
        result = run_segmentation_crew(llm=llm, config=config, config_yaml_path=config_yaml_path)

        duration = (datetime.now() - start_time).total_seconds()

        # End token tracking and save crew cost report
        tokens_info = ""
        if cost_tracker:
            try:
                seg_output_dir = os.path.join(config.artifact_base_path, "segmentation_output")
                crew_report = cost_tracker.end_crew(crew_name, seg_output_dir)
                tokens_info = f"\n  - Tokens: {crew_report.total_input_tokens:,} input / {crew_report.total_output_tokens:,} output"
                tokens_info += f"\n  - Est. Cost: ${crew_report.total_cost_usd:.4f}"
            except Exception as e:
                logging.warning(f"Could not save crew cost report: {e}")

        summary = f"""Segmentation complete.
  - Segments: {result.per_key_with_segments_path}
  - Modeling strategy: {result.modeling_strategy_path}
  - Feature recommendations: {result.feature_recommendations_path}
  - Report: {result.segmentation_report_markdown_path}{tokens_info}
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

        # Still end tracking even on failure
        if cost_tracker:
            try:
                seg_output_dir = os.path.join(config.artifact_base_path, "segmentation_output")
                cost_tracker.end_crew(crew_name, seg_output_dir)
            except Exception:
                pass

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
    """Run Feature Engineering crew with email notifications and token tracking."""
    from crews.feature_crew import run_feature_crew

    crew_name = "Feature Engineering Crew"
    crew_number = 4
    total_crews = 8

    logging.info(f"\n{'='*70}")
    logging.info(f"[{crew_number}/{total_crews}] Starting {crew_name}")
    logging.info(f"{'='*70}")

    send_crew_start_notification(to_email, crew_name, crew_number, total_crews)

    # Start token tracking for this crew
    try:
        from utils.cost_tracking import get_cost_tracker
        cost_tracker = get_cost_tracker()
        cost_tracker.start_crew(crew_name)
    except ImportError:
        cost_tracker = None

    start_time = datetime.now()

    try:
        result = run_feature_crew(llm=llm, config=config, config_yaml_path=config_yaml_path)

        duration = (datetime.now() - start_time).total_seconds()

        # End token tracking and save crew cost report
        tokens_info = ""
        if cost_tracker:
            try:
                feature_output_dir = os.path.join(config.artifact_base_path, "feature_output")
                crew_report = cost_tracker.end_crew(crew_name, feature_output_dir)
                tokens_info = f"\n  - Tokens: {crew_report.total_input_tokens:,} input / {crew_report.total_output_tokens:,} output"
                tokens_info += f"\n  - Est. Cost: ${crew_report.total_cost_usd:.4f}"
            except Exception as e:
                logging.warning(f"Could not save crew cost report: {e}")

        summary = f"""Feature engineering complete.
  - Metadata: {result.feature_metadata_path}
  - Quality summary: {result.feature_quality_summary_path}
  - Report: {result.feature_report_markdown_path}{tokens_info}
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

        # Still end tracking even on failure
        if cost_tracker:
            try:
                feature_output_dir = os.path.join(config.artifact_base_path, "feature_output")
                cost_tracker.end_crew(crew_name, feature_output_dir)
            except Exception:
                pass

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
    """Run Model Training crew with email notifications and token tracking."""
    from crews.training_crew import run_training_crew

    crew_name = "Model Training Crew"
    crew_number = 5
    total_crews = 8

    logging.info(f"\n{'='*70}")
    logging.info(f"[{crew_number}/{total_crews}] Starting {crew_name}")
    logging.info(f"{'='*70}")

    send_crew_start_notification(to_email, crew_name, crew_number, total_crews)

    # Start token tracking for this crew
    try:
        from utils.cost_tracking import get_cost_tracker
        cost_tracker = get_cost_tracker()
        cost_tracker.start_crew(crew_name)
    except ImportError:
        cost_tracker = None

    start_time = datetime.now()

    try:
        result = run_training_crew(llm=llm, config=config, config_yaml_path=config_yaml_path)

        duration = (datetime.now() - start_time).total_seconds()

        # End token tracking and save crew cost report
        tokens_info = ""
        if cost_tracker:
            try:
                model_output_dir = os.path.join(config.artifact_base_path, "model_artifacts")
                crew_report = cost_tracker.end_crew(crew_name, model_output_dir)
                tokens_info = f"\n  - Tokens: {crew_report.total_input_tokens:,} input / {crew_report.total_output_tokens:,} output"
                tokens_info += f"\n  - Est. Cost: ${crew_report.total_cost_usd:.4f}"
            except Exception as e:
                logging.warning(f"Could not save crew cost report: {e}")

        summary = f"""Model training complete.
  - Model specs: {result.final_model_specs_path}
  - Val forecasts directory: {result.model_dir}
  - Test forecasts directory: {result.model_dir}{tokens_info}
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

        # Still end tracking even on failure
        if cost_tracker:
            try:
                model_output_dir = os.path.join(config.artifact_base_path, "model_artifacts")
                cost_tracker.end_crew(crew_name, model_output_dir)
            except Exception:
                pass

        send_crew_failure_notification(
            to_email, crew_name, crew_number, total_crews, error_msg, tb
        )

        logging.error(f"❌ {crew_name} failed after {duration:.2f} seconds")
        raise


def run_inference_with_notifications(
    config,
    to_email: Optional[str],
) -> Dict[str, Any]:
    """Run Inference pipeline with email notifications.

    Note: This is NOT a CrewAI crew - it's a deterministic pipeline that generates
    forward forecasts on the test period. This gives us true model performance metrics.
    """
    from utils.inference import run_inference_pipeline

    step_name = "Inference Pipeline"
    step_number = 5
    total_steps = 7

    logging.info(f"\n{'='*70}")
    logging.info(f"[{step_number}/{total_steps}] Starting {step_name}")
    logging.info(f"{'='*70}")

    send_crew_start_notification(to_email, step_name, step_number, total_steps)

    start_time = datetime.now()

    try:
        result = run_inference_pipeline(config=config)

        duration = (datetime.now() - start_time).total_seconds()

        if result.success:
            parquet_line = f"\n  - Forecasts (parquet): {result.forecasts_parquet_path}" if result.forecasts_parquet_path else ""
            summary = f"""Inference pipeline complete.
  - Forecasts: {result.forecasts_path}{parquet_line}
  - Summary: {result.summary_path}
  - Total keys: {result.total_keys}
  - New keys: {result.new_keys_count}
  - Existing keys: {result.existing_keys_count}
  - Total forecasts: {result.total_forecasts}
  - Forecast horizon: {result.forecast_horizon}
"""

            send_crew_completion_notification(
                to_email, step_name, step_number, total_steps, duration, summary
            )

            logging.info(f"✅ {step_name} completed in {duration:.2f} seconds")
            return {"name": step_name, "duration": duration / 60.0, "result": result}
        else:
            raise RuntimeError(f"Inference pipeline failed: {result.error_message}")

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        tb = traceback.format_exc()

        send_crew_failure_notification(
            to_email, step_name, step_number, total_steps, error_msg, tb
        )

        logging.error(f"❌ {step_name} failed after {duration:.2f} seconds")
        raise


def run_diagnostic_crew_with_notifications(
    llm,
    config,
    config_yaml_path: str,
    to_email: Optional[str],
) -> Dict[str, Any]:
    """Run Diagnostic crew with email notifications and token tracking."""
    from crews.diagnostic_crew import run_diagnostic_crew

    crew_name = "Diagnostic Crew"
    crew_number = 7
    total_crews = 8

    logging.info(f"\n{'='*70}")
    logging.info(f"[{crew_number}/{total_crews}] Starting {crew_name}")
    logging.info(f"{'='*70}")

    send_crew_start_notification(to_email, crew_name, crew_number, total_crews)

    # Start token tracking for this crew
    try:
        from utils.cost_tracking import get_cost_tracker
        cost_tracker = get_cost_tracker()
        cost_tracker.start_crew(crew_name)
    except ImportError:
        cost_tracker = None

    start_time = datetime.now()

    try:
        result = run_diagnostic_crew(llm=llm, config=config, config_yaml_path=config_yaml_path)

        duration = (datetime.now() - start_time).total_seconds()

        # End token tracking and save crew cost report
        tokens_info = ""
        if cost_tracker:
            try:
                diag_output_dir = os.path.join(config.artifact_base_path, "diagnostics_output")
                crew_report = cost_tracker.end_crew(crew_name, diag_output_dir)
                tokens_info = f"\n  - Tokens: {crew_report.total_input_tokens:,} input / {crew_report.total_output_tokens:,} output"
                tokens_info += f"\n  - Est. Cost: ${crew_report.total_cost_usd:.4f}"
            except Exception as e:
                logging.warning(f"Could not save crew cost report: {e}")

        summary = f"""Diagnostic analysis complete.
  - Group diagnostics: {result.group_diagnostics_path}
  - Segment diagnostics: {result.segment_diagnostics_path}
  - Summary JSON: {result.diagnostics_summary_path}
  - Report: {result.diagnostics_report_markdown_path}{tokens_info}
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

        # Still end tracking even on failure
        if cost_tracker:
            try:
                diag_output_dir = os.path.join(config.artifact_base_path, "diagnostics_output")
                cost_tracker.end_crew(crew_name, diag_output_dir)
            except Exception:
                pass

        send_crew_failure_notification(
            to_email, crew_name, crew_number, total_crews, error_msg, tb
        )

        logging.error(f"❌ {crew_name} failed after {duration:.2f} seconds")
        raise


def run_backtest_with_notifications(
    config,
    to_email: Optional[str],
) -> Dict[str, Any]:
    """Run Rolling-Origin Backtesting with email notifications.

    Note: This runs after diagnostics to provide comprehensive model validation
    across multiple forecast origins.
    """
    from utils.backtesting import run_rolling_origin_backtest, generate_backtest_origins

    step_name = "Rolling-Origin Backtest"
    step_number = 7
    total_steps = 7

    logging.info(f"\n{'='*70}")
    logging.info(f"[{step_number}/{total_steps}] Starting {step_name}")
    logging.info(f"{'='*70}")

    send_crew_start_notification(to_email, step_name, step_number, total_steps)

    start_time = datetime.now()

    try:
        # Generate origins to get total count
        origins = generate_backtest_origins(
            val_end=config.val_end,
            test_start=config.test_start,
            test_end=config.test_end,
            forecast_horizon=config.forecast_horizon,
        )
        total_origins = len(origins)

        logging.info(f"Backtest will run {total_origins} origins")

        # Create callback for per-origin email notifications
        def origin_email_callback(origin_idx, origin_result, metrics):
            """Send email after each origin completes."""
            if not to_email:
                return
            try:
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                wape = metrics.get('wape', 0) * 100
                bias_pct = metrics.get('bias_pct', 0)

                if wape < 15:
                    status_emoji = "🟢"
                elif wape < 25:
                    status_emoji = "🟡"
                else:
                    status_emoji = "🔴"

                subject = f"[FEU-Agentic-Forecasting Backtest] {status_emoji} Origin {origin_idx}/{total_origins} - WAPE: {wape:.1f}%"

                html_body = f"""
                <html>
                <body style="font-family: Arial, sans-serif;">
                    <h3>{status_emoji} Backtest Origin {origin_idx}/{total_origins} Complete</h3>
                    <p><strong>Origin Period:</strong> {origin_result.get('origin_period', 'N/A')}</p>
                    <p><strong>Train Cutoff:</strong> {origin_result.get('train_cutoff', 'N/A')}</p>
                    <p><strong>WAPE:</strong> {wape:.2f}%</p>
                    <p><strong>Bias:</strong> {bias_pct:+.2f}%</p>
                    <p><strong>Forecasts:</strong> {metrics.get('n_forecasts', 0):,}</p>
                    <p><em>Timestamp: {now}</em></p>
                </body>
                </html>
                """
                send_outlook_desktop_email(to_email, subject, html_body)
            except Exception as e:
                logging.warning(f"Failed to send origin email: {e}")

        # Run backtesting
        result = run_rolling_origin_backtest(
            config=config,
            output_dir=None,  # Use default
            verbose=False,
            origin_callback=origin_email_callback if to_email else None,
        )

        duration = (datetime.now() - start_time).total_seconds()

        if result.success:
            # Read overall metrics
            import pandas as pd
            try:
                metrics_df = pd.read_csv(result.metrics_path)
                overall_row = metrics_df[metrics_df['group_type'] == 'overall'].iloc[0]
                overall_wape = overall_row.get('wape', 0) * 100
                overall_bias = overall_row.get('bias_pct', 0)
            except Exception:
                overall_wape = 0
                overall_bias = 0

            summary = f"""Backtesting complete.
  - Total Origins: {result.total_origins}
  - Total Forecasts: {result.total_forecasts:,}
  - Forecast Horizon: {result.forecast_horizon}
  - Overall WAPE: {overall_wape:.2f}%
  - Overall Bias: {overall_bias:+.2f}%
  - Forecasts: {result.forecasts_path}
  - Metrics: {result.metrics_path}
  - Summary: {result.summary_path}
"""

            send_crew_completion_notification(
                to_email, step_name, step_number, total_steps, duration, summary
            )

            logging.info(f"✅ {step_name} completed in {duration:.2f} seconds")
            return {"name": step_name, "duration": duration / 60.0, "result": result}
        else:
            raise RuntimeError(f"Backtesting failed: {result.error_message}")

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        tb = traceback.format_exc()

        send_crew_failure_notification(
            to_email, step_name, step_number, total_steps, error_msg, tb
        )

        logging.error(f"❌ {step_name} failed after {duration:.2f} seconds")
        raise


# =============================================================================
# Main Pipeline
# =============================================================================

def run_complete_pipeline(
    config_yaml_path: str,
    to_email: Optional[str] = None,
    enable_trace: bool = True,
) -> bool:
    """
    Run the complete forecasting pipeline.

    Parameters
    ----------
    config_yaml_path : str
        Path to config YAML file
    to_email : Optional[str]
        Email address for notifications (if None, no emails sent)
    enable_trace : bool
        Whether to enable detailed CrewAI trace logging (default: True)

    Returns
    -------
    bool
        True if pipeline completed successfully, False otherwise
    """
    from config.schema import load_config_from_yaml
    from config.llm_config import get_llm

    pipeline_start = datetime.now()

    logging.info("\n" + "="*70)
    logging.info("FEU-AGENTIC-FORECASTING DEMAND FORECASTING PIPELINE")
    logging.info("="*70)
    logging.info(f"Config: {config_yaml_path}")
    logging.info(f"Email notifications: {'ENABLED' if to_email else 'DISABLED'}")
    if to_email:
        logging.info(f"Recipient: {to_email}")
    logging.info(f"Trace logging: {'ENABLED' if enable_trace else 'DISABLED'}")
    logging.info("="*70 + "\n")

    try:
        # Load configuration
        logging.info("Loading configuration...")
        cfg = load_config_from_yaml(config_yaml_path)
        llm = get_llm(config_path=config_yaml_path)
        logging.info("✓ Configuration loaded\n")

        # Auto-detect train/val/test splits if not specified in config
        if not all([cfg.train_start, cfg.train_end, cfg.val_start, cfg.val_end]):
            logging.info("Auto-detecting train/val/test splits from data...")
            from utils.agent_utilities import load_source_data
            from utils.period_utils import normalise_period_column
            _source_df = load_source_data(cfg.input_data_path)
            _source_df = normalise_period_column(_source_df, cfg.timestamp_col)
            cfg.auto_detect_splits(_source_df)
            del _source_df  # Free memory
            logging.info(f"✓ Splits: train={cfg.train_start}..{cfg.train_end}, "
                         f"val={cfg.val_start}..{cfg.val_end}, "
                         f"test={cfg.test_start}..{cfg.test_end}\n")

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

        # Initialize cost tracking for the pipeline with LiteLLM callbacks
        # This MUST be done BEFORE creating any crews
        try:
            from utils.cost_tracking import (
                get_cost_tracker,
                setup_litellm_callbacks,
                cleanup_litellm_callbacks,
                is_token_tracking_active,
            )

            # Setup LiteLLM callbacks to intercept ALL LLM calls
            # This is the KEY to getting token counts - it bypasses CrewAI's
            # limitations by hooking directly into LiteLLM
            callback_setup = setup_litellm_callbacks()

            cost_tracker = get_cost_tracker()
            cost_tracker.start_pipeline()

            # Get model ID for pricing calculations
            model_id = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0")
            if not model_id.startswith("bedrock/"):
                # Normalize to just the model part for pricing lookup
                model_id = model_id.replace("bedrock/", "")
            cost_tracker.set_model(model_id)

            if callback_setup:
                logging.info("✓ Token tracking enabled via LiteLLM callbacks")
                logging.info(f"  Model for pricing: {model_id}")
            else:
                logging.warning("⚠️ LiteLLM callback setup failed, token tracking may be incomplete")

            logging.info("✓ Cost tracking enabled\n")
        except ImportError as e:
            cost_tracker = None
            cleanup_litellm_callbacks = None
            logging.warning(f"Could not enable cost tracking: {e}")

        crew_summaries = []

        def _check_prereq(stage_name, required_dirs):
            """Validate prerequisite stage outputs exist before running a stage."""
            for d in required_dirs:
                path = os.path.join(cfg.artifact_base_path, d)
                if not os.path.isdir(path):
                    raise FileNotFoundError(
                        f"{stage_name} requires {d}/ output but it doesn't exist at {path}. "
                        f"Run the prerequisite stage first."
                    )

        # 1. EDA Crew
        eda_summary = run_eda_crew_with_notifications(llm, cfg, config_yaml_path, to_email)
        crew_summaries.append(eda_summary)

        # 2. Feature Availability Detection (auto-detect which features are available in future)
        if getattr(cfg.design, 'enable_feature_availability_detection', True):
            fa_summary = run_feature_availability_with_notifications(llm, cfg, config_yaml_path, to_email)
            crew_summaries.append(fa_summary)
        else:
            logging.info("Feature Availability Detection disabled in config — skipping")

        # 3. Segmentation Crew
        _check_prereq("Segmentation", ["eda_output"])
        seg_summary = run_segmentation_crew_with_notifications(llm, cfg, config_yaml_path, to_email)
        crew_summaries.append(seg_summary)

        # 4. Feature Engineering Crew
        _check_prereq("Feature Engineering", ["eda_output"])
        feat_summary = run_feature_crew_with_notifications(llm, cfg, config_yaml_path, to_email)
        crew_summaries.append(feat_summary)

        # 5. Model Training Crew
        _check_prereq("Training", ["eda_output"])
        train_summary = run_training_crew_with_notifications(llm, cfg, config_yaml_path, to_email)
        crew_summaries.append(train_summary)

        # 6. Inference / Forward Forecast (if run_mode requires it)
        if cfg.should_forward_forecast:
            _check_prereq("Inference", ["model_artifacts"])
            inference_summary = run_inference_with_notifications(cfg, to_email)
            crew_summaries.append(inference_summary)

            # 7. Diagnostic Crew (analyzes inference results)
            diag_summary = run_diagnostic_crew_with_notifications(llm, cfg, config_yaml_path, to_email)
            crew_summaries.append(diag_summary)
        else:
            logging.info(f"Skipping Inference & Diagnostics (run_mode={cfg.run_mode})")

        # 8. Rolling-Origin Backtesting (if run_mode requires it)
        if cfg.should_backtest:
            backtest_summary = run_backtest_with_notifications(cfg, to_email)
            crew_summaries.append(backtest_summary)
        else:
            logging.info(f"Skipping Backtesting (run_mode={cfg.run_mode})")

        # Pipeline completed
        total_duration = (datetime.now() - pipeline_start).total_seconds()

        logging.info("\n" + "="*70)
        logging.info("🎉 PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("="*70)
        logging.info(f"Total duration: {total_duration / 60.0:.2f} minutes")
        logging.info("\nCrew Summary:")
        for summary in crew_summaries:
            logging.info(f"  ✅ {summary['name']}: {summary['duration']:.2f} min")

        # Log context flow summary
        logging.info("\nContext Flow Summary:")
        logging.info("-" * 40)
        logging.info("  EDA → Segmentation:           eda_to_segmentation_context.json")
        logging.info("  EDA → Feature:                eda_to_feature_context.json")
        logging.info("  Feature Avail → Feature:      feature_availability_to_feature_context.json")
        logging.info("  Seg → Feature:                segmentation_to_feature_context.json")
        logging.info("  Seg → Training:               segmentation_to_training_context.json")
        logging.info("  Feature → Training:           feature_to_training_context.json")
        logging.info("  Training → Diag:              training_to_diagnostic_context.json")
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

                # Log token summary
                logging.info("\nToken Usage Summary:")
                logging.info("-" * 40)
                logging.info(f"  Total Input Tokens:  {pipeline_cost_report.total_input_tokens:,}")
                logging.info(f"  Total Output Tokens: {pipeline_cost_report.total_output_tokens:,}")
                logging.info(f"  Total Tokens:        {pipeline_cost_report.total_tokens:,}")
                logging.info(f"  Estimated Cost:      ${pipeline_cost_report.total_cost_usd:.4f}")
                logging.info("-" * 40)
            except Exception as e:
                logging.warning(f"Could not save cost summary: {e}")

        # Cleanup LiteLLM callbacks
        try:
            from utils.cost_tracking import cleanup_litellm_callbacks
            cleanup_litellm_callbacks()
        except Exception:
            pass

        logging.info("="*70 + "\n")

        send_pipeline_completion_notification(to_email, total_duration, crew_summaries, cfg.artifact_base_path)

        return True

    except Exception as e:
        total_duration = (datetime.now() - pipeline_start).total_seconds()

        logging.error("\n" + "="*70)
        logging.error("❌ PIPELINE FAILED")
        logging.error("="*70)
        logging.error(f"Duration before failure: {total_duration / 60.0:.2f} minutes")
        logging.error(f"Error: {e}")
        logging.error("="*70 + "\n")

        # Cleanup LiteLLM callbacks even on failure
        try:
            from utils.cost_tracking import cleanup_litellm_callbacks
            cleanup_litellm_callbacks()
        except Exception:
            pass

        # Failure notification already sent by individual crew runner
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run FEU-Agentic-Forecasting demand forecasting pipeline with optional email notifications'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file (e.g., config/config.yaml)'
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
        to_email=to_email,
        enable_trace=enable_trace,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
