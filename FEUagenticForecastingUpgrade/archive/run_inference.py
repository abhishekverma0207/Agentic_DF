#!/usr/bin/env python3
"""
FEU-Agentic-Forecasting Inference Pipeline Runner

This script runs the production inference pipeline to generate forecasts for all keys
in the test/inference period with intelligent handling of new keys (keys present in
inference data but not in the original training manifest).

Features:
- Detects new keys not present in training manifest
- Assigns new keys to existing segments using saved cluster model
- Updates training manifest with new keys
- Retrains all models with data up to val_end
- Generates recursive multi-step forecasts over inference period
- Applies validation-learned bias calibration factors

Usage:
    python run_inference.py --config config/config.yaml
    python run_inference.py --config config/config.yaml --output-dir ./custom_output
    python run_inference.py --config config/config.yaml --verbose

Example:
    # Standard run with default config
    python run_inference.py --config config/config.yaml

    # Run with custom output directory
    python run_inference.py --config config/config.yaml --output-dir ./forecasts

    # Run with verbose logging
    python run_inference.py --config config/config.yaml --verbose
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(console_handler)

    # Suppress noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


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
║                       INFERENCE PIPELINE                                      ║
║                                                                               ║
║   Production forecasting with automatic new key detection and assignment      ║
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
    print(f"  Inference Period: {config.test_start} to {config.test_end}")
    print(f"  Bias Calibration: {'Enabled' if config.design.apply_bias_calibration else 'Disabled'}")
    print("=" * 70 + "\n")


def print_result_summary(result) -> None:
    """Print result summary."""
    print("\n" + "=" * 70)
    print("INFERENCE RESULTS")
    print("=" * 70)

    if result.success:
        print(f"  Status:            ✅ SUCCESS")
        print(f"  Total Keys:        {result.total_keys}")
        print(f"  New Keys:          {result.new_keys_count}")
        print(f"  Existing Keys:     {result.existing_keys_count}")
        print(f"  Dead Keys:         {result.dead_keys_count}")
        print(f"  Models Retrained:  {result.models_retrained}")
        print(f"  Total Forecasts:   {result.total_forecasts}")
        print(f"  Forecast Horizon:  {result.forecast_horizon}")

        if result.new_keys_by_segment:
            print(f"\n  New Keys by Segment:")
            for seg, count in sorted(result.new_keys_by_segment.items()):
                print(f"    Segment {seg}: {count} keys")

        print(f"\n  Output Files:")
        print(f"    Forecasts:       {result.forecasts_path}")
        print(f"    Summary:         {result.summary_path}")
        print(f"    Manifest Backup: {result.backup_manifest_path}")
    else:
        print(f"  Status:            ❌ FAILED")
        print(f"  Error:             {result.error_message}")

    print("=" * 70 + "\n")


def run_inference_only(config_path: str, output_dir: str = None, verbose: bool = False) -> None:
    """
    Run inference pipeline programmatically (for use by databricks_runner.py and other scripts).

    Args:
        config_path: Path to config YAML file
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

    # Format-agnostic check for train/val/test_features.csv entries: they
    # may have been saved as parquet (the new fast-path default) or CSV
    # (legacy). Either form counts as "present".
    from utils.feature_io import features_intermediate_exists
    _features_bases = {'train_features.csv', 'val_features.csv', 'test_features.csv'}
    for subdir, filename in required_files:
        if filename in _features_bases:
            base = filename[:-4]  # strip .csv
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

    # Run inference pipeline
    logger.info("Starting inference pipeline...")
    start_time = datetime.now()

    try:
        from utils.inference import run_inference_pipeline

        result = run_inference_pipeline(
            config=config,
            output_dir=output_dir,
            verbose=verbose,
        )

    except Exception as e:
        import traceback
        logger.error(f"Pipeline failed with exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    print_result_summary(result)

    # Calculate elapsed time
    elapsed = datetime.now() - start_time
    logger.info(f"Total elapsed time: {elapsed}")

    # Exit with appropriate code
    if result.success:
        logger.info("Inference pipeline completed successfully!")
    else:
        logger.error("Inference pipeline failed!")
        sys.exit(1)


def main():
    """Main entry point for inference pipeline."""
    parser = argparse.ArgumentParser(
        description='Generate production forecasts with intelligent new key handling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_inference.py --config config/config.yaml
  python run_inference.py --config config/config.yaml --verbose
  python run_inference.py --config config/config.yaml --output-dir ./forecasts
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
        help='Override output directory (default: {artifact_base}/model_artifacts)'
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

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)
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

    # Format-agnostic check for train/val/test_features.csv entries: they
    # may have been saved as parquet (the new fast-path default) or CSV
    # (legacy). Either form counts as "present".
    from utils.feature_io import features_intermediate_exists
    _features_bases = {'train_features.csv', 'val_features.csv', 'test_features.csv'}
    for subdir, filename in required_files:
        if filename in _features_bases:
            base = filename[:-4]  # strip .csv
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

    # Run inference pipeline
    logger.info("Starting inference pipeline...")
    start_time = datetime.now()

    try:
        from utils.inference import run_inference_pipeline

        result = run_inference_pipeline(
            config=config,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )

    except Exception as e:
        import traceback
        logger.error(f"Pipeline failed with exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Print result summary
    if not args.quiet:
        print_result_summary(result)

    # Calculate elapsed time
    elapsed = datetime.now() - start_time
    logger.info(f"Total elapsed time: {elapsed}")

    # Exit with appropriate code
    if result.success:
        logger.info("Inference pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("Inference pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
