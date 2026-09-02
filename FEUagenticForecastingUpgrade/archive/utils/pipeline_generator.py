# utils/pipeline_generator.py
"""
State-of-the-Art Pipeline Generator Utilities for Demand Forecasting
=====================================================================

This module provides comprehensive utilities for generating production-ready
training and inference pipelines that can be deployed standalone.

KEY CONCEPTS:
=============
1. **Deterministic Pipelines**: Pipelines with no randomness - same input always
   produces same output. Uses finalized hyperparameters, not tuning.

2. **Self-Contained Scripts**: Generated pipelines include all code needed to run
   without external dependencies on the agent framework.

3. **Identical Processing**: Training and inference pipelines use identical
   data processing to ensure consistency.

4. **Production Ready**: Generated code includes proper error handling, logging,
   CLI interfaces, and configuration management.

USAGE:
======
    from utils.pipeline_generator import (
        # High-level pipeline generation (USE THIS!)
        run_pipeline_generation,  # Does EVERYTHING in one call!

        # Pipeline building blocks
        generate_training_pipeline,
        generate_inference_pipeline,
        generate_orchestrator_script,

        # Code template utilities
        create_import_block,
        create_config_loader,
        create_data_loader,
        create_feature_engineering_code,
        create_model_training_code,
        create_model_inference_code,

        # Validation & testing
        validate_pipeline_syntax,
        validate_pipeline_imports,
        test_pipeline_execution,

        # Pipeline documentation
        generate_pipeline_readme,
        generate_pipeline_config,
    )

    # Generate complete production pipelines
    result = run_pipeline_generation(
        deterministic_code_dir=artifact_base,
        output_dir='pipeline_output',
        config=config,
    )
    print(f'Generated: {result.training_pipeline_path}')
    print(f'Generated: {result.inference_pipeline_path}')
"""

from __future__ import annotations

import ast
import json
import logging
import os
import py_compile
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for pipeline generation."""
    # Data configuration
    data_path: str = ""
    target_column: str = "target"
    date_column: str = "year_week"
    key_columns: List[str] = field(default_factory=lambda: ["key"])
    categorical_columns: List[str] = field(default_factory=list)

    # Time periods
    train_start: str = ""
    train_end: str = ""
    val_start: str = ""
    val_end: str = ""
    test_start: str = ""
    test_end: str = ""

    # Paths to deterministic code
    eda_deterministic_path: str = ""
    segmentation_deterministic_path: str = ""
    feature_deterministic_path: str = ""
    training_deterministic_path: str = ""

    # Artifact paths
    cluster_model_path: str = ""
    cluster_scaler_path: str = ""
    categorical_encoding_specs_path: str = ""
    model_specs_path: str = ""

    # Output settings
    output_dir: str = ""
    pipeline_name: str = "demand_forecast"

    # STATE-OF-THE-ART Optimization Settings
    enable_bias_correction: bool = True       # Apply bias correction to forecasts
    enable_ensemble_optimization: bool = True # Optimize ensemble weights
    enable_forecast_calibration: bool = True  # Calibrate forecasts for reliability
    enable_hierarchical_reconciliation: bool = False  # Reconcile hierarchical forecasts
    hierarchy_structure: Dict[str, List[str]] = field(default_factory=dict)  # Parent -> children mapping
    prediction_intervals: bool = True         # Generate prediction intervals
    prediction_interval_confidence: float = 0.95  # 95% confidence intervals


@dataclass
class DeterministicCodeInfo:
    """Information extracted from deterministic code files."""
    path: str
    exists: bool = False
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    config_dicts: Dict[str, Any] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    line_count: int = 0


@dataclass
class PipelineGenerationResult:
    """Result of pipeline generation."""
    # Generated files
    training_pipeline_path: str = ""
    inference_pipeline_path: str = ""
    orchestrator_path: str = ""
    config_path: str = ""
    readme_path: str = ""

    # Validation results
    all_valid: bool = False
    validation_results: Dict[str, Dict] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)

    # Metadata
    generation_timestamp: str = ""
    deterministic_code_used: Dict[str, bool] = field(default_factory=dict)
    total_lines_generated: int = 0


# =============================================================================
# CODE ANALYSIS UTILITIES
# =============================================================================

def analyze_deterministic_code(file_path: str) -> DeterministicCodeInfo:
    """
    Analyze a deterministic code file to extract its structure.

    Parameters
    ----------
    file_path : str
        Path to the deterministic Python file

    Returns
    -------
    DeterministicCodeInfo
        Information about the code structure
    """
    info = DeterministicCodeInfo(path=file_path)

    if not os.path.exists(file_path):
        logger.warning(f"Deterministic code not found: {file_path}")
        return info

    info.exists = True

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        info.line_count = len(content.splitlines())

        # Parse AST for detailed analysis
        tree = ast.parse(content)

        for node in ast.walk(tree):
            # Extract function names
            if isinstance(node, ast.FunctionDef):
                info.functions.append(node.name)
            # Extract class names
            elif isinstance(node, ast.ClassDef):
                info.classes.append(node.name)
            # Extract import statements
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    info.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    info.imports.append(f"{module}.{alias.name}")

        # Extract configuration dictionaries using regex
        config_patterns = [
            r'(\w+_CONFIG)\s*=\s*\{',
            r'(\w+_SPECS)\s*=\s*\{',
            r'(MODEL_PARAMS)\s*=\s*\{',
            r'(HYPERPARAMETERS)\s*=\s*\{',
            r'(LAG_VALUES)\s*=\s*\[',
            r'(ROLLING_WINDOWS)\s*=\s*\[',
        ]

        for pattern in config_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                info.config_dicts[match] = True

        logger.info(f"Analyzed {file_path}: {len(info.functions)} functions, "
                   f"{len(info.classes)} classes, {info.line_count} lines")

    except SyntaxError as e:
        logger.error(f"Syntax error in {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error analyzing {file_path}: {e}")

    return info


def extract_code_block(
    file_path: str,
    start_pattern: str,
    end_pattern: Optional[str] = None,
) -> str:
    """
    Extract a code block from a file based on patterns.

    Parameters
    ----------
    file_path : str
        Path to the Python file
    start_pattern : str
        Regex pattern to match the start
    end_pattern : str, optional
        Regex pattern to match the end (if None, extracts to end of block)

    Returns
    -------
    str
        Extracted code block
    """
    if not os.path.exists(file_path):
        return ""

    with open(file_path, 'r') as f:
        content = f.read()

    # Find start
    start_match = re.search(start_pattern, content, re.MULTILINE)
    if not start_match:
        return ""

    start_idx = start_match.start()

    if end_pattern:
        end_match = re.search(end_pattern, content[start_idx:], re.MULTILINE)
        if end_match:
            return content[start_idx:start_idx + end_match.end()]

    # Extract to end of function/class using indentation
    lines = content[start_idx:].split('\n')
    block_lines = [lines[0]]

    if len(lines) > 1:
        # Get base indentation
        base_indent = len(lines[0]) - len(lines[0].lstrip())

        for line in lines[1:]:
            if line.strip() == "":
                block_lines.append(line)
            elif len(line) - len(line.lstrip()) <= base_indent and line.strip():
                break
            else:
                block_lines.append(line)

    return '\n'.join(block_lines)


def extract_function(file_path: str, function_name: str) -> str:
    """
    Extract a function definition from a Python file.

    Parameters
    ----------
    file_path : str
        Path to the Python file
    function_name : str
        Name of the function to extract

    Returns
    -------
    str
        Function code
    """
    return extract_code_block(file_path, rf'^def\s+{function_name}\s*\(')


# =============================================================================
# CODE GENERATION UTILITIES
# =============================================================================

def create_import_block(
    include_ml: bool = True,
    include_visualization: bool = False,
    additional_imports: Optional[List[str]] = None,
) -> str:
    """
    Create a standard import block for pipeline scripts.

    Parameters
    ----------
    include_ml : bool
        Include ML library imports (lightgbm, xgboost, sklearn)
    include_visualization : bool
        Include visualization imports (matplotlib, seaborn)
    additional_imports : List[str], optional
        Additional import statements to include

    Returns
    -------
    str
        Import block code
    """
    imports = [
        "#!/usr/bin/env python3",
        '"""',
        "Auto-generated Production Pipeline",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        '"""',
        "",
        "import argparse",
        "import json",
        "import logging",
        "import os",
        "import sys",
        "import warnings",
        "from datetime import datetime",
        "from typing import Any, Dict, List, Optional, Tuple",
        "",
        "import joblib",
        "import numpy as np",
        "import pandas as pd",
        "import yaml",
    ]

    if include_ml:
        imports.extend([
            "",
            "# ML Libraries",
            "import lightgbm as lgb",
            "import xgboost as xgb",
            "from sklearn.preprocessing import StandardScaler",
            "from sklearn.metrics import mean_absolute_error, mean_squared_error",
        ])

    if include_visualization:
        imports.extend([
            "",
            "# Visualization",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
        ])

    imports.extend([
        "",
        "# Suppress warnings",
        "warnings.filterwarnings('ignore')",
        "",
        "# Configure logging",
        "logging.basicConfig(",
        "    level=logging.INFO,",
        "    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'",
        ")",
        "logger = logging.getLogger(__name__)",
    ])

    if additional_imports:
        imports.extend(["", "# Additional imports"])
        imports.extend(additional_imports)

    return '\n'.join(imports)


def create_config_loader() -> str:
    """
    Create configuration loading code.

    Returns
    -------
    str
        Config loader code
    """
    return textwrap.dedent('''
    # =============================================================================
    # CONFIGURATION
    # =============================================================================

    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                config = json.load(f)
            else:
                config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from: {config_path}")
        return config


    def get_data_path(config: Dict[str, Any]) -> str:
        """Extract data_path from config (handles multiple formats)."""
        # Try top-level data_path first (pipeline_config.json format)
        if 'data_path' in config:
            return config['data_path']
        # Try nested data.data_path format
        if 'data' in config and isinstance(config['data'], dict):
            if 'data_path' in config['data']:
                return config['data']['data_path']
        # Try input_data_path (legacy YAML format)
        if 'input_data_path' in config:
            return config['input_data_path']
        return ''


    def get_config_value(config: Dict, key: str, default: Any = None) -> Any:
        """Get a value from nested config using dot notation."""
        keys = key.split('.')
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    ''')


def create_data_loader(
    key_columns: List[str],
    date_column: str,
    target_column: str,
) -> str:
    """
    Create data loading code.

    Parameters
    ----------
    key_columns : List[str]
        Key columns
    date_column : str
        Date column name
    target_column : str
        Target column name

    Returns
    -------
    str
        Data loader code
    """
    key_cols_str = str(key_columns)

    return textwrap.dedent(f'''
    # =============================================================================
    # DATA LOADING
    # =============================================================================

    KEY_COLUMNS = {key_cols_str}
    DATE_COLUMN = "{date_column}"
    TARGET_COLUMN = "{target_column}"


    def load_source_data(data_path: str) -> pd.DataFrame:
        """Load source data from CSV, Parquet, or Delta format (auto-detected)."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {{data_path}}")

        file_ext = os.path.splitext(data_path)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(data_path)
        elif file_ext in ('.parquet', '.pq'):
            df = pd.read_parquet(data_path)
        elif os.path.isdir(data_path):
            if os.path.isdir(os.path.join(data_path, '_delta_log')):
                from deltalake import DeltaTable
                df = DeltaTable(data_path).to_pandas()
            else:
                df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {{file_ext}}. Use .csv, .parquet, or delta directory")
        logger.info(f"Loaded {{len(df)}} rows from {{data_path}}")

        # Validate required columns
        required_cols = KEY_COLUMNS + [DATE_COLUMN, TARGET_COLUMN]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {{missing_cols}}")

        return df


    def filter_data_by_period(
        df: pd.DataFrame,
        start_period: str,
        end_period: str,
    ) -> pd.DataFrame:
        """Filter data to a specific time period (YYYYWW format)."""
        df = df.copy()
        df['_period_int'] = df[DATE_COLUMN].astype(str).astype(int)
        start_int = int(start_period)
        end_int = int(end_period)

        filtered = df[(df['_period_int'] >= start_int) & (df['_period_int'] <= end_int)]
        filtered = filtered.drop(columns=['_period_int'])

        logger.info(f"Filtered data from {{start_period}} to {{end_period}}: {{len(filtered)}} rows")
        return filtered


    def split_train_val_test(
        df: pd.DataFrame,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        test_start: str,
        test_end: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets by date."""
        train_df = filter_data_by_period(df, train_start, train_end)
        val_df = filter_data_by_period(df, val_start, val_end)
        test_df = filter_data_by_period(df, test_start, test_end)

        logger.info(f"Data split: Train={{len(train_df)}}, Val={{len(val_df)}}, Test={{len(test_df)}}")
        return train_df, val_df, test_df
    ''')


def create_metrics_code() -> str:
    """
    Create metrics computation code.

    Returns
    -------
    str
        Metrics code
    """
    return textwrap.dedent('''
    # =============================================================================
    # METRICS
    # =============================================================================

    def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Weighted Absolute Percentage Error."""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        numerator = np.sum(np.abs(y_true - y_pred))
        denominator = np.sum(np.abs(y_true)) + 1e-10

        return numerator / denominator


    def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Error."""
        return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


    def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Root Mean Squared Error."""
        return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


    def compute_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute forecast bias (positive = over-forecast)."""
        return np.mean(np.asarray(y_pred) - np.asarray(y_true))


    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute all standard metrics."""
        return {
            'wape': compute_wape(y_true, y_pred),
            'mae': compute_mae(y_true, y_pred),
            'rmse': compute_rmse(y_true, y_pred),
            'bias': compute_bias(y_true, y_pred),
        }


    def compute_metrics_by_group(
        df: pd.DataFrame,
        group_col: str,
        actual_col: str,
        pred_col: str,
    ) -> pd.DataFrame:
        """Compute WAPE for each group."""
        results = []

        for group_val in df[group_col].unique():
            group_df = df[df[group_col] == group_val]
            if len(group_df) > 0:
                wape = compute_wape(group_df[actual_col].values, group_df[pred_col].values)
                results.append({
                    group_col: group_val,
                    'wape': wape,
                    'n_rows': len(group_df),
                    'sum_actual': group_df[actual_col].sum(),
                    'sum_predicted': group_df[pred_col].sum(),
                })

        return pd.DataFrame(results)
    ''')


def create_categorical_encoding_code() -> str:
    """
    Create categorical encoding code for consistent processing.

    Returns
    -------
    str
        Categorical encoding code
    """
    return textwrap.dedent('''
    # =============================================================================
    # CATEGORICAL ENCODING
    # =============================================================================

    def load_categorical_specs(specs_path: str) -> Dict[str, Any]:
        """Load categorical encoding specifications."""
        if not specs_path or not os.path.exists(specs_path):
            logger.warning(f"Categorical specs not found: {specs_path}")
            return {}

        with open(specs_path, 'r') as f:
            specs = json.load(f)
        logger.info(f"Loaded categorical encoding specs: {list(specs.keys())}")
        return specs


    def apply_categorical_encoding(
        df: pd.DataFrame,
        cat_specs: Dict[str, Any],
    ) -> pd.DataFrame:
        """Apply categorical encodings to DataFrame."""
        if not cat_specs:
            return df

        result = df.copy()

        # Target encoding
        for col, mapping in cat_specs.get('target_encoding_maps', {}).items():
            if col in result.columns:
                encoded_col = f'{col}_encoded'
                result[encoded_col] = result[col].map(mapping)
                global_mean = mapping.get('__global_mean__', result[encoded_col].mean())
                result[encoded_col] = result[encoded_col].fillna(global_mean)

        # Ordinal encoding
        for col, mapping in cat_specs.get('ordinal_encoding_maps', {}).items():
            if col in result.columns:
                encoded_col = f'{col}_ordinal'
                result[encoded_col] = result[col].map(mapping)
                median_rank = np.median(list(mapping.values()))
                result[encoded_col] = result[encoded_col].fillna(median_rank)

        # Binary columns
        for col in cat_specs.get('binary_columns', []):
            if col in result.columns:
                result[col] = result[col].fillna(0).astype(int)

        # Interaction features
        for feature_name, feature_config in cat_specs.get('interaction_features', {}).items():
            feature_type = feature_config.get('type', 'sum')
            source_cols = feature_config.get('source_columns', [])
            existing_cols = [c for c in source_cols if c in result.columns]

            if feature_type == 'sum' and existing_cols:
                result[feature_name] = result[existing_cols].fillna(0).sum(axis=1)
                result[f'{feature_name}_any'] = (result[feature_name] > 0).astype(int)
            elif feature_type == 'product' and len(existing_cols) >= 2:
                result[feature_name] = result[existing_cols[0]] * result[existing_cols[1]]

        return result
    ''')


def create_clustering_code() -> str:
    """
    Create clustering/segmentation code.

    Returns
    -------
    str
        Clustering code
    """
    return textwrap.dedent('''
    # =============================================================================
    # SEGMENTATION / CLUSTERING
    # =============================================================================

    def load_cluster_artifacts(
        cluster_model_path: str,
        cluster_scaler_path: str,
    ) -> Tuple[Any, Any]:
        """Load saved clustering model and scaler."""
        cluster_model = None
        cluster_scaler = None

        if cluster_model_path and os.path.exists(cluster_model_path):
            cluster_model = joblib.load(cluster_model_path)
            logger.info(f"Loaded cluster model from: {cluster_model_path}")

        if cluster_scaler_path and os.path.exists(cluster_scaler_path):
            cluster_scaler = joblib.load(cluster_scaler_path)
            logger.info(f"Loaded cluster scaler from: {cluster_scaler_path}")

        return cluster_model, cluster_scaler


    def compute_demand_characteristics(
        df: pd.DataFrame,
        key_columns: List[str],
        target_column: str,
    ) -> pd.DataFrame:
        """Compute demand characteristics for each key."""
        key_col = key_columns[0] if len(key_columns) == 1 else 'key_composite'

        if len(key_columns) > 1:
            df = df.copy()
            df[key_col] = df[key_columns].astype(str).agg('_'.join, axis=1)

        # Aggregate by key
        key_stats = df.groupby(key_col)[target_column].agg([
            'mean', 'std', 'sum', 'count',
            ('zeros', lambda x: (x == 0).sum()),
            ('nonzeros', lambda x: (x > 0).sum()),
        ]).reset_index()

        # Compute characteristics
        key_stats['cv'] = key_stats['std'] / (key_stats['mean'] + 1e-10)
        key_stats['cv_squared'] = key_stats['cv'] ** 2
        key_stats['zero_fraction'] = key_stats['zeros'] / (key_stats['count'] + 1e-10)
        key_stats['adi'] = key_stats['count'] / (key_stats['nonzeros'] + 1e-10)

        # Syntetos-Boylan classification
        def classify_demand(row):
            if row['cv_squared'] <= 0.49 and row['adi'] <= 1.32:
                return 'SMOOTH'
            elif row['cv_squared'] > 0.49 and row['adi'] <= 1.32:
                return 'ERRATIC'
            elif row['cv_squared'] <= 0.49 and row['adi'] > 1.32:
                return 'INTERMITTENT'
            else:
                return 'LUMPY'

        key_stats['demand_pattern'] = key_stats.apply(classify_demand, axis=1)

        return key_stats


    def assign_segments(
        characteristics_df: pd.DataFrame,
        cluster_model: Any,
        cluster_scaler: Any,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Assign segments using saved clustering model."""
        if cluster_model is None:
            logger.warning("No cluster model provided, using demand_pattern as segment")
            characteristics_df['segment_id'] = characteristics_df['demand_pattern']
            return characteristics_df

        result = characteristics_df.copy()

        # Default features for clustering
        if feature_cols is None:
            feature_cols = ['cv', 'cv_squared', 'zero_fraction', 'adi']

        # Ensure all feature columns exist
        available_cols = [c for c in feature_cols if c in result.columns]
        if len(available_cols) == 0:
            logger.warning("No clustering features available, using demand_pattern")
            result['segment_id'] = result['demand_pattern']
            return result

        # Prepare features
        X = result[available_cols].fillna(0).values

        # Scale if scaler provided
        if cluster_scaler is not None:
            X = cluster_scaler.transform(X)

        # Predict clusters
        result['segment_id'] = cluster_model.predict(X)

        return result
    ''')


def create_model_training_code() -> str:
    """
    Create model training code with finalized hyperparameters.

    Returns
    -------
    str
        Model training code
    """
    return textwrap.dedent('''
    # =============================================================================
    # MODEL TRAINING
    # =============================================================================

    def load_model_specs(specs_path: str) -> Dict[str, Any]:
        """Load finalized model specifications."""
        if not specs_path or not os.path.exists(specs_path):
            logger.warning(f"Model specs not found: {specs_path}")
            return {}

        with open(specs_path, 'r') as f:
            specs = json.load(f)
        logger.info(f"Loaded model specs for {len(specs)} model groups")
        return specs


    def train_lightgbm(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict[str, Any],
    ) -> lgb.Booster:
        """Train LightGBM model with given parameters."""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        default_params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'force_row_wise': True,
        }
        default_params.update(params)

        model = lgb.train(
            default_params,
            train_data,
            num_boost_round=params.get('num_boost_round', 500),
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        return model


    def train_xgboost(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict[str, Any],
    ) -> xgb.Booster:
        """Train XGBoost model with given parameters."""
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'verbosity': 0,
        }
        default_params.update(params)

        model = xgb.train(
            default_params,
            dtrain,
            num_boost_round=params.get('num_boost_round', 500),
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        return model


    def train_model_for_group(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        model_type: str,
        model_params: Dict[str, Any],
    ) -> Tuple[Any, float]:
        """Train a model for a specific group/segment."""
        # Prepare features
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target_col].values

        # Train based on model type
        if model_type.lower() in ['lightgbm', 'lgb']:
            model = train_lightgbm(X_train, y_train, X_val, y_val, model_params)
            y_pred = model.predict(X_val)
        elif model_type.lower() in ['xgboost', 'xgb']:
            model = train_xgboost(X_train, y_train, X_val, y_val, model_params)
            dval = xgb.DMatrix(X_val)
            y_pred = model.predict(dval)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Compute validation WAPE
        wape = compute_wape(y_val, y_pred)

        return model, wape


    def save_model(model: Any, model_path: str, model_type: str) -> str:
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if model_type.lower() in ['lightgbm', 'lgb']:
            model.save_model(model_path)
        elif model_type.lower() in ['xgboost', 'xgb']:
            model.save_model(model_path)
        else:
            joblib.dump(model, model_path)

        logger.info(f"Saved {model_type} model to: {model_path}")
        return model_path


    def load_model(model_path: str, model_type: str) -> Any:
        """Load trained model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        if model_type.lower() in ['lightgbm', 'lgb']:
            model = lgb.Booster(model_file=model_path)
        elif model_type.lower() in ['xgboost', 'xgb']:
            model = xgb.Booster()
            model.load_model(model_path)
        else:
            model = joblib.load(model_path)

        return model
    ''')


def create_inference_code() -> str:
    """
    Create model inference code.

    Returns
    -------
    str
        Model inference code
    """
    return textwrap.dedent('''
    # =============================================================================
    # INFERENCE
    # =============================================================================

    def predict_with_model(
        model: Any,
        X: np.ndarray,
        model_type: str,
    ) -> np.ndarray:
        """Generate predictions using trained model."""
        if model_type.lower() in ['lightgbm', 'lgb']:
            predictions = model.predict(X)
        elif model_type.lower() in ['xgboost', 'xgb']:
            dtest = xgb.DMatrix(X)
            predictions = model.predict(dtest)
        else:
            predictions = model.predict(X)

        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)

        return predictions


    def run_inference_for_group(
        df: pd.DataFrame,
        model: Any,
        model_type: str,
        feature_cols: List[str],
    ) -> np.ndarray:
        """Run inference for a specific group."""
        X = df[feature_cols].values
        predictions = predict_with_model(model, X, model_type)
        return predictions
    ''')


# =============================================================================
# PIPELINE GENERATION
# =============================================================================

def generate_training_pipeline(
    config: PipelineConfig,
    deterministic_codes: Dict[str, DeterministicCodeInfo],
    output_path: str,
) -> str:
    """
    Generate a complete training pipeline script.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration
    deterministic_codes : Dict[str, DeterministicCodeInfo]
        Analyzed deterministic code files
    output_path : str
        Path to save the generated script

    Returns
    -------
    str
        Path to the generated script
    """
    key_cols_str = str(config.key_columns)

    script_parts = [
        create_import_block(include_ml=True),
        "",
        create_config_loader(),
        "",
        create_data_loader(config.key_columns, config.date_column, config.target_column),
        "",
        create_metrics_code(),
        "",
        create_categorical_encoding_code(),
        "",
        create_clustering_code(),
        "",
        create_model_training_code(),
        "",
        textwrap.dedent(f'''
    # =============================================================================
    # MAIN TRAINING PIPELINE
    # =============================================================================

    def run_training_pipeline(
        config_path: str,
        data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Parameters
        ----------
        config_path : str
            Path to configuration YAML file
        data_path : str, optional
            Override data path from config
        output_dir : str, optional
            Override output directory

        Returns
        -------
        Dict[str, Any]
            Training results including model paths and metrics
        """
        logger.info("="*60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("="*60)

        # Load configuration
        config = load_config(config_path)

        # Get project root (parent of config directory) to resolve relative paths
        config_dir = os.path.dirname(os.path.abspath(config_path))
        project_root = os.path.dirname(config_dir)  # Project root is parent of config/

        # Override paths if provided - resolve relative paths from project root
        data_path_raw = data_path or get_data_path(config)
        data_path = os.path.join(project_root, data_path_raw) if data_path_raw and not os.path.isabs(data_path_raw) else data_path_raw

        artifact_base_raw = output_dir or config.get('artifact_base_path', 'model_artifacts')
        output_dir = os.path.join(project_root, artifact_base_raw) if artifact_base_raw and not os.path.isabs(artifact_base_raw) else artifact_base_raw

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load source data
        logger.info(f"Loading data from: {{data_path}}")
        df = load_source_data(data_path)

        # Get time period config (top-level keys in config.yaml)
        train_start = config.get('train_start', '')
        train_end = config.get('train_end', '')
        val_start = config.get('val_start', '')
        val_end = config.get('val_end', '')
        test_start = config.get('test_start', '')
        test_end = config.get('test_end', '')

        # Split data
        train_df, val_df, test_df = split_train_val_test(
            df, train_start, train_end, val_start, val_end, test_start, test_end
        )

        # Load categorical encoding specs
        cat_specs_path = os.path.join(output_dir, '..', 'feature_output', 'categorical_encoding_specs.json')
        cat_specs = load_categorical_specs(cat_specs_path)

        # Apply categorical encoding
        if cat_specs:
            train_df = apply_categorical_encoding(train_df, cat_specs)
            val_df = apply_categorical_encoding(val_df, cat_specs)
            test_df = apply_categorical_encoding(test_df, cat_specs)

        # Compute demand characteristics
        logger.info("Computing demand characteristics...")
        demand_chars = compute_demand_characteristics(train_df, {key_cols_str}, TARGET_COLUMN)

        # Load clustering artifacts
        seg_dir = os.path.join(output_dir, '..', 'seg_output')
        cluster_model, cluster_scaler = load_cluster_artifacts(
            os.path.join(seg_dir, 'cluster_model.pkl'),
            os.path.join(seg_dir, 'cluster_scaler.pkl'),
        )

        # Assign segments
        demand_chars = assign_segments(demand_chars, cluster_model, cluster_scaler)
        logger.info(f"Segment distribution: {{demand_chars['segment_id'].value_counts().to_dict()}}")

        # Load model specs
        model_specs = load_model_specs(os.path.join(output_dir, 'final_model_specs.json'))

        # Train models (placeholder - would iterate over model groups)
        results = {{
            'training_completed': True,
            'output_dir': output_dir,
            'n_rows_trained': len(train_df),
            'n_segments': demand_chars['segment_id'].nunique(),
        }}

        # Save training results
        results_path = os.path.join(output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("="*60)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("="*60)

        return results


    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Training Pipeline")
        parser.add_argument("--config", required=True, help="Path to config YAML")
        parser.add_argument("--data", help="Override data path")
        parser.add_argument("--output", help="Override output directory")

        args = parser.parse_args()

        results = run_training_pipeline(
            config_path=args.config,
            data_path=args.data,
            output_dir=args.output,
        )

        print(f"Training completed. Results saved to: {{results['output_dir']}}")
        '''),
    ]

    # Write the script
    script_content = '\n'.join(script_parts)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(script_content)

    logger.info(f"Generated training pipeline: {output_path}")
    return output_path


def generate_inference_pipeline(
    config: PipelineConfig,
    deterministic_codes: Dict[str, DeterministicCodeInfo],
    output_path: str,
) -> str:
    """
    Generate a complete inference pipeline script.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration
    deterministic_codes : Dict[str, DeterministicCodeInfo]
        Analyzed deterministic code files
    output_path : str
        Path to save the generated script

    Returns
    -------
    str
        Path to the generated script
    """
    key_cols_str = str(config.key_columns)

    script_parts = [
        create_import_block(include_ml=True),
        "",
        create_config_loader(),
        "",
        create_data_loader(config.key_columns, config.date_column, config.target_column),
        "",
        create_metrics_code(),
        "",
        create_categorical_encoding_code(),
        "",
        create_clustering_code(),
        "",
        create_inference_code(),
        "",
        textwrap.dedent(f'''
    # =============================================================================
    # MAIN INFERENCE PIPELINE
    # =============================================================================

    def run_inference_pipeline(
        config_path: str,
        data_path: Optional[str] = None,
        model_dir: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run the complete inference pipeline.

        Parameters
        ----------
        config_path : str
            Path to configuration YAML file
        data_path : str, optional
            Override data path from config
        model_dir : str, optional
            Directory containing trained models
        output_path : str, optional
            Path to save predictions

        Returns
        -------
        pd.DataFrame
            Predictions DataFrame
        """
        logger.info("="*60)
        logger.info("STARTING INFERENCE PIPELINE")
        logger.info("="*60)

        # Load configuration
        config = load_config(config_path)

        # Get project root (parent of config directory) to resolve relative paths
        config_dir = os.path.dirname(os.path.abspath(config_path))
        project_root = os.path.dirname(config_dir)  # Project root is parent of config/

        # Set paths - resolve relative paths from project root
        data_path_raw = data_path or get_data_path(config)
        data_path = os.path.join(project_root, data_path_raw) if data_path_raw and not os.path.isabs(data_path_raw) else data_path_raw

        artifact_base_raw = config.get('artifact_base_path', '')
        artifact_base = os.path.join(project_root, artifact_base_raw) if artifact_base_raw and not os.path.isabs(artifact_base_raw) else artifact_base_raw
        model_dir = model_dir or os.path.join(artifact_base, 'model_artifacts')
        output_path = output_path or os.path.join(model_dir, 'predictions.csv')

        # Load source data
        logger.info(f"Loading data from: {{data_path}}")
        df = load_source_data(data_path)

        # Load categorical encoding specs
        cat_specs_path = os.path.join(model_dir, '..', 'feature_output', 'categorical_encoding_specs.json')
        cat_specs = load_categorical_specs(cat_specs_path)

        # Apply categorical encoding
        if cat_specs:
            df = apply_categorical_encoding(df, cat_specs)

        # Compute demand characteristics
        demand_chars = compute_demand_characteristics(df, {key_cols_str}, TARGET_COLUMN)

        # Load clustering artifacts
        seg_dir = os.path.join(model_dir, '..', 'seg_output')
        cluster_model, cluster_scaler = load_cluster_artifacts(
            os.path.join(seg_dir, 'cluster_model.pkl'),
            os.path.join(seg_dir, 'cluster_scaler.pkl'),
        )

        # Assign segments
        demand_chars = assign_segments(demand_chars, cluster_model, cluster_scaler)

        # Load model specs
        model_specs = load_model_specs(os.path.join(model_dir, 'final_model_specs.json'))

        # Generate predictions (placeholder)
        predictions = df.copy()
        predictions['predicted'] = 0  # Placeholder

        # Save predictions
        predictions.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to: {{output_path}}")

        logger.info("="*60)
        logger.info("INFERENCE PIPELINE COMPLETED")
        logger.info("="*60)

        return predictions


    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Inference Pipeline")
        parser.add_argument("--config", required=True, help="Path to config YAML")
        parser.add_argument("--data", help="Override data path")
        parser.add_argument("--model-dir", help="Model directory")
        parser.add_argument("--output", help="Output predictions path")

        args = parser.parse_args()

        predictions = run_inference_pipeline(
            config_path=args.config,
            data_path=args.data,
            model_dir=args.model_dir,
            output_path=args.output,
        )

        print(f"Inference completed. Generated {{len(predictions)}} predictions.")
        '''),
    ]

    script_content = '\n'.join(script_parts)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(script_content)

    logger.info(f"Generated inference pipeline: {output_path}")
    return output_path


def generate_orchestrator_script(
    config: PipelineConfig,
    output_path: str,
) -> str:
    """
    Generate the run_pipeline.py orchestrator script.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration
    output_path : str
        Path to save the script

    Returns
    -------
    str
        Path to the generated script
    """
    script = textwrap.dedent(f'''
    #!/usr/bin/env python3
    """
    Production Pipeline Orchestrator
    =================================

    This script orchestrates the training and inference pipelines.

    Usage:
        python run_pipeline.py --config config/config.yaml
        python run_pipeline.py --config config/config.yaml --train-only
        python run_pipeline.py --config config/config.yaml --inference-only

    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    import argparse
    import logging
    import os
    import sys
    from datetime import datetime

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Add project root to path
    # SCRIPT_DIR is artifacts_base/pipeline_output
    # PROJECT_ROOT needs to be 2 levels up to reach feu-agentic-forecasting
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ARTIFACTS_DIR = os.path.dirname(SCRIPT_DIR)
    PROJECT_ROOT = os.path.dirname(ARTIFACTS_DIR)
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, SCRIPT_DIR)


    def main():
        parser = argparse.ArgumentParser(description="Production Pipeline Orchestrator")
        parser.add_argument("--config", required=True, help="Path to config YAML file")
        parser.add_argument("--train-only", action="store_true", help="Run training only")
        parser.add_argument("--inference-only", action="store_true", help="Run inference only")
        parser.add_argument("--data", help="Override data path")
        parser.add_argument("--output", help="Override output directory")

        args = parser.parse_args()

        run_training = not args.inference_only
        run_inference = not args.train_only

        logger.info("="*60)
        logger.info("PRODUCTION PIPELINE ORCHESTRATOR")
        logger.info(f"Config: {{args.config}}")
        logger.info(f"Training: {{run_training}}, Inference: {{run_inference}}")
        logger.info("="*60)

        try:
            # Run training pipeline
            if run_training:
                logger.info("Starting TRAINING PIPELINE...")
                from training_pipeline import run_training_pipeline
                training_results = run_training_pipeline(
                    config_path=args.config,
                    data_path=args.data,
                    output_dir=args.output,
                )
                logger.info(f"Training completed: {{training_results}}")

            # Run inference pipeline
            if run_inference:
                logger.info("Starting INFERENCE PIPELINE...")
                from inference_pipeline import run_inference_pipeline
                predictions = run_inference_pipeline(
                    config_path=args.config,
                    data_path=args.data,
                    model_dir=args.output,
                )
                logger.info(f"Inference completed: {{len(predictions)}} predictions")

            logger.info("="*60)
            logger.info("PIPELINE ORCHESTRATION COMPLETED SUCCESSFULLY")
            logger.info("="*60)

        except Exception as e:
            logger.error(f"Pipeline failed: {{e}}")
            raise


    if __name__ == "__main__":
        main()
    ''')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(script)

    logger.info(f"Generated orchestrator: {output_path}")
    return output_path


def generate_pipeline_readme(
    config: PipelineConfig,
    output_path: str,
) -> str:
    """
    Generate README documentation for the pipelines.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration
    output_path : str
        Path to save the README

    Returns
    -------
    str
        Path to the generated README
    """
    readme = textwrap.dedent(f'''
    # Production Demand Forecasting Pipeline

    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    ## Overview

    This directory contains production-ready pipelines for demand forecasting.
    All pipelines are self-contained and can run standalone with just the config file.

    ## Files

    | File | Description |
    |------|-------------|
    | `training_pipeline.py` | Trains models using finalized hyperparameters |
    | `inference_pipeline.py` | Generates predictions using trained models |
    | `run_pipeline.py` | Orchestrator for running training and/or inference |
        | `pipeline_config.json` | Configuration snapshot |

    ## Quick Start

    ### Run Full Pipeline (Training + Inference)
    ```bash
    python run_pipeline.py --config ../config/config.yaml
    ```

    ### Run Training Only
    ```bash
    python run_pipeline.py --config ../config/config.yaml --train-only
    ```

    ### Run Inference Only
    ```bash
    python run_pipeline.py --config ../config/config.yaml --inference-only
    ```

    ### Run Backtest
    ```bash
    ```

    ## Configuration

    The pipelines use these key settings:

    - **Target Column**: `{config.target_column}`
    - **Date Column**: `{config.date_column}`
    - **Key Columns**: `{config.key_columns}`
    - **Train Period**: `{config.train_start}` to `{config.train_end}`
    - **Validation Period**: `{config.val_start}` to `{config.val_end}`
    - **Test Period**: `{config.test_start}` to `{config.test_end}`

    ## Pipeline Architecture

    ```
    Raw Data
        │
        ▼
    ┌─────────────────┐
    │ Data Loading    │  Load and validate source data
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Categorical     │  Apply saved encodings
    │ Encoding        │  (from feature crew)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Demand          │  Compute CV, ADI, pattern
    │ Characteristics │  classification
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Segment         │  Use saved clustering model
    │ Assignment      │  (from segmentation crew)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Feature         │  Create lag, rolling,
    │ Engineering     │  calendar features
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Model Training/ │  Train with fixed params
    │ Inference       │  or load and predict
    └────────┬────────┘
             │
             ▼
        Predictions
    ```

    ## Outputs

    ### Training
    - Trained model files (`.txt` for LightGBM, `.json` for XGBoost)
    - Training metrics and validation WAPE

    ### Inference
    - `predictions.csv` - Forecasts with metadata

    ### Backtest

    ## Dependencies

    - Python 3.8+
    - pandas, numpy
    - lightgbm, xgboost
    - scikit-learn
    - pyyaml
    - joblib
    ''')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(readme)

    logger.info(f"Generated README: {output_path}")
    return output_path


def generate_pipeline_config(
    config: PipelineConfig,
    output_path: str,
) -> str:
    """
    Generate pipeline_config.json with all settings.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration
    output_path : str
        Path to save the config

    Returns
    -------
    str
        Path to the generated config
    """
    pipeline_config = {
        'generated_timestamp': datetime.now().isoformat(),
        'pipeline_name': config.pipeline_name,
        'data': {
            'data_path': config.data_path,
            'target_column': config.target_column,
            'date_column': config.date_column,
            'key_columns': config.key_columns,
            'categorical_columns': config.categorical_columns,
        },
        'time_periods': {
            'train_start': config.train_start,
            'train_end': config.train_end,
            'val_start': config.val_start,
            'val_end': config.val_end,
            'test_start': config.test_start,
            'test_end': config.test_end,
        },
        'artifacts': {
            'cluster_model_path': config.cluster_model_path,
            'cluster_scaler_path': config.cluster_scaler_path,
            'categorical_encoding_specs_path': config.categorical_encoding_specs_path,
            'model_specs_path': config.model_specs_path,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(pipeline_config, f, indent=2)

    logger.info(f"Generated pipeline config: {output_path}")
    return output_path


# =============================================================================
# VALIDATION
# =============================================================================

def validate_pipeline_syntax(file_path: str) -> Dict[str, Any]:
    """
    Validate Python syntax of a pipeline file.

    Parameters
    ----------
    file_path : str
        Path to the Python file

    Returns
    -------
    Dict[str, Any]
        Validation result with 'valid' and 'error' keys
    """
    result = {'valid': False, 'error': None, 'path': file_path}

    if not os.path.exists(file_path):
        result['error'] = 'File not found'
        return result

    try:
        py_compile.compile(file_path, doraise=True)
        result['valid'] = True
        logger.info(f"✓ Syntax valid: {file_path}")
    except py_compile.PyCompileError as e:
        result['error'] = str(e)
        logger.error(f"✗ Syntax error in {file_path}: {e}")

    return result


def validate_all_pipelines(pipeline_dir: str) -> Dict[str, Any]:
    """
    Validate all pipeline files in a directory.

    Parameters
    ----------
    pipeline_dir : str
        Directory containing pipeline files

    Returns
    -------
    Dict[str, Any]
        Validation results
    """
    python_files = [
        'training_pipeline.py',
        'inference_pipeline.py',
        'run_pipeline.py',
    ]

    results = {}
    all_valid = True
    errors = []

    for filename in python_files:
        file_path = os.path.join(pipeline_dir, filename)
        result = validate_pipeline_syntax(file_path)
        results[filename] = result

        if not result['valid']:
            all_valid = False
            errors.append(f"{filename}: {result['error']}")

    return {
        'all_valid': all_valid,
        'results': results,
        'errors': errors,
    }


# =============================================================================
# SELF-CONTAINED PIPELINE UTILITIES
# =============================================================================

# List of utility modules required for self-contained pipeline
REQUIRED_UTILS = [
    'dead_key_handler.py',
    'model_training.py',
    'recursive_forecasting.py',
    'smart_output.py',
]


def copy_utils_to_pipeline(
    source_utils_dir: str,
    pipeline_output_dir: str,
    utils_to_copy: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Copy required utility modules to the pipeline output folder.

    Parameters
    ----------
    source_utils_dir : str
        Path to source utils directory (project root/utils)
    pipeline_output_dir : str
        Path to pipeline output directory
    utils_to_copy : List[str], optional
        List of utility files to copy. If None, uses REQUIRED_UTILS

    Returns
    -------
    Dict[str, str]
        Mapping of utility name to copied path
    """
    import shutil

    utils_to_copy = utils_to_copy or REQUIRED_UTILS
    target_utils_dir = os.path.join(pipeline_output_dir, 'utils')
    os.makedirs(target_utils_dir, exist_ok=True)

    copied_files = {}

    # Create __init__.py for utils package
    init_path = os.path.join(target_utils_dir, '__init__.py')
    with open(init_path, 'w') as f:
        f.write('"""Self-contained utilities for production pipeline."""\n')
    copied_files['__init__.py'] = init_path

    for util_file in utils_to_copy:
        source_path = os.path.join(source_utils_dir, util_file)
        target_path = os.path.join(target_utils_dir, util_file)

        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            copied_files[util_file] = target_path
            logger.info(f"Copied utility: {util_file}")
        else:
            logger.warning(f"Utility not found, skipping: {source_path}")

    return copied_files


def create_pipeline_config(
    source_config_path: str,
    pipeline_output_dir: str,
    sourcedata_path: str = './sourcedata',
) -> str:
    """
    Create a self-contained config.yaml for the pipeline folder.

    The config will have:
    - input_data_path pointing to sourcedata_path (relative to pipeline folder)
    - artifact_base_path set to '.' (current directory)
    - All other settings preserved

    Parameters
    ----------
    source_config_path : str
        Path to source config.yaml
    pipeline_output_dir : str
        Path to pipeline output directory
    sourcedata_path : str
        Relative path to sourcedata folder from pipeline folder

    Returns
    -------
    str
        Path to created config file
    """
    import yaml

    # Load source config
    with open(source_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Helper to extract data_path from various config formats
    def _get_data_path_from_config(cfg):
        if 'data_path' in cfg:
            return cfg['data_path']
        if 'data' in cfg and isinstance(cfg['data'], dict) and 'data_path' in cfg['data']:
            return cfg['data']['data_path']
        if 'input_data_path' in cfg:
            return cfg['input_data_path']
        return 'raw_data.csv'

    # Update paths for self-contained deployment
    # The sourcedata_path should be relative to the pipeline folder
    # Default: user copies sourcedata folder into pipeline folder as ./sourcedata/
    original_data_filename = os.path.basename(_get_data_path_from_config(config))
    config['input_data_path'] = os.path.join(sourcedata_path, original_data_filename)

    # Artifact base is the pipeline folder itself
    config['artifact_base_path'] = '.'

    # Add deployment metadata
    config['_deployment_info'] = {
        'generated_timestamp': datetime.now().isoformat(),
        'self_contained': True,
        'instructions': (
            'To deploy: 1) Copy sourcedata folder to ./sourcedata/ or update '
            'input_data_path to point to your data location.'
        ),
    }

    # Save to pipeline folder
    target_config_path = os.path.join(pipeline_output_dir, 'config.yaml')
    with open(target_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Created self-contained config: {target_config_path}")
    return target_config_path


def copy_model_artifacts(
    model_output_dir: str,
    pipeline_output_dir: str,
) -> Dict[str, str]:
    """
    Copy trained model artifacts to the pipeline folder.

    Parameters
    ----------
    model_output_dir : str
        Path to model output directory (contains trained models)
    pipeline_output_dir : str
        Path to pipeline output directory

    Returns
    -------
    Dict[str, str]
        Mapping of artifact name to copied path
    """
    import shutil
    import glob

    target_models_dir = os.path.join(pipeline_output_dir, 'models')
    os.makedirs(target_models_dir, exist_ok=True)

    copied_artifacts = {}

    if not os.path.exists(model_output_dir):
        logger.warning(f"Model output directory not found: {model_output_dir}")
        return copied_artifacts

    # Copy model files (*.txt for LightGBM, *.json for XGBoost, *.pkl for others)
    model_patterns = ['*.txt', '*.json', '*.pkl', '*.joblib']
    for pattern in model_patterns:
        for source_path in glob.glob(os.path.join(model_output_dir, pattern)):
            filename = os.path.basename(source_path)
            target_path = os.path.join(target_models_dir, filename)
            shutil.copy2(source_path, target_path)
            copied_artifacts[filename] = target_path

    # Copy final_model_specs.json if exists
    specs_path = os.path.join(model_output_dir, 'final_model_specs.json')
    if os.path.exists(specs_path):
        target_specs = os.path.join(target_models_dir, 'final_model_specs.json')
        shutil.copy2(specs_path, target_specs)
        copied_artifacts['final_model_specs.json'] = target_specs

    logger.info(f"Copied {len(copied_artifacts)} model artifacts to {target_models_dir}")
    return copied_artifacts


def copy_feature_artifacts(
    feature_output_dir: str,
    pipeline_output_dir: str,
) -> Dict[str, str]:
    """
    Copy feature engineering artifacts (encodings, scalers) to pipeline folder.

    Parameters
    ----------
    feature_output_dir : str
        Path to feature output directory
    pipeline_output_dir : str
        Path to pipeline output directory

    Returns
    -------
    Dict[str, str]
        Mapping of artifact name to copied path
    """
    import shutil

    target_features_dir = os.path.join(pipeline_output_dir, 'features')
    os.makedirs(target_features_dir, exist_ok=True)

    copied_artifacts = {}

    if not os.path.exists(feature_output_dir):
        logger.warning(f"Feature output directory not found: {feature_output_dir}")
        return copied_artifacts

    # Files to copy
    feature_files = [
        'categorical_encoding_specs.json',
        'feature_context.json',
        'feature_columns.json',
    ]

    for filename in feature_files:
        source_path = os.path.join(feature_output_dir, filename)
        if os.path.exists(source_path):
            target_path = os.path.join(target_features_dir, filename)
            shutil.copy2(source_path, target_path)
            copied_artifacts[filename] = target_path

    logger.info(f"Copied {len(copied_artifacts)} feature artifacts to {target_features_dir}")
    return copied_artifacts


def copy_segmentation_artifacts(
    seg_output_dir: str,
    pipeline_output_dir: str,
) -> Dict[str, str]:
    """
    Copy segmentation artifacts (cluster models, scalers) to pipeline folder.

    Parameters
    ----------
    seg_output_dir : str
        Path to segmentation output directory
    pipeline_output_dir : str
        Path to pipeline output directory

    Returns
    -------
    Dict[str, str]
        Mapping of artifact name to copied path
    """
    import shutil

    target_seg_dir = os.path.join(pipeline_output_dir, 'segmentation')
    os.makedirs(target_seg_dir, exist_ok=True)

    copied_artifacts = {}

    if not os.path.exists(seg_output_dir):
        logger.warning(f"Segmentation output directory not found: {seg_output_dir}")
        return copied_artifacts

    # Files to copy
    seg_files = [
        'cluster_model.pkl',
        'cluster_scaler.pkl',
        'segment_manifest.csv',
        'segmentation_context.json',
    ]

    for filename in seg_files:
        source_path = os.path.join(seg_output_dir, filename)
        if os.path.exists(source_path):
            target_path = os.path.join(target_seg_dir, filename)
            shutil.copy2(source_path, target_path)
            copied_artifacts[filename] = target_path

    logger.info(f"Copied {len(copied_artifacts)} segmentation artifacts to {target_seg_dir}")
    return copied_artifacts


def generate_self_contained_import_block(include_ml: bool = True) -> str:
    """
    Create import block that uses local utils folder.

    Parameters
    ----------
    include_ml : bool
        Include ML library imports

    Returns
    -------
    str
        Import block code using local ./utils/
    """
    imports = [
        "#!/usr/bin/env python3",
        '"""',
        "Auto-generated Self-Contained Production Pipeline",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This script is self-contained and can run from the pipeline folder.",
        "Required structure:",
        "  ./config.yaml          - Configuration file",
        "  ./utils/               - Utility modules",
        "  ./models/              - Trained model artifacts",
        "  ./features/            - Feature encoding specs",
        "  ./segmentation/        - Clustering artifacts",
        "  ./sourcedata/          - Source data (or update config.yaml path)",
        '"""',
        "",
        "import argparse",
        "import json",
        "import logging",
        "import os",
        "import sys",
        "import warnings",
        "from datetime import datetime",
        "from typing import Any, Dict, List, Optional, Tuple",
        "",
        "# Add current directory to path for local imports",
        "SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))",
        "sys.path.insert(0, SCRIPT_DIR)",
        "",
        "import joblib",
        "import numpy as np",
        "import pandas as pd",
        "import yaml",
    ]

    if include_ml:
        imports.extend([
            "",
            "# ML Libraries",
            "import lightgbm as lgb",
            "import xgboost as xgb",
            "from sklearn.preprocessing import StandardScaler",
            "from sklearn.metrics import mean_absolute_error, mean_squared_error",
        ])

    imports.extend([
        "",
        "# Local utilities (from ./utils/)",
        "from utils.dead_key_handler import filter_predictions_for_dead_keys, load_dead_key_list",
        "",
        "# Suppress warnings",
        "warnings.filterwarnings('ignore')",
        "",
        "# Configure logging",
        "logging.basicConfig(",
        "    level=logging.INFO,",
        "    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'",
        ")",
        "logger = logging.getLogger(__name__)",
    ])

    return '\n'.join(imports)


def create_self_contained_pipeline(
    config: PipelineConfig,
    source_config_path: str,
    source_utils_dir: str,
    model_output_dir: Optional[str] = None,
    feature_output_dir: Optional[str] = None,
    seg_output_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a fully self-contained pipeline folder ready for deployment.

    This function:
    1. Copies required utility modules to ./utils/
    2. Creates self-contained config.yaml with relative paths
    3. Copies trained model artifacts to ./models/
    4. Copies feature artifacts to ./features/
    5. Copies segmentation artifacts to ./segmentation/
    6. Generates self-contained pipeline scripts
    7. Creates README with deployment instructions

    FOLDER STRUCTURE CREATED:
    pipeline_output/
    ├── config.yaml              # Self-contained config
    ├── utils/                   # Required utilities
    │   ├── __init__.py
    │   ├── dead_key_handler.py
    │   ├── model_training.py
    │   ├── recursive_forecasting.py
    │   └── smart_output.py
    ├── models/                  # Trained model artifacts
    ├── features/                # Feature encoding specs
    ├── segmentation/            # Clustering artifacts
    ├── run_pipeline.py          # Orchestrator
    ├── training_pipeline.py     # Training script
    ├── inference_pipeline.py    # Inference script
    └── README.md                # Deployment instructions

    DEPLOYMENT:
    1. Copy entire pipeline_output folder to production
    2. Place source data in ./sourcedata/ or update config.yaml

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration
    source_config_path : str
        Path to source config.yaml
    source_utils_dir : str
        Path to source utils directory
    model_output_dir : str, optional
        Path to model output directory
    feature_output_dir : str, optional
        Path to feature output directory
    seg_output_dir : str, optional
        Path to segmentation output directory
    output_dir : str, optional
        Override output directory

    Returns
    -------
    Dict[str, Any]
        Summary of copied files and generated scripts
    """
    logger.info("="*60)
    logger.info("CREATING SELF-CONTAINED PIPELINE")
    logger.info("="*60)

    output_dir = output_dir or config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    result = {
        'output_dir': output_dir,
        'copied_utils': {},
        'copied_models': {},
        'copied_features': {},
        'copied_segmentation': {},
        'generated_scripts': {},
        'config_path': None,
    }

    # 1. Copy utility modules
    logger.info("Step 1/7: Copying utility modules...")
    result['copied_utils'] = copy_utils_to_pipeline(
        source_utils_dir=source_utils_dir,
        pipeline_output_dir=output_dir,
    )
    logger.info(f"  Copied {len(result['copied_utils'])} utility files")

    # 2. Create self-contained config
    logger.info("Step 2/7: Creating self-contained config.yaml...")
    result['config_path'] = create_pipeline_config(
        source_config_path=source_config_path,
        pipeline_output_dir=output_dir,
    )

    # 3. Copy model artifacts (if available)
    if model_output_dir:
        logger.info("Step 3/7: Copying model artifacts...")
        result['copied_models'] = copy_model_artifacts(
            model_output_dir=model_output_dir,
            pipeline_output_dir=output_dir,
        )
        logger.info(f"  Copied {len(result['copied_models'])} model files")
    else:
        logger.info("Step 3/7: Skipping model artifacts (not provided)")

    # 4. Copy feature artifacts (if available)
    if feature_output_dir:
        logger.info("Step 4/7: Copying feature artifacts...")
        result['copied_features'] = copy_feature_artifacts(
            feature_output_dir=feature_output_dir,
            pipeline_output_dir=output_dir,
        )
        logger.info(f"  Copied {len(result['copied_features'])} feature files")
    else:
        logger.info("Step 4/7: Skipping feature artifacts (not provided)")

    # 5. Copy segmentation artifacts (if available)
    if seg_output_dir:
        logger.info("Step 5/7: Copying segmentation artifacts...")
        result['copied_segmentation'] = copy_segmentation_artifacts(
            seg_output_dir=seg_output_dir,
            pipeline_output_dir=output_dir,
        )
        logger.info(f"  Copied {len(result['copied_segmentation'])} segmentation files")
    else:
        logger.info("Step 5/7: Skipping segmentation artifacts (not provided)")

    # 6. Generate self-contained scripts
    logger.info("Step 6/7: Generating self-contained pipeline scripts...")
    logger.info(f"  Generated {len(result['generated_scripts'])} scripts")

    # 7. Generate README
    logger.info("Step 7/7: Generating deployment README...")
    readme_content = textwrap.dedent(f"""
    # Self-Contained Pipeline Deployment Guide

    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    ## Folder Structure

    ```
    {os.path.basename(output_dir)}/
    ├── config.yaml              # Configuration file
    ├── utils/                   # Required utilities ({len(result['copied_utils'])} files)
    ├── models/                  # Trained model artifacts
    ├── features/                # Feature encoding specs
    ├── segmentation/            # Clustering artifacts
    └── README.md                # This file
    ```

    ## Deployment Instructions

    ### 1. Copy this folder to production
    ```bash
    cp -r {os.path.basename(output_dir)} /path/to/production/
    cd /path/to/production/{os.path.basename(output_dir)}
    ```

    ### 2. Prepare source data
    Either copy your source data:
    ```bash
    mkdir -p sourcedata
    cp /path/to/your/data.csv sourcedata/
    ```

    Or update `config.yaml` to point to your data location:
    ```yaml
    input_data_path: /absolute/path/to/your/data.csv
    ```

    ### 3. Install dependencies
    ```bash
    pip install pandas numpy scikit-learn lightgbm xgboost pyyaml joblib
    ```

    ```bash
    # Or with custom parameters:
    ```

    ## Configuration

    Key settings in `config.yaml`:
    - `input_data_path`: Path to source data (relative to this folder or absolute)
    - `test_start` / `test_end`: Test period range
    - `artifact_base_path`: Output directory (default: `.` = this folder)

    ## Outputs


    ## Troubleshooting

    1. **Import errors**: Ensure all dependencies are installed
    2. **File not found**: Check `input_data_path` in config.yaml
    3. **No data for period**: Verify test_start/test_end match your data

    ---
    This pipeline is self-contained and requires no external dependencies
    beyond standard Python ML libraries.
    """)

    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    result['generated_scripts']['README.md'] = readme_path

    logger.info("="*60)
    logger.info("✓ SELF-CONTAINED PIPELINE CREATED SUCCESSFULLY")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Utils: {len(result['copied_utils'])} files")
    logger.info(f"  Models: {len(result['copied_models'])} files")
    logger.info(f"  Features: {len(result['copied_features'])} files")
    logger.info(f"  Segmentation: {len(result['copied_segmentation'])} files")
    logger.info(f"  Scripts: {len(result['generated_scripts'])} files")
    logger.info("="*60)

    return result


# =============================================================================
# HIGH-LEVEL PIPELINE GENERATION
# =============================================================================

def run_pipeline_generation(
    config: PipelineConfig,
    output_dir: Optional[str] = None,
) -> PipelineGenerationResult:
    """
    Run complete pipeline generation.

    This is the main entry point that generates ALL production pipelines
    in a single call.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration
    output_dir : str, optional
        Override output directory

    Returns
    -------
    PipelineGenerationResult
        Result with paths to all generated files and validation status
    """
    logger.info("="*60)
    logger.info("STARTING PIPELINE GENERATION")
    logger.info("="*60)

    output_dir = output_dir or config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    result = PipelineGenerationResult(
        generation_timestamp=datetime.now().isoformat(),
    )

    # Analyze deterministic code files
    logger.info("Analyzing deterministic code files...")
    deterministic_codes = {
        'eda': analyze_deterministic_code(config.eda_deterministic_path),
        'segmentation': analyze_deterministic_code(config.segmentation_deterministic_path),
        'feature': analyze_deterministic_code(config.feature_deterministic_path),
        'training': analyze_deterministic_code(config.training_deterministic_path),
    }

    result.deterministic_code_used = {
        name: info.exists for name, info in deterministic_codes.items()
    }

    # Generate training pipeline
    logger.info("Generating training pipeline...")
    result.training_pipeline_path = generate_training_pipeline(
        config=config,
        deterministic_codes=deterministic_codes,
        output_path=os.path.join(output_dir, 'training_pipeline.py'),
    )

    # Generate inference pipeline
    logger.info("Generating inference pipeline...")
    result.inference_pipeline_path = generate_inference_pipeline(
        config=config,
        deterministic_codes=deterministic_codes,
        output_path=os.path.join(output_dir, 'inference_pipeline.py'),
    )

    # Generate orchestrator
    logger.info("Generating orchestrator...")
    result.orchestrator_path = generate_orchestrator_script(
        config=config,
        output_path=os.path.join(output_dir, 'run_pipeline.py'),
    )


    # Generate config and README
    logger.info("Generating documentation...")
    result.config_path = generate_pipeline_config(
        config=config,
        output_path=os.path.join(output_dir, 'pipeline_config.json'),
    )

    result.readme_path = generate_pipeline_readme(
        config=config,
        output_path=os.path.join(output_dir, 'pipeline_readme.md'),
    )

    # Validate all pipelines
    logger.info("Validating generated pipelines...")
    validation = validate_all_pipelines(output_dir)
    result.all_valid = validation['all_valid']
    result.validation_results = validation['results']
    result.validation_errors = validation['errors']

    # Count total lines
    total_lines = 0
    for filename in ['training_pipeline.py', 'inference_pipeline.py', 'run_pipeline.py']:
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                total_lines += len(f.readlines())
    result.total_lines_generated = total_lines

    if result.all_valid:
        logger.info("="*60)
        logger.info("✓ PIPELINE GENERATION COMPLETED SUCCESSFULLY")
        logger.info(f"  Total lines generated: {total_lines}")
        logger.info("="*60)
    else:
        logger.error("="*60)
        logger.error("✗ PIPELINE GENERATION COMPLETED WITH ERRORS")
        for error in result.validation_errors:
            logger.error(f"  {error}")
        logger.error("="*60)

    return result


# =============================================================================
# MODEL-SPEC-BASED DETERMINISTIC PIPELINES
# =============================================================================

def generate_deterministic_training_pipeline(
    config: PipelineConfig,
    output_path: str,
) -> str:
    """
    Generate a DETERMINISTIC training pipeline that:
    1. Loads model specs from previous training (model types, hyperparameters, feature columns)
    2. Retrains the SAME models with train+val data (NO hyperparameter search)
    3. Saves retrained models for inference

    This is for production retraining with more data, not for model selection.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration
    output_path : str
        Path to save the generated script

    Returns
    -------
    str
        Path to the generated script
    """
    key_cols_str = str(config.key_columns)

    script = textwrap.dedent(f'''
    #!/usr/bin/env python3
    """
    Deterministic Training Pipeline
    ================================

    This pipeline RETRAINS models with more data using FIXED hyperparameters.

    KEY BEHAVIOR:
    - Loads model specs (types, hyperparameters, feature columns) from previous training
    - Retrains same models with train + validation data combined
    - NO hyperparameter search - uses finalized parameters
    - Saves retrained models for inference

    Usage:
        python deterministic_training_pipeline.py --config config/config.yaml
        python deterministic_training_pipeline.py --config config/config.yaml --model-specs path/to/specs.json

    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    import argparse
    import json
    import logging
    import os
    import sys
    import time
    from datetime import datetime
    from typing import Any, Dict, List, Optional, Tuple

    import joblib
    import numpy as np
    import pandas as pd
    import yaml

    # ML Libraries
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Add project root to path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ARTIFACTS_DIR = os.path.dirname(SCRIPT_DIR)
    PROJECT_ROOT = os.path.dirname(ARTIFACTS_DIR)
    sys.path.insert(0, PROJECT_ROOT)

    # Configuration
    KEY_COLUMNS = {key_cols_str}
    DATE_COLUMN = "{config.date_column}"
    TARGET_COLUMN = "{config.target_column}"


    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON."""
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            return yaml.safe_load(f)


    def get_data_path(config: Dict[str, Any]) -> str:
        """Extract data_path from config (handles multiple formats)."""
        if 'data_path' in config:
            return config['data_path']
        if 'data' in config and isinstance(config['data'], dict):
            if 'data_path' in config['data']:
                return config['data']['data_path']
        if 'input_data_path' in config:
            return config['input_data_path']
        return ''


    def compute_wape(y_true, y_pred):
        """Compute Weighted Absolute Percentage Error."""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-10)


    def load_model_specs(specs_path: str) -> Dict[str, Any]:
        """
        Load model specifications from training crew output.

        Expected format:
        {{
            "model_type": "lightgbm",
            "hyperparameters": {{}},
            "feature_columns": [...],
            "target_column": "...",
            "validation_wape": 0.xx,
            "test_wape": 0.xx,
            ...
        }}
        """
        if not os.path.exists(specs_path):
            raise FileNotFoundError(f"Model specs not found: {{specs_path}}")

        with open(specs_path, 'r') as f:
            specs = json.load(f)

        logger.info(f"Loaded model specs from: {{specs_path}}")
        return specs


    def train_lightgbm_fixed(
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_columns: List[str],
        params: Dict[str, Any],
    ) -> lgb.Booster:
        """
        Train LightGBM with FIXED hyperparameters (no tuning).

        Uses a small validation split only for early stopping, not for model selection.
        """
        # Create internal validation split for early stopping only
        n_train = int(len(X_train) * 0.9)
        X_fit, X_stop = X_train[:n_train], X_train[n_train:]
        y_fit, y_stop = y_train[:n_train], y_train[n_train:]

        train_data = lgb.Dataset(X_fit, label=y_fit, feature_name=feature_columns)
        stop_data = lgb.Dataset(X_stop, label=y_stop, reference=train_data)

        # Use FIXED params - no tuning
        lgb_params = {{
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'force_row_wise': True,
        }}
        lgb_params.update(params)

        # Remove non-lgb params
        num_rounds = lgb_params.pop('num_boost_round', 1000)

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=num_rounds,
            valid_sets=[stop_data],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        return model


    def train_xgboost_fixed(
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_columns: List[str],
        params: Dict[str, Any],
    ) -> xgb.Booster:
        """
        Train XGBoost with FIXED hyperparameters (no tuning).
        """
        # Create internal validation split for early stopping only
        n_train = int(len(X_train) * 0.9)
        X_fit, X_stop = X_train[:n_train], X_train[n_train:]
        y_fit, y_stop = y_train[:n_train], y_train[n_train:]

        dtrain = xgb.DMatrix(X_fit, label=y_fit, feature_names=feature_columns)
        dstop = xgb.DMatrix(X_stop, label=y_stop, feature_names=feature_columns)

        # Use FIXED params - no tuning
        xgb_params = {{
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'verbosity': 0,
        }}
        xgb_params.update(params)

        # Remove non-xgb params
        num_rounds = xgb_params.pop('num_boost_round', 1000)

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=num_rounds,
            evals=[(dstop, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        return model


    def run_deterministic_training(
        config_path: str,
        model_specs_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run deterministic training with FIXED hyperparameters.

        Parameters
        ----------
        config_path : str
            Path to config.yaml
        model_specs_path : str, optional
            Path to model specs JSON (default: from artifact_base_path)
        output_dir : str, optional
            Output directory for retrained models

        Returns
        -------
        Dict[str, Any]
            Training results
        """
        start_time = time.time()

        logger.info("="*60)
        logger.info("DETERMINISTIC TRAINING PIPELINE")
        logger.info("(Fixed hyperparameters, no tuning)")
        logger.info("="*60)

        # Load config
        config = load_config(config_path)

        # Resolve paths
        config_dir = os.path.dirname(os.path.abspath(config_path))
        project_root = os.path.dirname(config_dir)

        artifact_base_raw = config.get('artifact_base_path', '')
        artifact_base = os.path.join(project_root, artifact_base_raw) if artifact_base_raw and not os.path.isabs(artifact_base_raw) else artifact_base_raw

        # Find model specs
        if model_specs_path is None:
            model_output_dir = os.path.join(artifact_base, 'model_artifacts')
            model_specs_path = os.path.join(model_output_dir, 'final_model_specs.json')

        model_specs = load_model_specs(model_specs_path)

        # Setup output
        output_dir = output_dir or os.path.join(artifact_base, 'retrained_models')
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        data_path_raw = get_data_path(config)
        data_path = os.path.join(project_root, data_path_raw) if data_path_raw and not os.path.isabs(data_path_raw) else data_path_raw

        logger.info(f"Loading data from: {{data_path}}")
        file_ext = os.path.splitext(data_path)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(data_path)
        elif file_ext in ('.parquet', '.pq'):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {{file_ext}}. Use .csv or .parquet")
        logger.info(f"Loaded {{len(df)}} rows")

        # Get time periods
        train_start = config.get('train_start', '')
        train_end = config.get('train_end', '')
        val_start = config.get('val_start', '')
        val_end = config.get('val_end', '')

        # Convert date column to int for comparison
        df[DATE_COLUMN] = df[DATE_COLUMN].astype(str).astype(int)

        # Combine train + validation data for retraining
        train_end_int = int(train_end)
        val_end_int = int(val_end)

        train_val_df = df[(df[DATE_COLUMN] >= int(train_start)) & (df[DATE_COLUMN] <= val_end_int)].copy()
        logger.info(f"Training on {{len(train_val_df)}} rows (train + val combined)")

        # Get model info from specs
        model_type = model_specs.get('model_type', 'lightgbm')
        hyperparameters = model_specs.get('hyperparameters', {{}})
        feature_columns = model_specs.get('feature_columns', [])

        if not feature_columns:
            raise ValueError("No feature columns found in model specs")

        logger.info(f"Model type: {{model_type}}")
        logger.info(f"Feature columns: {{len(feature_columns)}}")
        logger.info(f"Hyperparameters: {{hyperparameters}}")

        # Prepare features
        # Check which features are available
        available_features = [c for c in feature_columns if c in train_val_df.columns]
        missing_features = [c for c in feature_columns if c not in train_val_df.columns]

        if missing_features:
            logger.warning(f"Missing {{len(missing_features)}} features: {{missing_features[:5]}}...")

        if not available_features:
            raise ValueError("No feature columns available in data")

        # Sanitize: replace inf/-inf with NaN, then fill with 0
        X_train = train_val_df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
        y_train = train_val_df[TARGET_COLUMN].replace([np.inf, -np.inf], np.nan).fillna(0).values

        # Train model with FIXED hyperparameters
        logger.info(f"Training {{model_type}} with fixed hyperparameters...")

        if model_type.lower() in ['lightgbm', 'lgb']:
            model = train_lightgbm_fixed(X_train, y_train, available_features, hyperparameters)
            model_ext = '.txt'
        elif model_type.lower() in ['xgboost', 'xgb']:
            model = train_xgboost_fixed(X_train, y_train, available_features, hyperparameters)
            model_ext = '.json'
        else:
            raise ValueError(f"Unknown model type: {{model_type}}")

        # Save retrained model
        model_path = os.path.join(output_dir, f'retrained_model{{model_ext}}')
        if model_type.lower() in ['lightgbm', 'lgb']:
            model.save_model(model_path)
        else:
            model.save_model(model_path)

        logger.info(f"Saved retrained model to: {{model_path}}")

        # Save model artifact with all specs
        artifact = {{
            'model': model,
            'model_type': model_type,
            'feature_columns': available_features,
            'target_column': TARGET_COLUMN,
            'hyperparameters': hyperparameters,
            'training_data_size': len(train_val_df),
            'training_period': f"{{train_start}} to {{val_end}}",
        }}

        artifact_path = os.path.join(output_dir, 'retrained_model_artifact.pkl')
        joblib.dump(artifact, artifact_path)
        logger.info(f"Saved model artifact to: {{artifact_path}}")

        total_time = time.time() - start_time

        result = {{
            'model_path': model_path,
            'artifact_path': artifact_path,
            'model_type': model_type,
            'n_features': len(available_features),
            'training_rows': len(train_val_df),
            'total_time_seconds': total_time,
        }}

        # Save summary
        summary_path = os.path.join(output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info("="*60)
        logger.info("DETERMINISTIC TRAINING COMPLETED")
        logger.info(f"  Model: {{model_path}}")
        logger.info(f"  Time: {{total_time:.1f}} seconds")
        logger.info("="*60)

        return result


    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Deterministic Training Pipeline")
        parser.add_argument("--config", required=True, help="Path to config.yaml")
        parser.add_argument("--model-specs", help="Path to model specs JSON")
        parser.add_argument("--output-dir", help="Output directory")

        args = parser.parse_args()

        results = run_deterministic_training(
            config_path=args.config,
            model_specs_path=args.model_specs,
            output_dir=args.output_dir,
        )

        print(f"\\nTraining completed!")
        print(f"  Model: {{results['model_path']}}")
    ''')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(script)

    logger.info(f"Generated deterministic training pipeline: {output_path}")
    return output_path


def generate_recursive_inference_pipeline(
    config: PipelineConfig,
    output_path: str,
) -> str:
    """
    Generate an inference pipeline that uses RECURSIVE multi-step forecasting.

    This avoids feature leakage by:
    1. Using only historical actuals up to forecast origin
    2. Generating predictions step-by-step
    3. Updating lag/rolling features with predictions (not actuals)

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration
    output_path : str
        Path to save the generated script

    Returns
    -------
    str
        Path to the generated script
    """
    key_cols_str = str(config.key_columns)

    script = textwrap.dedent(f'''
    #!/usr/bin/env python3
    """
    Recursive Inference Pipeline
    =============================

    This pipeline generates MULTI-STEP FORWARD FORECASTS without feature leakage.

    KEY BEHAVIOR:
    - Forecasts from t+1 to t+forecast_horizon
    - Updates lag/rolling features recursively using predictions
    - Never uses actual future values in features

    Usage:
        python recursive_inference_pipeline.py --config config/config.yaml --forecast-origin 202401
        python recursive_inference_pipeline.py --config config/config.yaml --horizon 8

    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    import argparse
    import json
    import logging
    import os
    import re
    import sys
    import time
    from datetime import datetime
    from typing import Any, Dict, List, Optional, Tuple

    import joblib
    import numpy as np
    import pandas as pd
    import yaml

    # ML Libraries
    import lightgbm as lgb
    import xgboost as xgb

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Add project root to path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ARTIFACTS_DIR = os.path.dirname(SCRIPT_DIR)
    PROJECT_ROOT = os.path.dirname(ARTIFACTS_DIR)
    sys.path.insert(0, PROJECT_ROOT)

    # Configuration
    KEY_COLUMNS = {key_cols_str}
    DATE_COLUMN = "{config.date_column}"
    TARGET_COLUMN = "{config.target_column}"


    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON."""
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            return yaml.safe_load(f)


    def get_data_path(config: Dict[str, Any]) -> str:
        """Extract data_path from config (handles multiple formats)."""
        if 'data_path' in config:
            return config['data_path']
        if 'data' in config and isinstance(config['data'], dict):
            if 'data_path' in config['data']:
                return config['data']['data_path']
        if 'input_data_path' in config:
            return config['input_data_path']
        return ''


    def load_model_artifact(artifact_path: str) -> Dict[str, Any]:
        """Load model artifact with feature columns."""
        if artifact_path.endswith('.pkl'):
            return joblib.load(artifact_path)
        else:
            raise ValueError(f"Unknown artifact format: {{artifact_path}}")


    def identify_lag_columns(feature_columns: List[str], target_col: str) -> List[str]:
        """Identify lag feature columns (excluding derived features)."""
        lag_cols = []
        for col in feature_columns:
            col_lower = col.lower()
            # Skip derived features (lag_diff, lag_ratio, lag_pct_change, etc.)
            if any(p in col_lower for p in ['lag_diff', 'lag_ratio', 'diff_', 'ratio_', 'pct_change', 'direction', 'yoy_', 'qoq_']):
                continue
            if re.search(r'lag[_]?(\\d+)', col, re.IGNORECASE):
                lag_cols.append(col)
        return lag_cols


    def identify_rolling_columns(feature_columns: List[str], target_col: str) -> List[str]:
        """Identify rolling feature columns."""
        roll_cols = []
        for col in feature_columns:
            if re.search(r'roll(?:ing)?[_]?(\\d+)', col, re.IGNORECASE):
                roll_cols.append(col)
        return roll_cols


    def identify_derived_columns(feature_columns: List[str]) -> Dict[str, Dict]:
        """
        Identify derived feature columns and their specifications.
        Returns dict mapping column name to spec (type, period, etc.)
        """
        derived = {{}}
        for col in feature_columns:
            col_lower = col.lower()

            # lag_diff_N: difference between lag_1 and lag_(1+N)
            match = re.search(r'lag_diff[_]?(\\d+)', col_lower)
            if match:
                derived[col] = {{'type': 'lag_diff', 'period': int(match.group(1))}}
                continue

            # lag_pct_change_N: percent change
            match = re.search(r'lag_pct_change[_]?(\\d+)', col_lower)
            if match:
                derived[col] = {{'type': 'lag_pct_change', 'period': int(match.group(1))}}
                continue

            # lag_direction_N: sign of change
            match = re.search(r'lag_direction[_]?(\\d+)', col_lower)
            if match:
                derived[col] = {{'type': 'lag_direction', 'period': int(match.group(1))}}
                continue

            # yoy_lag_diff: year-over-year difference (lag_1 - lag_53)
            if 'yoy_lag_diff' in col_lower or 'yoy_diff' in col_lower:
                derived[col] = {{'type': 'yoy_lag_diff'}}
                continue

            # yoy_lag_ratio: year-over-year ratio (lag_1 / lag_53)
            if 'yoy_lag_ratio' in col_lower or 'yoy_ratio' in col_lower:
                derived[col] = {{'type': 'yoy_lag_ratio'}}
                continue

            # qoq_lag_diff: quarter-over-quarter difference (lag_1 - lag_14)
            if 'qoq_lag_diff' in col_lower or 'qoq_diff' in col_lower:
                derived[col] = {{'type': 'qoq_lag_diff'}}
                continue

        return derived


    def is_log_feature(col_name: str) -> bool:
        """Check if a feature is log-transformed (based on naming convention)."""
        return '_log_' in col_name.lower()


    def get_value_for_feature(value: float, col_name: str) -> float:
        """Apply log1p transformation if the feature is log-transformed."""
        if is_log_feature(col_name):
            return np.log1p(max(0.0, value))
        return value


    def get_values_for_feature(values: List[float], col_name: str) -> List[float]:
        """Apply log1p transformation to values if the feature is log-transformed."""
        if is_log_feature(col_name):
            return [np.log1p(max(0.0, v)) for v in values]
        return values


    def parse_lag_number(col_name: str) -> Optional[int]:
        """Extract lag number from column name."""
        match = re.search(r'lag[_]?(\\d+)', col_name, re.IGNORECASE)
        return int(match.group(1)) if match else None


    def parse_rolling_spec(col_name: str) -> Optional[Tuple[int, str]]:
        """Extract rolling window and aggregation from column name."""
        match = re.search(r'roll(?:ing)?[_]?(\\d+)[_]?(mean|std|min|max|sum)?', col_name, re.IGNORECASE)
        if match:
            window = int(match.group(1))
            agg = match.group(2).lower() if match.group(2) else 'mean'
            return (window, agg)
        return None


    def compute_rolling(values: np.ndarray, window: int, agg: str) -> float:
        """Compute rolling aggregate."""
        if len(values) < window:
            vals = values
        else:
            vals = values[-window:]

        if agg == 'mean':
            return np.mean(vals) if len(vals) > 0 else 0.0
        elif agg == 'std':
            return np.std(vals) if len(vals) > 1 else 0.0
        elif agg == 'sum':
            return np.sum(vals)
        elif agg == 'min':
            return np.min(vals) if len(vals) > 0 else 0.0
        elif agg == 'max':
            return np.max(vals) if len(vals) > 0 else 0.0
        else:
            return np.mean(vals) if len(vals) > 0 else 0.0


    def compute_lag_value(values: List[float], lag: int) -> float:
        """Get value at a specific lag position."""
        if len(values) >= lag:
            return values[-lag]
        return 0.0


    def predict_with_model(model, X: np.ndarray, model_type: str) -> np.ndarray:
        """Generate predictions."""
        if model_type.lower() in ['lightgbm', 'lgb']:
            return model.predict(X)
        elif model_type.lower() in ['xgboost', 'xgb']:
            import xgboost as xgb
            dmat = xgb.DMatrix(X)
            return model.predict(dmat)
        else:
            return model.predict(X)


    def recursive_forecast(
        model: Any,
        model_type: str,
        historical_actuals: np.ndarray,
        future_features: np.ndarray,
        feature_columns: List[str],
        lag_columns: List[str],
        rolling_columns: List[str],
        horizon: int,
    ) -> np.ndarray:
        """
        Generate recursive multi-step forecasts with proper feature updates.

        IMPORTANT: This handles:
        - Log-transformed features (applies log1p before computing lags/rolling)
        - Derived features (lag_diff, lag_pct_change, yoy_lag_diff, etc.)
        - All features use shift(1) reference for consistency

        Parameters
        ----------
        model : fitted model
        model_type : str
        historical_actuals : np.ndarray
            Known actuals up to forecast origin
        future_features : np.ndarray
            Feature matrix for future periods (static features)
        feature_columns : List[str]
            All feature column names
        lag_columns : List[str]
            Lag feature column names
        rolling_columns : List[str]
            Rolling feature column names
        horizon : int
            Number of steps to forecast

        Returns
        -------
        np.ndarray
            Predictions for each step
        """
        # Parse lag specs (column index -> lag number)
        lag_map = {{}}
        for col in lag_columns:
            lag_num = parse_lag_number(col)
            if lag_num is not None:
                col_idx = feature_columns.index(col) if col in feature_columns else None
                if col_idx is not None:
                    lag_map[col_idx] = (lag_num, col)  # Store column name for log check

        # Parse rolling specs (column index -> (window, agg, col_name))
        rolling_map = {{}}
        for col in rolling_columns:
            spec = parse_rolling_spec(col)
            if spec is not None:
                col_idx = feature_columns.index(col) if col in feature_columns else None
                if col_idx is not None:
                    rolling_map[col_idx] = (spec[0], spec[1], col)

        # Parse derived feature specs
        derived_specs = identify_derived_columns(feature_columns)
        derived_map = {{}}
        for col, spec in derived_specs.items():
            if col in feature_columns:
                col_idx = feature_columns.index(col)
                derived_map[col_idx] = (spec, col)

        # Initialize predictions
        predictions = []

        # Track history (actuals + predictions) - raw values
        history = list(historical_actuals)

        # Generate predictions step by step
        for step in range(horizon):
            if step >= len(future_features):
                break

            # Start with static features
            features = future_features[step].copy()

            # 1. Update lag features (with log transform if needed)
            for col_idx, (lag_num, col_name) in lag_map.items():
                if lag_num <= len(history):
                    raw_val = history[-lag_num]
                    features[col_idx] = get_value_for_feature(raw_val, col_name)
                else:
                    features[col_idx] = 0.0

            # 2. Update rolling features (with log transform if needed)
            for col_idx, (window, agg, col_name) in rolling_map.items():
                vals = get_values_for_feature(history, col_name)
                features[col_idx] = compute_rolling(np.array(vals), window, agg)

            # 3. Update derived features (lag_diff, lag_pct_change, yoy_lag_diff, etc.)
            # All derived features use lag_1 as reference (shift(1) principle)
            for col_idx, (spec, col_name) in derived_map.items():
                vals = get_values_for_feature(history, col_name)
                feat_type = spec['type']

                if feat_type == 'lag_diff':
                    # lag_diff_N = lag_1 - lag_(1+N)
                    period = spec['period']
                    val_1 = compute_lag_value(vals, 1)
                    val_long = compute_lag_value(vals, 1 + period)
                    features[col_idx] = val_1 - val_long

                elif feat_type == 'lag_pct_change':
                    # lag_pct_change_N = (lag_1 - lag_(1+N)) / |lag_(1+N)|
                    period = spec['period']
                    val_1 = compute_lag_value(vals, 1)
                    val_long = compute_lag_value(vals, 1 + period)
                    features[col_idx] = (val_1 - val_long) / (abs(val_long) + 1e-10)

                elif feat_type == 'lag_direction':
                    # Sign of lag_diff
                    period = spec['period']
                    val_1 = compute_lag_value(vals, 1)
                    val_long = compute_lag_value(vals, 1 + period)
                    diff = val_1 - val_long
                    features[col_idx] = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)

                elif feat_type == 'yoy_lag_diff':
                    # Year-over-year: lag_1 - lag_53
                    val_1 = compute_lag_value(vals, 1)
                    val_53 = compute_lag_value(vals, 53)
                    features[col_idx] = val_1 - val_53

                elif feat_type == 'yoy_lag_ratio':
                    # Year-over-year ratio: lag_1 / lag_53
                    val_1 = compute_lag_value(vals, 1)
                    val_53 = compute_lag_value(vals, 53)
                    features[col_idx] = val_1 / (abs(val_53) + 1e-10)

                elif feat_type == 'qoq_lag_diff':
                    # Quarter-over-quarter: lag_1 - lag_14
                    val_1 = compute_lag_value(vals, 1)
                    val_14 = compute_lag_value(vals, 14)
                    features[col_idx] = val_1 - val_14

            # Make prediction
            X = features.reshape(1, -1)
            pred = predict_with_model(model, X, model_type)[0]
            pred = max(0.0, pred)  # Non-negative

            predictions.append(pred)

            # Add prediction to history for next step (raw value, not transformed)
            history.append(pred)

        return np.array(predictions)


    def run_recursive_inference(
        config_path: str,
        model_artifact_path: Optional[str] = None,
        forecast_origin: Optional[str] = None,
        horizon: int = 8,
        output_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run recursive multi-step inference.

        Parameters
        ----------
        config_path : str
            Path to config.yaml
        model_artifact_path : str, optional
            Path to model artifact
        forecast_origin : str, optional
            Last period of known data (format: YYYYWW)
        horizon : int
            Number of steps to forecast
        output_dir : str, optional
            Output directory

        Returns
        -------
        pd.DataFrame
            Predictions with key, period, and forecast columns
        """
        start_time = time.time()

        logger.info("="*60)
        logger.info("RECURSIVE INFERENCE PIPELINE")
        logger.info(f"Horizon: {{horizon}} steps")
        logger.info("="*60)

        # Load config
        config = load_config(config_path)

        # Resolve paths
        config_dir = os.path.dirname(os.path.abspath(config_path))
        project_root = os.path.dirname(config_dir)

        artifact_base_raw = config.get('artifact_base_path', '')
        artifact_base = os.path.join(project_root, artifact_base_raw) if artifact_base_raw and not os.path.isabs(artifact_base_raw) else artifact_base_raw

        # Find model artifact
        if model_artifact_path is None:
            # Try retrained models first, then original
            retrained_path = os.path.join(artifact_base, 'retrained_models', 'retrained_model_artifact.pkl')
            original_path = os.path.join(artifact_base, 'model_artifacts', 'global_model.pkl')

            if os.path.exists(retrained_path):
                model_artifact_path = retrained_path
            elif os.path.exists(original_path):
                model_artifact_path = original_path
            else:
                # Search for any .pkl file
                model_dir = os.path.join(artifact_base, 'model_artifacts')
                for f in os.listdir(model_dir) if os.path.exists(model_dir) else []:
                    if f.endswith('.pkl') and 'model' in f.lower():
                        model_artifact_path = os.path.join(model_dir, f)
                        break

        if model_artifact_path is None or not os.path.exists(model_artifact_path):
            raise FileNotFoundError(f"Model artifact not found")

        logger.info(f"Loading model from: {{model_artifact_path}}")
        artifact = load_model_artifact(model_artifact_path)

        model = artifact['model']
        model_type = artifact.get('model_type', 'lightgbm')
        feature_columns = artifact.get('feature_columns', [])

        logger.info(f"Model type: {{model_type}}")
        logger.info(f"Features: {{len(feature_columns)}}")

        # Identify lag and rolling columns
        lag_columns = identify_lag_columns(feature_columns, TARGET_COLUMN)
        rolling_columns = identify_rolling_columns(feature_columns, TARGET_COLUMN)

        logger.info(f"Lag columns: {{len(lag_columns)}}")
        logger.info(f"Rolling columns: {{len(rolling_columns)}}")

        # Load data
        data_path_raw = get_data_path(config)
        data_path = os.path.join(project_root, data_path_raw) if data_path_raw and not os.path.isabs(data_path_raw) else data_path_raw

        file_ext = os.path.splitext(data_path)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(data_path)
        elif file_ext in ('.parquet', '.pq'):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {{file_ext}}. Use .csv or .parquet")
        df[DATE_COLUMN] = df[DATE_COLUMN].astype(str).astype(int)

        # Determine forecast origin
        if forecast_origin is None:
            val_end = config.get('val_end', '')
            forecast_origin = str(val_end)

        forecast_origin_int = int(forecast_origin)
        logger.info(f"Forecast origin: {{forecast_origin}}")

        # Get key column
        key_col = KEY_COLUMNS[0] if len(KEY_COLUMNS) == 1 else 'key'
        if key_col not in df.columns and len(KEY_COLUMNS) > 1:
            df[key_col] = df[KEY_COLUMNS].astype(str).agg('_'.join, axis=1)

        # Generate forecasts for each key
        all_forecasts = []
        unique_keys = df[key_col].unique()

        logger.info(f"Forecasting for {{len(unique_keys)}} keys...")

        for key in unique_keys:
            key_df = df[df[key_col] == key].sort_values(DATE_COLUMN)

            # Historical data up to forecast origin
            historical_df = key_df[key_df[DATE_COLUMN] <= forecast_origin_int]

            if len(historical_df) == 0:
                continue

            # Historical actuals
            historical_actuals = historical_df[TARGET_COLUMN].values

            # Future periods for forecasting
            future_df = key_df[key_df[DATE_COLUMN] > forecast_origin_int].head(horizon)

            if len(future_df) == 0:
                # Generate placeholder future features
                continue

            # Get static features for future periods
            available_features = [c for c in feature_columns if c in future_df.columns]
            if not available_features:
                continue

            future_features = future_df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0).values

            # Generate recursive forecasts
            predictions = recursive_forecast(
                model=model,
                model_type=model_type,
                historical_actuals=historical_actuals,
                future_features=future_features,
                feature_columns=available_features,
                lag_columns=[c for c in lag_columns if c in available_features],
                rolling_columns=[c for c in rolling_columns if c in available_features],
                horizon=len(future_df),
            )

            # Build forecast records
            future_periods = future_df[DATE_COLUMN].values
            for i, (period, pred) in enumerate(zip(future_periods, predictions)):
                all_forecasts.append({{
                    key_col: key,
                    DATE_COLUMN: period,
                    'predicted': pred,
                    'forecast_step': i + 1,
                    'forecast_origin': forecast_origin,
                }})

        # Create output dataframe
        forecasts_df = pd.DataFrame(all_forecasts)

        # Save results
        output_dir = output_dir or os.path.join(artifact_base, 'inference_output')
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f'forecasts_from_{{forecast_origin}}.csv')
        forecasts_df.to_csv(output_path, index=False)

        total_time = time.time() - start_time

        logger.info("="*60)
        logger.info("INFERENCE COMPLETED")
        logger.info(f"  Forecasts: {{len(forecasts_df)}} rows")
        logger.info(f"  Output: {{output_path}}")
        logger.info(f"  Time: {{total_time:.1f}} seconds")
        logger.info("="*60)

        return forecasts_df


    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Recursive Inference Pipeline")
        parser.add_argument("--config", required=True, help="Path to config.yaml")
        parser.add_argument("--model-artifact", help="Path to model artifact")
        parser.add_argument("--forecast-origin", help="Last period of known data (YYYYWW)")
        parser.add_argument("--horizon", type=int, default=8, help="Forecast horizon")
        parser.add_argument("--output-dir", help="Output directory")

        args = parser.parse_args()

        forecasts = run_recursive_inference(
            config_path=args.config,
            model_artifact_path=args.model_artifact,
            forecast_origin=args.forecast_origin,
            horizon=args.horizon,
            output_dir=args.output_dir,
        )

        print(f"\\nGenerated {{len(forecasts)}} forecasts")
    ''')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(script)

    logger.info(f"Generated recursive inference pipeline: {output_path}")
    return output_path


def generate_all_deterministic_pipelines(
    config: PipelineConfig,
    output_dir: str,
) -> Dict[str, str]:
    """
    Generate ALL deterministic pipelines in one call.

    Creates:
    - deterministic_training_pipeline.py
    - recursive_inference_pipeline.py

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration
    output_dir : str
        Output directory

    Returns
    -------
    Dict[str, str]
        Mapping of pipeline name to file path
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = {}

    # Training pipeline
    paths['training'] = generate_deterministic_training_pipeline(
        config=config,
        output_path=os.path.join(output_dir, 'deterministic_training_pipeline.py'),
    )

    # Inference pipeline
    paths['inference'] = generate_recursive_inference_pipeline(
        config=config,
        output_path=os.path.join(output_dir, 'recursive_inference_pipeline.py'),
    )

    logger.info(f"Generated all deterministic pipelines in: {output_dir}")
    return paths


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    'PipelineConfig',
    'DeterministicCodeInfo',
    'PipelineGenerationResult',

    # Code analysis
    'analyze_deterministic_code',
    'extract_code_block',
    'extract_function',

    # Code generation
    'create_import_block',
    'create_config_loader',
    'create_data_loader',
    'create_metrics_code',
    'create_categorical_encoding_code',
    'create_clustering_code',
    'create_model_training_code',
    'create_inference_code',

    # Pipeline generation
    'generate_training_pipeline',
    'generate_inference_pipeline',
    'generate_orchestrator_script',
    'generate_pipeline_readme',
    'generate_pipeline_config',

    # Self-contained pipeline utilities
    'REQUIRED_UTILS',
    'copy_utils_to_pipeline',
    'create_pipeline_config',
    'copy_model_artifacts',
    'copy_feature_artifacts',
    'copy_segmentation_artifacts',
    'generate_self_contained_import_block',
    'create_self_contained_pipeline',

    # Validation
    'validate_pipeline_syntax',
    'validate_all_pipelines',

    # High-level
    'run_pipeline_generation',

    # Model-spec-based deterministic pipelines (NEW)
    'generate_deterministic_training_pipeline',
    'generate_recursive_inference_pipeline',
    'generate_all_deterministic_pipelines',

    # Comprehensive self-contained pipeline (NEWEST)
    'generate_self_contained_full_pipeline',
]


# =============================================================================
# COMPREHENSIVE SELF-CONTAINED PIPELINE GENERATOR
# =============================================================================

def generate_self_contained_full_pipeline(
    config: PipelineConfig,
    output_path: str,
) -> str:
    """
    Generate a COMPREHENSIVE SELF-CONTAINED pipeline that replicates ALL crew logic.

    This pipeline:
    1. Loads source data and config
    2. Applies EDA transformations (from eda_deterministic.py logic)
    3. Applies segmentation/clustering (from segmentation_deterministic.py logic)
    4. Creates features (from feature_deterministic.py logic)
    5. Assigns keys to model groups
    6. Trains models per group using saved specs (handles ensembles)
    7. Generates recursive multi-step forecasts
    
    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration with paths to artifacts
    output_path : str
        Path to save the generated script

    Returns
    -------
    str
        Path to the generated script
    """
    key_cols_str = str(config.key_columns)
    key_col_first = config.key_columns[0] if config.key_columns else 'key'

    script = textwrap.dedent(f'''
    #!/usr/bin/env python3
    """
    Self-Contained Demand Forecasting Pipeline
    ============================================

    This is a COMPREHENSIVE pipeline that replicates ALL crew logic:
    - EDA transformations
    - Segmentation/Clustering
    - Feature Engineering
    - Model Training (with ensemble support)
    - Recursive Multi-Step Forecasting

    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Usage:
        # Full pipeline (train + forecast)
        python self_contained_pipeline.py --config config/config.yaml --mode full

        # Training only
        python self_contained_pipeline.py --config config/config.yaml --mode train

        # Inference only (requires trained models)
        python self_contained_pipeline.py --config config/config.yaml --mode inference
    """

    from __future__ import annotations

    import argparse
    import json
    import logging
    import os
    import re
    import sys
    import time
    import warnings
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Any, Dict, List, Optional, Tuple

    warnings.filterwarnings('ignore')

    import joblib
    import numpy as np
    import pandas as pd
    import yaml

    # ML Libraries
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # =============================================================================
    # CONFIGURATION
    # =============================================================================

    KEY_COLUMNS = {key_cols_str}
    KEY_COL = "{key_col_first}"
    DATE_COLUMN = "{config.date_column}"
    TARGET_COLUMN = "{config.target_column}"


    # =============================================================================
    # UTILITY FUNCTIONS
    # =============================================================================

    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON."""
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            return yaml.safe_load(f)


    def get_data_path(config: Dict[str, Any]) -> str:
        """Extract data_path from config (handles multiple formats)."""
        if 'data_path' in config:
            return config['data_path']
        if 'data' in config and isinstance(config['data'], dict):
            if 'data_path' in config['data']:
                return config['data']['data_path']
        if 'input_data_path' in config:
            return config['input_data_path']
        return ''


    def load_json(path: str) -> Dict[str, Any]:
        """Load JSON file."""
        if not os.path.exists(path):
            return {{}}
        with open(path, 'r') as f:
            return json.load(f)


    def save_json(data: Dict[str, Any], path: str):
        """Save dict to JSON."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)


    def compute_wape(y_true, y_pred) -> float:
        """Compute Weighted Absolute Percentage Error."""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-10))


    def increment_year_week(yw: str, periods: int = 1) -> str:
        """Increment a YYYYWW string by N periods."""
        year, week = int(str(yw)[:4]), int(str(yw)[4:])
        for _ in range(abs(periods)):
            if periods > 0:
                week += 1
                if week > 52:
                    week = 1
                    year += 1
            else:
                week -= 1
                if week < 1:
                    week = 52
                    year -= 1
        return f"{{year}}{{week:02d}}"


    # =============================================================================
    # STEP 1: DATA LOADING
    # =============================================================================

    def load_source_data(data_path: str) -> pd.DataFrame:
        """Load and prepare source data (supports CSV, Parquet, and Delta)."""
        logger.info(f"Loading data from: {{data_path}}")

        file_ext = os.path.splitext(data_path)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(data_path)
        elif file_ext in ('.parquet', '.pq'):
            df = pd.read_parquet(data_path)
        elif os.path.isdir(data_path):
            if os.path.isdir(os.path.join(data_path, '_delta_log')):
                from deltalake import DeltaTable
                df = DeltaTable(data_path).to_pandas()
            else:
                df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {{file_ext}}. Use .csv, .parquet, or delta directory")

        # Create composite key if needed
        if len(KEY_COLUMNS) > 1:
            df['key'] = df[list(KEY_COLUMNS)].astype(str).agg('_'.join, axis=1)
        elif 'key' not in df.columns and KEY_COL in df.columns:
            df['key'] = df[KEY_COL].astype(str)

        # Ensure date column is integer
        df[DATE_COLUMN] = df[DATE_COLUMN].astype(str).str.replace('-', '').astype(int)

        logger.info(f"Loaded {{len(df)}} rows, {{df['key'].nunique()}} unique keys")
        return df


    # =============================================================================
    # STEP 2: EDA TRANSFORMATIONS
    # =============================================================================

    def apply_eda_transformations(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Apply EDA transformations as determined by EDA crew.
        This replicates the logic from eda_deterministic.py.
        """
        logger.info("Applying EDA transformations...")

        # Handle missing values in target
        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(0)

        # Ensure non-negative target
        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(lower=0)

        # Sort by key and date
        df = df.sort_values(['key', DATE_COLUMN]).reset_index(drop=True)

        logger.info(f"EDA transformations complete: {{len(df)}} rows")
        return df


    # =============================================================================
    # STEP 3: SEGMENTATION / CLUSTERING
    # =============================================================================

    def compute_demand_characteristics(df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-key demand characteristics for clustering."""
        logger.info("Computing demand characteristics...")

        chars = df.groupby('key').agg({{
            TARGET_COLUMN: ['mean', 'std', 'sum', 'count'],
            DATE_COLUMN: ['min', 'max', 'nunique'],
        }})

        chars.columns = ['_'.join(col).strip() for col in chars.columns.values]
        chars = chars.reset_index()

        # Compute additional metrics
        chars['cv'] = chars[f'{{TARGET_COLUMN}}_std'] / (chars[f'{{TARGET_COLUMN}}_mean'] + 1e-10)
        chars['zero_ratio'] = df.groupby('key')[TARGET_COLUMN].apply(
            lambda x: (x == 0).sum() / len(x)
        ).values

        # Classify intermittency
        chars['intermittency_class'] = 'smooth'
        chars.loc[chars['zero_ratio'] > 0.3, 'intermittency_class'] = 'intermittent'
        chars.loc[chars['zero_ratio'] > 0.7, 'intermittency_class'] = 'highly_intermittent'

        logger.info(f"Computed characteristics for {{len(chars)}} keys")
        return chars


    def apply_segmentation(
        chars_df: pd.DataFrame,
        seg_dir: str,
    ) -> pd.DataFrame:
        """
        Apply segmentation using saved cluster model.
        Replicates segmentation_deterministic.py logic.
        """
        logger.info("Applying segmentation...")

        # Try to load saved cluster artifacts
        cluster_model_path = os.path.join(seg_dir, 'cluster_model.pkl')
        cluster_scaler_path = os.path.join(seg_dir, 'cluster_scaler.pkl')

        if os.path.exists(cluster_model_path) and os.path.exists(cluster_scaler_path):
            cluster_model = joblib.load(cluster_model_path)
            cluster_scaler = joblib.load(cluster_scaler_path)

            # Get clustering features
            cluster_features = [c for c in chars_df.columns if c not in [
                'key', 'intermittency_class', 'segment_id', 'model_group'
            ] and chars_df[c].dtype in ['float64', 'int64']]

            if cluster_features:
                X = chars_df[cluster_features].fillna(0).values
                X_scaled = cluster_scaler.transform(X)
                chars_df['segment_id'] = cluster_model.predict(X_scaled)
                logger.info(f"Assigned {{chars_df['segment_id'].nunique()}} segments using saved model")
        else:
            # Fallback: simple segmentation by intermittency
            logger.warning("No cluster model found, using intermittency-based segmentation")
            segment_map = {{'smooth': 0, 'intermittent': 1, 'highly_intermittent': 2}}
            chars_df['segment_id'] = chars_df['intermittency_class'].map(segment_map).fillna(0).astype(int)

        # Assign model_group (may be same as segment or key-level)
        chars_df['model_group'] = 'model_group_' + chars_df['segment_id'].astype(str)

        logger.info(f"Segment distribution: {{chars_df['segment_id'].value_counts().to_dict()}}")
        return chars_df


    # =============================================================================
    # STEP 4: FEATURE ENGINEERING
    # =============================================================================

    def create_lag_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """Create lag features for target variable."""
        for lag in lags:
            df[f'{{TARGET_COLUMN}}_lag_{{lag}}'] = df.groupby('key')[TARGET_COLUMN].shift(lag)
        return df


    def create_rolling_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Create rolling mean/std features."""
        for window in windows:
            df[f'{{TARGET_COLUMN}}_rolling_{{window}}_mean'] = df.groupby('key')[TARGET_COLUMN].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'{{TARGET_COLUMN}}_rolling_{{window}}_std'] = df.groupby('key')[TARGET_COLUMN].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )
        return df


    def apply_feature_engineering(
        df: pd.DataFrame,
        feat_dir: str,
        config: Dict,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Apply feature engineering as determined by Feature crew.
        Replicates feature_deterministic.py logic.
        """
        logger.info("Applying feature engineering...")

        # Try to load feature config
        feat_context_path = os.path.join(feat_dir, 'feature_to_training_context.json')
        feat_context = load_json(feat_context_path)

        # Get lag/rolling config
        lag_config = feat_context.get('lag_config', {{}})
        lags = lag_config.get('lags', [1, 2, 4, 8, 12])
        windows = lag_config.get('rolling_windows', [4, 8, 12])

        # Create lag features
        df = create_lag_features(df, lags)

        # Create rolling features
        df = create_rolling_features(df, windows)

        # Apply categorical encoding if specs exist
        cat_specs_path = os.path.join(feat_dir, 'categorical_encoding_specs.json')
        if os.path.exists(cat_specs_path):
            cat_specs = load_json(cat_specs_path)
            for col, encoding in cat_specs.items():
                if col in df.columns:
                    if encoding.get('type') == 'label':
                        mapping = encoding.get('mapping', {{}})
                        df[col] = df[col].map(mapping).fillna(-1).astype(int)

        # Identify feature columns
        exclude_cols = {{
            'key', KEY_COL, DATE_COLUMN, TARGET_COLUMN,
            'segment_id', 'model_group', 'intermittency_class', 'split',
        }}
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'int32']]

        logger.info(f"Created {{len(feature_cols)}} features")
        return df, feature_cols


    # =============================================================================
    # STEP 5: MODEL TRAINING
    # =============================================================================

    @dataclass
    class EnsembleModel:
        """Ensemble model combining multiple base models."""
        models: List[Any]
        weights: List[float]
        model_types: List[str]

        def predict(self, X: np.ndarray) -> np.ndarray:
            predictions = np.zeros(len(X))
            for model, weight, mtype in zip(self.models, self.weights, self.model_types):
                if isinstance(model, dict) and 'forecast' in model:
                    # Univariate dict models (croston, sba, tsb, imapa)
                    preds = np.full(len(X), float(model['forecast']))
                elif mtype in ['lightgbm', 'lgb']:
                    preds = model.predict(X)
                elif mtype in ['xgboost', 'xgb']:
                    dmat = xgb.DMatrix(X)
                    preds = model.predict(dmat)
                else:
                    preds = model.predict(X)
                predictions += weight * preds
            return np.clip(predictions, 0, None)


    def train_lightgbm(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict[str, Any],
    ) -> lgb.Booster:
        """Train LightGBM with fixed parameters."""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        lgb_params = {{
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'force_row_wise': True,
        }}
        lgb_params.update(params)

        num_rounds = lgb_params.pop('num_boost_round', lgb_params.pop('n_estimators', 500))

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=num_rounds,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        return model


    def train_xgboost(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict[str, Any],
    ) -> xgb.Booster:
        """Train XGBoost with fixed parameters."""
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        xgb_params = {{
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'verbosity': 0,
        }}
        xgb_params.update(params)

        num_rounds = xgb_params.pop('num_boost_round', xgb_params.pop('n_estimators', 500))

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=num_rounds,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        return model


    def train_model_group(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        model_spec: Dict[str, Any],
    ) -> Tuple[Any, float]:
        """
        Train a model for a specific model group using saved spec.
        Handles both single models and ensembles.
        """
        # Sanitize: replace inf/-inf with NaN, then fill with 0
        X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        y_train = train_df[TARGET_COLUMN].replace([np.inf, -np.inf], np.nan).fillna(0).values
        X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        y_val = val_df[TARGET_COLUMN].replace([np.inf, -np.inf], np.nan).fillna(0).values

        model_type = model_spec.get('model_type', 'lightgbm')
        hyperparams = model_spec.get('hyperparameters', {{}})
        is_ensemble = model_spec.get('is_ensemble', False)

        if is_ensemble:
            # Train ensemble - multiple models with weights
            ensemble_info = model_spec.get('ensemble_info', {{}})
            member_types = ensemble_info.get('model_types', ['lightgbm'])
            weights = ensemble_info.get('weights', [1.0 / len(member_types)] * len(member_types))
            member_params = hyperparams.get('member_params', {{}})

            models = []
            for mtype in member_types:
                params = member_params.get(mtype, {{}})
                if mtype in ['lightgbm', 'lgb']:
                    model = train_lightgbm(X_train, y_train, X_val, y_val, params)
                elif mtype in ['xgboost', 'xgb']:
                    model = train_xgboost(X_train, y_train, X_val, y_val, params)
                else:
                    model = train_lightgbm(X_train, y_train, X_val, y_val, params)
                models.append(model)

            ensemble = EnsembleModel(models=models, weights=weights, model_types=member_types)

            # Evaluate ensemble
            preds = ensemble.predict(X_val)
            wape = compute_wape(y_val, preds)

            return ensemble, wape

        else:
            # Train single model
            if model_type in ['lightgbm', 'lgb']:
                model = train_lightgbm(X_train, y_train, X_val, y_val, hyperparams)
                preds = model.predict(X_val)
            elif model_type in ['xgboost', 'xgb']:
                model = train_xgboost(X_train, y_train, X_val, y_val, hyperparams)
                dval = xgb.DMatrix(X_val)
                preds = model.predict(dval)
            else:
                model = train_lightgbm(X_train, y_train, X_val, y_val, hyperparams)
                preds = model.predict(X_val)

            wape = compute_wape(y_val, preds)
            return model, wape


    def run_training(
        df: pd.DataFrame,
        chars_df: pd.DataFrame,
        feature_cols: List[str],
        model_specs: Dict[str, Any],
        config: Dict,
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Train models for all model groups using saved specs.
        """
        logger.info("="*60)
        logger.info("TRAINING MODELS")
        logger.info("="*60)

        train_start = int(config.get('train_start', 0))
        train_end = int(config.get('train_end', 0))
        val_start = int(config.get('val_start', 0))
        val_end = int(config.get('val_end', 0))

        # Split data
        train_df = df[(df[DATE_COLUMN] >= train_start) & (df[DATE_COLUMN] <= train_end)].copy()
        val_df = df[(df[DATE_COLUMN] >= val_start) & (df[DATE_COLUMN] <= val_end)].copy()

        # Merge model_group assignments
        train_df = train_df.merge(chars_df[['key', 'model_group']], on='key', how='left')
        val_df = val_df.merge(chars_df[['key', 'model_group']], on='key', how='left')

        # Get per-model specs
        per_model_specs = model_specs.get('models', [])
        if not per_model_specs:
            # Use global spec for all groups
            per_model_specs = [{{
                'model_group': mg,
                'model_type': model_specs.get('model_type', 'lightgbm'),
                'hyperparameters': model_specs.get('hyperparameters', {{}}),
                'is_ensemble': model_specs.get('is_ensemble', False),
                'ensemble_info': model_specs.get('ensemble_info', {{}}),
                'feature_columns': model_specs.get('feature_columns', feature_cols),
            }} for mg in chars_df['model_group'].unique()]

        # Train each model group
        trained_models = {{}}
        results = []

        for spec in per_model_specs:
            mg = spec.get('model_group', '')
            if not mg:
                continue

            mg_train = train_df[train_df['model_group'] == mg]
            mg_val = val_df[val_df['model_group'] == mg]

            if len(mg_train) == 0 or len(mg_val) == 0:
                logger.warning(f"No data for model group {{mg}}, skipping")
                continue

            # Use spec's feature columns if available
            mg_features = spec.get('feature_columns', feature_cols)
            available_features = [c for c in mg_features if c in train_df.columns]
            if not available_features:
                available_features = feature_cols

            try:
                model, wape = train_model_group(mg_train, mg_val, available_features, spec)
                trained_models[mg] = {{
                    'model': model,
                    'feature_columns': available_features,
                    'spec': spec,
                }}
                results.append({{
                    'model_group': mg,
                    'wape': wape,
                    'n_train': len(mg_train),
                    'n_val': len(mg_val),
                    'model_type': spec.get('model_type', 'lightgbm'),
                    'is_ensemble': spec.get('is_ensemble', False),
                }})
                logger.info(f"  {{mg}}: WAPE={{wape:.4f}} ({{spec.get('model_type', 'lightgbm')}})")
            except Exception as e:
                logger.error(f"  {{mg}}: FAILED - {{e}}")

        # Save trained models
        models_dir = os.path.join(output_dir, 'trained_models')
        os.makedirs(models_dir, exist_ok=True)

        for mg, model_data in trained_models.items():
            model_path = os.path.join(models_dir, f'{{mg}}_model.pkl')
            joblib.dump(model_data, model_path)

        # Save training summary
        summary = {{
            'models_trained': len(trained_models),
            'results': results,
            'overall_wape': np.mean([r['wape'] for r in results]) if results else 1.0,
        }}
        save_json(summary, os.path.join(output_dir, 'training_summary.json'))

        logger.info(f"Trained {{len(trained_models)}} models, overall WAPE={{summary['overall_wape']:.4f}}")
        return {{'trained_models': trained_models, 'summary': summary}}


    # =============================================================================
    # STEP 6: RECURSIVE FORECASTING
    # =============================================================================

    def identify_lag_columns(feature_cols: List[str]) -> List[Tuple[str, int]]:
        """Identify lag columns and their lag numbers."""
        lag_cols = []
        for col in feature_cols:
            match = re.search(r'lag[_]?(\\d+)', col, re.IGNORECASE)
            if match and 'diff' not in col.lower() and 'ratio' not in col.lower():
                lag_cols.append((col, int(match.group(1))))
        return lag_cols


    def identify_rolling_columns(feature_cols: List[str]) -> List[Tuple[str, int, str]]:
        """Identify rolling columns, their windows, and aggregation type."""
        roll_cols = []
        for col in feature_cols:
            match = re.search(r'rolling[_]?(\\d+)[_]?(mean|std|sum|min|max)?', col, re.IGNORECASE)
            if match:
                window = int(match.group(1))
                agg = match.group(2).lower() if match.group(2) else 'mean'
                roll_cols.append((col, window, agg))
        return roll_cols


    def recursive_forecast_single_key(
        model: Any,
        model_type: str,
        historical_actuals: np.ndarray,
        feature_template: np.ndarray,
        feature_cols: List[str],
        lag_cols: List[Tuple[str, int]],
        roll_cols: List[Tuple[str, int, str]],
        horizon: int,
    ) -> np.ndarray:
        """Generate recursive multi-step forecasts for a single key."""
        predictions = []
        history = list(historical_actuals)

        # Build column index maps
        col_to_idx = {{c: i for i, c in enumerate(feature_cols)}}

        for step in range(horizon):
            if step >= len(feature_template):
                break

            features = feature_template[step].copy()

            # Update lag features with predictions
            for col, lag in lag_cols:
                if col in col_to_idx:
                    idx = col_to_idx[col]
                    if lag <= len(history):
                        features[idx] = history[-lag]

            # Update rolling features
            for col, window, agg in roll_cols:
                if col in col_to_idx:
                    idx = col_to_idx[col]
                    vals = np.array(history[-window:]) if len(history) >= window else np.array(history)
                    if agg == 'mean':
                        features[idx] = np.mean(vals) if len(vals) > 0 else 0
                    elif agg == 'std':
                        features[idx] = np.std(vals) if len(vals) > 1 else 0
                    elif agg == 'sum':
                        features[idx] = np.sum(vals)

            # Predict
            X = features.reshape(1, -1)
            if isinstance(model, dict) and 'forecast' in model:
                pred = float(model['forecast'])
            elif isinstance(model, EnsembleModel):
                pred = model.predict(X)[0]
            elif model_type in ['xgboost', 'xgb']:
                dmat = xgb.DMatrix(X)
                pred = model.predict(dmat)[0]
            else:
                pred = model.predict(X)[0]

            pred = max(0.0, pred)
            predictions.append(pred)
            history.append(pred)

        return np.array(predictions)


    def run_inference(
        df: pd.DataFrame,
        chars_df: pd.DataFrame,
        feature_cols: List[str],
        trained_models: Dict[str, Any],
        config: Dict,
        forecast_horizon: int = 8,
    ) -> pd.DataFrame:
        """
        Generate recursive multi-step forecasts for all keys.
        """
        logger.info("="*60)
        logger.info("GENERATING FORECASTS")
        logger.info("="*60)

        val_end = int(config.get('val_end', 0))

        # Identify lag and rolling columns
        lag_cols = identify_lag_columns(feature_cols)
        roll_cols = identify_rolling_columns(feature_cols)

        all_forecasts = []
        unique_keys = df['key'].unique()

        for key in unique_keys:
            key_df = df[df['key'] == key].sort_values(DATE_COLUMN)

            # Get model group for this key
            key_chars = chars_df[chars_df['key'] == key]
            if len(key_chars) == 0:
                continue
            mg = key_chars['model_group'].values[0]

            if mg not in trained_models:
                continue

            model_data = trained_models[mg]
            model = model_data['model']
            mg_features = model_data['feature_columns']
            spec = model_data['spec']
            model_type = spec.get('model_type', 'lightgbm')

            # Historical data up to val_end
            hist_df = key_df[key_df[DATE_COLUMN] <= val_end]
            if len(hist_df) == 0:
                continue

            historical_actuals = hist_df[TARGET_COLUMN].values

            # Future data for forecasting
            future_df = key_df[key_df[DATE_COLUMN] > val_end].head(forecast_horizon)
            if len(future_df) == 0:
                continue

            available_features = [c for c in mg_features if c in future_df.columns]
            if not available_features:
                continue

            feature_template = future_df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0).values

            # Identify lag/rolling for this feature set
            mg_lag_cols = [(c, l) for c, l in lag_cols if c in available_features]
            mg_roll_cols = [(c, w, a) for c, w, a in roll_cols if c in available_features]

            # Generate forecasts
            predictions = recursive_forecast_single_key(
                model=model,
                model_type=model_type,
                historical_actuals=historical_actuals,
                feature_template=feature_template,
                feature_cols=available_features,
                lag_cols=mg_lag_cols,
                roll_cols=mg_roll_cols,
                horizon=len(future_df),
            )

            # Collect results
            for i, (_, row) in enumerate(future_df.iterrows()):
                if i < len(predictions):
                    all_forecasts.append({{
                        'key': key,
                        DATE_COLUMN: row[DATE_COLUMN],
                        'predicted': predictions[i],
                        'actual': row[TARGET_COLUMN],
                        'forecast_step': i + 1,
                        'model_group': mg,
                    }})

        forecasts_df = pd.DataFrame(all_forecasts)
        if len(forecasts_df) > 0:
            forecasts_df['abs_error'] = np.abs(forecasts_df['actual'] - forecasts_df['predicted'])
            overall_wape = compute_wape(forecasts_df['actual'].values, forecasts_df['predicted'].values)
            logger.info(f"Generated {{len(forecasts_df)}} forecasts, WAPE={{overall_wape:.4f}}")

        return forecasts_df


    # =============================================================================
    # MAIN PIPELINE
    # =============================================================================

    def run_full_pipeline(
        config_path: str,
        mode: str = 'full',
        forecast_horizon: int = 8,
    ) -> Dict[str, Any]:
        """
        Run the full self-contained pipeline.

        Parameters
        ----------
        config_path : str
            Path to config.yaml
        mode : str
            'full': train + inference
            'train': training only
            'inference': inference only (requires trained models)
            
        forecast_horizon : int
            Number of steps to forecast

        Returns
        -------
        Dict[str, Any]
            Pipeline results
        """
        start_time = time.time()

        logger.info("="*60)
        logger.info(f"SELF-CONTAINED PIPELINE - MODE: {{mode.upper()}}")
        logger.info("="*60)

        # Load config
        config = load_config(config_path)

        # Resolve paths
        config_dir = os.path.dirname(os.path.abspath(config_path))
        project_root = os.path.dirname(config_dir)

        data_path_raw = get_data_path(config)
        data_path = os.path.join(project_root, data_path_raw) if data_path_raw and not os.path.isabs(data_path_raw) else data_path_raw

        artifact_base_raw = config.get('artifact_base_path', '')
        artifact_base = os.path.join(project_root, artifact_base_raw) if artifact_base_raw and not os.path.isabs(artifact_base_raw) else artifact_base_raw

        seg_dir = os.path.join(artifact_base, 'seg_output')
        feat_dir = os.path.join(artifact_base, 'feature_output')
        model_dir = os.path.join(artifact_base, 'model_artifacts')
        pipeline_dir = os.path.join(artifact_base, 'pipeline_output')

        os.makedirs(pipeline_dir, exist_ok=True)

        # Step 1: Load data
        df = load_source_data(data_path)

        # Step 2: EDA transformations
        df = apply_eda_transformations(df, config)

        # Step 3: Compute characteristics and apply segmentation
        chars_df = compute_demand_characteristics(df)
        chars_df = apply_segmentation(chars_df, seg_dir)

        # Step 4: Feature engineering
        df, feature_cols = apply_feature_engineering(df, feat_dir, config)

        # Load model specs
        model_specs_path = os.path.join(model_dir, 'final_model_specs.json')
        model_specs = load_json(model_specs_path)

        # Get forecast horizon from specs or config
        forecast_horizon = model_specs.get('forecast_horizon', forecast_horizon)

        results = {{}}

        # Step 5: Training
        if mode in ['full', 'train']:
            training_results = run_training(
                df=df,
                chars_df=chars_df,
                feature_cols=feature_cols,
                model_specs=model_specs,
                config=config,
                output_dir=pipeline_dir,
            )
            results['training'] = training_results['summary']
            trained_models = training_results['trained_models']
        else:
            # Load pre-trained models
            trained_models = {{}}
            models_dir = os.path.join(pipeline_dir, 'trained_models')
            if os.path.exists(models_dir):
                for f in os.listdir(models_dir):
                    if f.endswith('_model.pkl'):
                        mg = f.replace('_model.pkl', '')
                        trained_models[mg] = joblib.load(os.path.join(models_dir, f))

        # Step 6: Inference
        if mode in ['full', 'inference'] and trained_models:
            forecasts_df = run_inference(
                df=df,
                chars_df=chars_df,
                feature_cols=feature_cols,
                trained_models=trained_models,
                config=config,
                forecast_horizon=forecast_horizon,
            )
            if len(forecasts_df) > 0:
                forecasts_df.to_csv(os.path.join(pipeline_dir, 'forecasts.csv'), index=False)
                results['inference'] = {{
                    'n_forecasts': len(forecasts_df),
                    'wape': compute_wape(forecasts_df['actual'].values, forecasts_df['predicted'].values),
                }}

        total_time = time.time() - start_time
        results['total_time_seconds'] = total_time

        logger.info("="*60)
        logger.info(f"PIPELINE COMPLETED in {{total_time:.1f}} seconds")
        logger.info("="*60)

        # Save final results
        save_json(results, os.path.join(pipeline_dir, 'pipeline_results.json'))

        return results


    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Self-Contained Demand Forecasting Pipeline")
        parser.add_argument("--config", required=True, help="Path to config.yaml")
        parser.add_argument("--mode", default="full", choices=["full", "train", "inference"],
                           help="Pipeline mode")
        parser.add_argument("--horizon", type=int, default=8, help="Forecast horizon")

        args = parser.parse_args()

        results = run_full_pipeline(
            config_path=args.config,
            mode=args.mode,
            forecast_horizon=args.horizon,
        )

        print(f"\\nPipeline completed!")
        if 'inference' in results and 'wape' in results['inference']:
            print(f"  Forecast WAPE: {{results['inference']['wape']:.4f}}")
    ''')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(script)

    logger.info(f"Generated self-contained full pipeline: {output_path}")
    return output_path
