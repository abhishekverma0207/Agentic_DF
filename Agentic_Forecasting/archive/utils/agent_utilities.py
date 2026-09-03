"""
Agent Utilities Module
======================
Pre-built utility functions for CrewAI agents to call instead of generating code.
Agents should import and use these functions, only writing custom code when
no utility exists for the required operation.

Usage in agent tasks:
    from utils.agent_utilities import (
        load_csv, save_csv, save_json, load_json,
        compute_wape, compute_wape_by_group, compute_metrics,
        create_lag_features, create_rolling_features, create_ewm_features,
        ...
    )

Categories:
1. Data I/O Utilities
2. Metrics Computation Utilities
3. Feature Engineering Utilities
4. Time Series Utilities
5. Clustering Utilities
6. Visualization Utilities
7. Categorical Encoding Utilities
8. Diagnostic Utilities
9. Smart Output Utilities
10. Context File Utilities
11. Source Data Loading (CSV, Parquet, Delta)
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# 1. DATA I/O UTILITIES
# =============================================================================

def _detect_data_format(file_path: str) -> str:
    """
    Detect data format from path.

    Returns one of: 'csv', 'parquet', 'delta'.

    Detection order:
    1. File extension (.csv, .parquet, .pq)
    2. Delta table (directory containing _delta_log/)
    3. Raise ValueError if unrecognized
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.csv':
        return 'csv'
    elif file_ext in ('.parquet', '.pq'):
        return 'parquet'
    elif os.path.isdir(file_path):
        # Check for delta table (directory with _delta_log/)
        if os.path.isdir(os.path.join(file_path, '_delta_log')):
            return 'delta'
        # Directory of parquet files (partitioned parquet)
        parquet_files = [f for f in os.listdir(file_path) if f.endswith('.parquet')]
        if parquet_files:
            return 'parquet'
        raise ValueError(
            f"Directory '{file_path}' is not a recognized data format. "
            f"Expected a delta table (_delta_log/) or partitioned parquet."
        )
    elif file_ext == '' and not os.path.exists(file_path):
        # Path without extension that doesn't exist yet — could be a delta table
        raise FileNotFoundError(f"Source data path not found: {file_path}")
    else:
        raise ValueError(
            f"Unsupported file format: '{file_ext}' for path '{file_path}'. "
            f"Supported: .csv, .parquet, .pq, or delta table directory."
        )


def load_source_data(
    file_path: str,
    columns: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load source data from CSV, Parquet, or Delta format (auto-detected).

    This is the primary function for loading input data in the pipeline.
    It automatically detects the file format and loads accordingly.

    Format detection:
    - .csv           → pandas read_csv
    - .parquet / .pq → pandas read_parquet
    - directory with _delta_log/ → deltalake DeltaTable.to_pandas()
    - directory with .parquet files → pandas read_parquet (partitioned)

    The file_path can be:
    - Relative (resolved by config before reaching here)
    - Absolute (any location on the filesystem)
    - Databricks Volumes path (/Volumes/...)

    Parameters
    ----------
    file_path : str
        Path to the data file or delta table directory
    columns : List[str], optional
        List of columns to load (loads all if None)
    dtype : Dict[str, Any], optional
        Column data types (only used for CSV)
    nrows : int, optional
        Number of rows to load (for sampling)

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame

    Raises
    ------
    FileNotFoundError
        If file/directory does not exist
    ValueError
        If file format is not supported

    Example
    -------
    >>> df = load_source_data('data.csv')
    >>> df = load_source_data('data.parquet')
    >>> df = load_source_data('/mnt/shared/data/sales.parquet')
    >>> df = load_source_data('/mnt/delta/sales_table')  # delta table
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source data file not found: {file_path}")

    fmt = _detect_data_format(file_path)

    if fmt == 'csv':
        logger.info(f"Loading CSV file: {file_path}")
        df = pd.read_csv(
            file_path,
            usecols=columns,
            dtype=dtype,
            nrows=nrows,
        )
    elif fmt == 'parquet':
        logger.info(f"Loading Parquet: {file_path}")
        if columns:
            df = pd.read_parquet(file_path, columns=columns)
        else:
            df = pd.read_parquet(file_path)
        # Apply nrows limit for parquet (sample)
        if nrows is not None and len(df) > nrows:
            df = df.head(nrows)
    elif fmt == 'delta':
        logger.info(f"Loading Delta table: {file_path}")
        try:
            from deltalake import DeltaTable
        except ImportError:
            raise ImportError(
                "The 'deltalake' package is required to read Delta tables. "
                "Install it with: pip install deltalake"
            )
        dt = DeltaTable(file_path)
        if columns:
            df = dt.to_pandas(columns=columns)
        else:
            df = dt.to_pandas()
        if nrows is not None and len(df) > nrows:
            df = df.head(nrows)

    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns from {file_path}")
    return df


def load_csv(
    file_path: str,
    columns: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
    parse_dates: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    **csv_kwargs,
) -> pd.DataFrame:
    """
    Load tabular data with optional column filtering and type specification.

    Despite the historical name (``load_csv``), this function is now
    format-aware: paths ending in ``.parquet`` or ``.pq`` are routed to
    ``pd.read_parquet`` so callers that received a parquet path don't
    silently crash.  This change is necessary because the DIQ runner now
    saves per-category checkpoints as parquet (much faster, smaller,
    dtype-preserving) but consumers like
    ``crews.feature_crew._run_deterministic_executor`` historically
    called ``load_csv(config.input_data_path)``.

    Extra ``**csv_kwargs`` (e.g. ``low_memory=False``) are forwarded to
    ``pd.read_csv`` for the CSV path and silently ignored for parquet
    (parquet is dtype-preserving so they have no meaning there).  This
    accommodates the dmh standalone-CLI path which passes
    ``low_memory=False`` against the per-category file.

    For parquet inputs, ``dtype`` and ``parse_dates`` are silently
    ignored — parquet preserves dtypes natively, so they're no-ops.

    Parameters
    ----------
    file_path : str
        Path to the data file (CSV, Parquet, or .pq)
    columns : List[str], optional
        List of columns to load (loads all if None)
    dtype : Dict[str, Any], optional
        Column data types (CSV only — ignored for parquet)
    parse_dates : List[str], optional
        Columns to parse as dates (CSV only — ignored for parquet)
    nrows : int, optional
        Number of rows to load (for sampling)
    **csv_kwargs
        Extra kwargs forwarded to ``pd.read_csv`` for CSV inputs;
        silently ignored for parquet.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame

    Example
    -------
    >>> df = load_csv('data.csv', columns=['key', 'date', 'sales'])
    >>> df = load_csv('data.parquet')  # also works
    >>> df = load_csv('data.csv', low_memory=False)  # forwarded to read_csv
    >>> df = load_csv('data.parquet', low_memory=False)  # ignored, no error
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Route by suffix — parquet support is required because DIQ
    # checkpoints are now parquet by default.
    fp_lower = file_path.lower()
    if fp_lower.endswith((".parquet", ".pq")):
        df = pd.read_parquet(file_path, columns=columns)
        if nrows is not None and len(df) > nrows:
            df = df.head(nrows)
        return df

    df = pd.read_csv(
        file_path,
        usecols=columns,
        dtype=dtype,
        parse_dates=parse_dates,
        nrows=nrows,
        **csv_kwargs,
    )

    return df


def save_csv(
    df: pd.DataFrame,
    file_path: str,
    index: bool = False,
    float_format: str = '%.6f',
) -> str:
    """
    Save DataFrame to CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    file_path : str
        Output file path
    index : bool
        Whether to include index
    float_format : str
        Format for floating point numbers

    Returns
    -------
    str
        Path to saved file

    Example
    -------
    >>> save_csv(result_df, 'output/results.csv')
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True) if os.path.dirname(file_path) else None
    df.to_csv(file_path, index=index, float_format=float_format)
    return file_path


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file.

    Parameters
    ----------
    file_path : str
        Path to JSON file

    Returns
    -------
    Dict[str, Any]
        Loaded JSON data

    Example
    -------
    >>> config = load_json('config/settings.json')
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(
    data: Dict[str, Any],
    file_path: str,
    indent: int = 2,
) -> str:
    """
    Save dictionary to JSON file.

    Parameters
    ----------
    data : Dict[str, Any]
        Data to save
    file_path : str
        Output file path
    indent : int
        JSON indentation

    Returns
    -------
    str
        Path to saved file

    Example
    -------
    >>> save_json({'key': 'value'}, 'output/config.json')
    """
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Handle numpy/pandas types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif hasattr(obj, '__dict__'):
            # For objects with __dict__, convert to dict representation
            return str(obj)
        return str(obj)  # Fallback: convert to string

    # Clean data to avoid circular references
    def clean_for_json(obj, seen=None):
        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return "<circular_reference>"

        if isinstance(obj, dict):
            seen.add(obj_id)
            return {k: clean_for_json(v, seen.copy()) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            seen.add(obj_id)
            return [clean_for_json(item, seen.copy()) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        else:
            return obj

    cleaned_data = clean_for_json(data)

    with open(file_path, 'w') as f:
        json.dump(cleaned_data, f, indent=indent, default=convert)

    return file_path


def merge_dataframes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Union[str, List[str]],
    how: str = 'left',
    suffixes: Tuple[str, str] = ('', '_y'),
) -> pd.DataFrame:
    """
    Merge two DataFrames on specified columns.

    Parameters
    ----------
    left : pd.DataFrame
        Left DataFrame
    right : pd.DataFrame
        Right DataFrame
    on : str or List[str]
        Column(s) to merge on
    how : str
        Type of merge ('left', 'right', 'inner', 'outer')
    suffixes : Tuple[str, str]
        Suffixes for overlapping columns

    Returns
    -------
    pd.DataFrame
        Merged DataFrame

    Example
    -------
    >>> merged = merge_dataframes(sales_df, segment_df, on='key')
    """
    return pd.merge(left, right, on=on, how=how, suffixes=suffixes)


# =============================================================================
# 2. METRICS COMPUTATION UTILITIES
# =============================================================================

def compute_wape(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute Weighted Absolute Percentage Error (WAPE).

    WAPE = sum(|actual - predicted|) / sum(|actual|)

    Parameters
    ----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        WAPE value (0 to inf, lower is better)

    Example
    -------
    >>> wape = compute_wape(df['actual'], df['predicted'])
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    abs_error = np.sum(np.abs(y_true - y_pred))
    total_actual = np.sum(np.abs(y_true))

    return abs_error / (total_actual + 1e-10)


def compute_mae(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        MAE value

    Example
    -------
    >>> mae = compute_mae(df['actual'], df['predicted'])
    """
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def compute_rmse(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        RMSE value

    Example
    -------
    >>> rmse = compute_rmse(df['actual'], df['predicted'])
    """
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def compute_bias(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute Bias (mean error, positive = over-predicting).

    Parameters
    ----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        Bias value

    Example
    -------
    >>> bias = compute_bias(df['actual'], df['predicted'])
    """
    return np.mean(np.array(y_pred) - np.array(y_true))


def compute_smape(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error (SMAPE).

    Parameters
    ----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        SMAPE value (0 to 2, lower is better)

    Example
    -------
    >>> smape = compute_smape(df['actual'], df['predicted'])
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = np.abs(y_true) + np.abs(y_pred)
    smape = np.mean(2 * np.abs(y_true - y_pred) / (denominator + 1e-10))

    return smape


def compute_all_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Compute all standard metrics at once.

    Parameters
    ----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values

    Returns
    -------
    Dict[str, float]
        Dictionary with all metrics

    Example
    -------
    >>> metrics = compute_all_metrics(df['actual'], df['predicted'])
    >>> print(f"WAPE: {metrics['wape']:.2%}")
    """
    return {
        'wape': compute_wape(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'rmse': compute_rmse(y_true, y_pred),
        'bias': compute_bias(y_true, y_pred),
        'smape': compute_smape(y_true, y_pred),
        'n_obs': len(y_true),
        'sum_actual': float(np.sum(y_true)),
        'sum_predicted': float(np.sum(y_pred)),
    }


def compute_wape_by_group(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    actual_col: str,
    pred_col: str,
) -> pd.DataFrame:
    """
    Compute WAPE aggregated by group(s).

    CRITICAL: Aggregates errors FIRST, then computes WAPE.
    This is the correct method for grouped WAPE.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    group_cols : str or List[str]
        Column(s) to group by
    actual_col : str
        Name of actual values column
    pred_col : str
        Name of predicted values column

    Returns
    -------
    pd.DataFrame
        DataFrame with WAPE per group

    Example
    -------
    >>> wape_by_segment = compute_wape_by_group(df, 'segment_id', 'actual', 'predicted')
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Add absolute error column
    df = df.copy()
    df['_abs_error'] = np.abs(df[actual_col] - df[pred_col])

    # Aggregate by group
    agg = df.groupby(group_cols).agg({
        '_abs_error': 'sum',
        actual_col: 'sum',
        pred_col: 'sum',
    }).reset_index()

    # Compute WAPE
    agg['wape'] = agg['_abs_error'] / (np.abs(agg[actual_col]) + 1e-10)
    agg['n_obs'] = df.groupby(group_cols).size().values

    # Clean up
    agg = agg.rename(columns={
        '_abs_error': 'sum_abs_error',
        actual_col: 'sum_actual',
        pred_col: 'sum_predicted',
    })

    return agg


def compute_metrics_at_level(
    df: pd.DataFrame,
    error_level_cols: List[str],
    actual_col: str,
    pred_col: str,
) -> pd.DataFrame:
    """
    Compute absolute error at specified aggregation level.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    error_level_cols : List[str]
        Columns defining the aggregation level
    actual_col : str
        Name of actual values column
    pred_col : str
        Name of predicted values column

    Returns
    -------
    pd.DataFrame
        DataFrame with errors computed at specified level

    Example
    -------
    >>> errors = compute_metrics_at_level(df, ['key', 'year_week'], 'actual', 'predicted')
    """
    agg = df.groupby(error_level_cols).agg({
        actual_col: 'sum',
        pred_col: 'sum',
    }).reset_index()

    agg['abs_error'] = np.abs(agg[actual_col] - agg[pred_col])

    return agg


# =============================================================================
# 3. FEATURE ENGINEERING UTILITIES
# =============================================================================

def create_lag_features(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str,
    lags: List[int] = [1, 2, 4, 8, 13, 26, 52],
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create lag features for time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (must be sorted by date within groups)
    group_cols : List[str]
        Columns to group by (e.g., key columns)
    target_col : str
        Column to create lags for
    lags : List[int]
        List of lag periods
    prefix : str, optional
        Prefix for lag column names (defaults to target_col)

    Returns
    -------
    pd.DataFrame
        DataFrame with lag features added

    Example
    -------
    >>> df = create_lag_features(df, ['key'], 'sales', lags=[1, 4, 52])
    """
    df = df.copy()
    prefix = prefix or target_col

    for lag in lags:
        col_name = f'{prefix}_lag_{lag}'
        df[col_name] = df.groupby(group_cols)[target_col].shift(lag)

    return df


def create_rolling_features(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str,
    windows: List[int] = [4, 8, 13, 26, 52],
    aggregations: List[str] = ['mean', 'std', 'min', 'max'],
    shift_periods: int = 1,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create rolling window features for time series.

    IMPORTANT: Uses shift(shift_periods) to prevent data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (must be sorted by date within groups)
    group_cols : List[str]
        Columns to group by
    target_col : str
        Column to compute rolling features for
    windows : List[int]
        List of window sizes
    aggregations : List[str]
        List of aggregation functions ('mean', 'std', 'min', 'max', 'sum')
    shift_periods : int
        Periods to shift before rolling (default 1 to prevent leakage)
    prefix : str, optional
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling features added

    Example
    -------
    >>> df = create_rolling_features(df, ['key'], 'sales', windows=[4, 13])
    """
    df = df.copy()
    prefix = prefix or target_col

    for window in windows:
        for agg in aggregations:
            col_name = f'{prefix}_roll_{window}_{agg}'
            df[col_name] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.shift(shift_periods).rolling(window, min_periods=1).agg(agg)
            )

    return df


def create_ewm_features(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str,
    spans: List[int] = [4, 13, 26],
    shift_periods: int = 1,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create Exponential Weighted Mean features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : List[str]
        Columns to group by
    target_col : str
        Column to compute EWM for
    spans : List[int]
        List of EWM spans
    shift_periods : int
        Periods to shift before EWM
    prefix : str, optional
        Prefix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with EWM features added

    Example
    -------
    >>> df = create_ewm_features(df, ['key'], 'sales', spans=[4, 13])
    """
    df = df.copy()
    prefix = prefix or target_col

    for span in spans:
        col_name = f'{prefix}_ewm_{span}'
        df[col_name] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(shift_periods).ewm(span=span, min_periods=1).mean()
        )

    return df


def create_cyclical_features(
    df: pd.DataFrame,
    week_col: str = 'week_of_year',
    month_col: Optional[str] = None,
    max_week: int = 52,
    max_month: int = 12,
) -> pd.DataFrame:
    """
    Create cyclical (sin/cos) encoding for calendar features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    week_col : str
        Week of year column name
    month_col : str, optional
        Month column name
    max_week : int
        Maximum week value (52 or 53)
    max_month : int
        Maximum month value (12)

    Returns
    -------
    pd.DataFrame
        DataFrame with cyclical features added

    Example
    -------
    >>> df = create_cyclical_features(df, week_col='week_of_year')
    """
    df = df.copy()

    if week_col in df.columns:
        df[f'{week_col}_sin'] = np.sin(2 * np.pi * df[week_col] / max_week)
        df[f'{week_col}_cos'] = np.cos(2 * np.pi * df[week_col] / max_week)

    if month_col and month_col in df.columns:
        df[f'{month_col}_sin'] = np.sin(2 * np.pi * df[month_col] / max_month)
        df[f'{month_col}_cos'] = np.cos(2 * np.pi * df[month_col] / max_month)

    return df


def create_calendar_features(
    df: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    """
    Create calendar features from a date column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_col : str
        Date column name (must be datetime or convertible)

    Returns
    -------
    pd.DataFrame
        DataFrame with calendar features added

    Example
    -------
    >>> df = create_calendar_features(df, 'date')
    """
    df = df.copy()

    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        sample_val = str(df[date_col].iloc[0])
        str_len = len(sample_val)

        # Try to parse year_week format (YYYYWW - 6 chars)
        if str_len == 6 and sample_val.isdigit():
            df['_year'] = df[date_col].astype(str).str[:4].astype(int)
            df['_week'] = df[date_col].astype(str).str[4:].astype(int)
            df['_date'] = pd.to_datetime(
                df['_year'].astype(str) + df['_week'].astype(str).str.zfill(2) + '1',
                format='%Y%W%w'
            )
            df = df.drop(columns=['_year', '_week'])
            date_col = '_date'
        # Try to parse year-week format (YYYY-WW - 7 chars with hyphen)
        elif str_len == 7 and '-' in sample_val:
            parts = df[date_col].astype(str).str.split('-', expand=True)
            df['_year'] = parts[0].astype(int)
            df['_week'] = parts[1].astype(int)
            # Create date from ISO week: year + week + day 1 (Monday)
            df['_date'] = pd.to_datetime(
                df['_year'].astype(str) + df['_week'].astype(str).str.zfill(2) + '1',
                format='%G%V%u'  # ISO year, ISO week, weekday (1=Monday)
            )
            df = df.drop(columns=['_year', '_week'])
            date_col = '_date'
        else:
            df[date_col] = pd.to_datetime(df[date_col])

    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['week_of_year'] = df[date_col].dt.isocalendar().week.astype(int)
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['quarter'] = df[date_col].dt.quarter
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[date_col].dt.is_year_start.astype(int)
    df['is_year_end'] = df[date_col].dt.is_year_end.astype(int)

    # Clean up temporary column if created
    if '_date' in df.columns and date_col == '_date':
        df = df.drop(columns=['_date'])

    return df


def create_intermittency_features(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str,
    windows: List[int] = [4, 13, 26],
) -> pd.DataFrame:
    """
    Create features for intermittent demand modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : List[str]
        Columns to group by
    target_col : str
        Target column
    windows : List[int]
        Windows for demand frequency calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with intermittency features added

    Example
    -------
    >>> df = create_intermittency_features(df, ['key'], 'sales')
    """
    df = df.copy()

    # Binary demand occurred
    df['demand_occurred'] = (df[target_col] > 0).astype(int)

    # Periods since last non-zero demand
    def periods_since_nonzero(x):
        result = []
        count = 0
        for val in x:
            if val > 0:
                result.append(count)
                count = 0
            else:
                result.append(count)
                count += 1
        return result

    df['periods_since_nonzero'] = df.groupby(group_cols)[target_col].transform(
        lambda x: periods_since_nonzero(x.values)
    )

    # Demand frequency in windows
    for window in windows:
        col_name = f'demand_frequency_{window}w'
        df[col_name] = df.groupby(group_cols)['demand_occurred'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # Average non-zero demand (rolling)
    df['avg_nonzero_demand'] = df.groupby(group_cols)[target_col].transform(
        lambda x: x.shift(1).where(x.shift(1) > 0).expanding().mean()
    )

    return df


def create_interaction_features(
    df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    interaction_type: str = 'sum',
) -> pd.DataFrame:
    """
    Create interaction features from feature groups.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_groups : Dict[str, List[str]]
        Dictionary mapping group names to list of column names
        e.g., {'promo': ['promo_flag', 'display_flag'], 'price': ['discount_flag']}
    interaction_type : str
        Type of interaction: 'sum', 'product', 'any'

    Returns
    -------
    pd.DataFrame
        DataFrame with interaction features added

    Example
    -------
    >>> df = create_interaction_features(df, {'promo': ['promo', 'display']}, 'sum')
    """
    df = df.copy()

    for group_name, columns in feature_groups.items():
        existing_cols = [c for c in columns if c in df.columns]

        if not existing_cols:
            continue

        if interaction_type == 'sum':
            df[f'{group_name}_intensity'] = df[existing_cols].fillna(0).sum(axis=1)
            df[f'{group_name}_any'] = (df[f'{group_name}_intensity'] > 0).astype(int)

        elif interaction_type == 'product' and len(existing_cols) >= 2:
            df[f'{group_name}_product'] = df[existing_cols[0]] * df[existing_cols[1]]

        elif interaction_type == 'any':
            df[f'{group_name}_any'] = (df[existing_cols].fillna(0).sum(axis=1) > 0).astype(int)

    return df


# =============================================================================
# 4. TIME SERIES UTILITIES
# =============================================================================

# Syntetos-Boylan classification thresholds (EXACT)
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49


def compute_demand_characteristics(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str,
    compute_advanced: bool = True,
    include_external: bool = True,
    price_features_numeric: Optional[List[str]] = None,
    price_features_categorical: Optional[List[str]] = None,
    promo_features_numeric: Optional[List[str]] = None,
    promo_features_categorical: Optional[List[str]] = None,
    min_variance_threshold: float = 1e-6,
    min_unique_ratio: float = 0.01,
) -> pd.DataFrame:
    """
    Compute COMPREHENSIVE per-key demand characteristics for world-class segmentation.

    This function computes ALL metrics needed for multi-dimensional segmentation:
    - Basic statistics (mean, std, cv, etc.)
    - Intermittency metrics (ADI, CV², zero_fraction)
    - Volume metrics (total_demand, demand_frequency)
    - Shape metrics (skewness, kurtosis)
    - Temporal metrics (autocorrelation, trend_strength, seasonal_strength)
    - External feature aggregates (price, promo, etc.) for price-sensitivity segmentation

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : List[str]
        Key columns
    target_col : str
        Target column
    compute_advanced : bool
        If True, compute advanced metrics (autocorr, trend, seasonality). Default True.
    include_external : bool
        If True, compute external feature aggregates (price, promo). Default True.
    price_features_numeric : List[str], optional
        List of numeric price feature columns from config
    price_features_categorical : List[str], optional
        List of categorical price feature columns from config
    promo_features_numeric : List[str], optional
        List of numeric promo feature columns from config
    promo_features_categorical : List[str], optional
        List of categorical promo feature columns from config
    min_variance_threshold : float
        Minimum variance threshold for including a feature (default 1e-6)
    min_unique_ratio : float
        Minimum ratio of unique values to total rows for including a feature (default 0.01)

    Returns
    -------
    pd.DataFrame
        DataFrame with COMPREHENSIVE demand characteristics per key

    Example
    -------
    >>> chars = compute_demand_characteristics(df, ['key'], 'sales',
    ...     price_features_numeric=['unit_price', 'list_price'],
    ...     promo_features_categorical=['promo_type', 'promo_mechanic'])
    >>> print(chars.columns)  # Many columns for multi-dimensional segmentation
    """
    grouped = df.groupby(group_cols)[target_col]

    # -------------------------------------------------------------------------
    # BASIC STATISTICS
    # -------------------------------------------------------------------------
    chars = pd.DataFrame({
        'n_obs': grouped.count(),
        'mean': grouped.mean(),
        'std': grouped.std(),
        'min': grouped.min(),
        'max': grouped.max(),
        'sum': grouped.sum(),
        'median': grouped.median(),
        'zero_count': grouped.apply(lambda x: (x == 0).sum()),
    })

    # -------------------------------------------------------------------------
    # DERIVED BASIC METRICS
    # -------------------------------------------------------------------------
    chars['cv'] = chars['std'] / (chars['mean'].abs() + 1e-10)
    chars['cv2'] = chars['cv'] ** 2
    chars['zero_fraction'] = chars['zero_count'] / chars['n_obs']
    chars['demand_frequency'] = 1 - chars['zero_fraction']  # Non-zero ratio
    chars['range'] = chars['max'] - chars['min']
    chars['range_normalized'] = chars['range'] / (chars['mean'].abs() + 1e-10)

    # -------------------------------------------------------------------------
    # ADI (Average Demand Interval) - Syntetos-Boylan
    # -------------------------------------------------------------------------
    def compute_adi(series):
        arr = series.values
        non_zero_idx = np.where(arr > 0)[0]
        if len(non_zero_idx) <= 1:
            return len(arr)
        return np.mean(np.diff(non_zero_idx))

    chars['adi'] = grouped.apply(compute_adi)

    # -------------------------------------------------------------------------
    # DISCRETE DEMAND DETECTION (for low-cardinality target values)
    # -------------------------------------------------------------------------
    chars['n_unique'] = grouped.nunique()
    chars['unique_ratio'] = chars['n_unique'] / chars['n_obs']
    # A key is discrete if it has very few unique values relative to observations,
    # OR has 25 or fewer unique values total (case-pack ordering, etc.)
    chars['is_discrete'] = (chars['unique_ratio'] < 0.05) | (chars['n_unique'] <= 25)

    # -------------------------------------------------------------------------
    # SHAPE STATISTICS (for distribution-based segmentation)
    # -------------------------------------------------------------------------
    def safe_skew(series):
        try:
            if len(series.dropna()) > 3:
                return float(series.skew())
        except Exception:
            pass
        return 0.0

    def safe_kurtosis(series):
        try:
            if len(series.dropna()) > 3:
                return float(series.kurtosis())
        except Exception:
            pass
        return 0.0

    chars['skewness'] = grouped.apply(safe_skew)
    chars['kurtosis'] = grouped.apply(safe_kurtosis)

    # -------------------------------------------------------------------------
    # VOLUME TIER (for business-priority segmentation)
    # -------------------------------------------------------------------------
    # Compute after we have all series metrics
    high_threshold = chars['mean'].quantile(0.70)
    low_threshold = chars['mean'].quantile(0.30)

    def assign_volume_tier(mean_val):
        if mean_val >= high_threshold:
            return 'high'
        elif mean_val <= low_threshold:
            return 'low'
        return 'medium'

    chars['volume_tier'] = chars['mean'].apply(assign_volume_tier)

    # -------------------------------------------------------------------------
    # VARIABILITY TIER
    # -------------------------------------------------------------------------
    def assign_variability_tier(cv_val):
        if cv_val < 0.5:
            return 'stable'
        elif cv_val > 1.5:
            return 'variable'
        return 'moderate'

    chars['variability_tier'] = chars['cv'].apply(assign_variability_tier)

    # -------------------------------------------------------------------------
    # ADVANCED TEMPORAL METRICS (if enabled)
    # -------------------------------------------------------------------------
    if compute_advanced:
        def compute_autocorr_lag1(series):
            """Compute lag-1 autocorrelation."""
            try:
                series = series.dropna()
                if len(series) > 5:
                    return float(series.autocorr(lag=1))
            except Exception:
                pass
            return 0.0

        def compute_trend_strength(series):
            """
            Compute trend strength using simple linear regression R².
            Range: 0 (no trend) to 1 (strong trend).
            """
            try:
                series = series.dropna().values
                if len(series) > 10:
                    x = np.arange(len(series))
                    # Fit linear regression
                    slope, intercept = np.polyfit(x, series, 1)
                    trend = slope * x + intercept
                    residuals = series - trend

                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((series - np.mean(series)) ** 2)

                    if ss_tot > 0:
                        r2 = 1 - (ss_res / ss_tot)
                        return float(max(0, min(1, r2)))
            except Exception:
                pass
            return 0.0

        def compute_seasonal_strength(series, period=52):
            """
            Compute seasonal strength using STL-like decomposition.
            Range: 0 (no seasonality) to 1 (strong seasonality).
            """
            try:
                series = series.dropna().values
                if len(series) >= 2 * period:
                    # Simple seasonal strength: variance of seasonal means / total variance
                    n_complete_seasons = len(series) // period
                    if n_complete_seasons >= 2:
                        seasonal_vals = []
                        for i in range(period):
                            idx = list(range(i, len(series), period))
                            if idx:
                                seasonal_vals.append(np.mean([series[j] for j in idx if j < len(series)]))

                        total_var = np.var(series)
                        seasonal_var = np.var(seasonal_vals) if len(seasonal_vals) > 1 else 0

                        if total_var > 0:
                            strength = seasonal_var / total_var
                            return float(max(0, min(1, strength)))
            except Exception:
                pass
            return 0.0

        chars['autocorr_lag1'] = grouped.apply(compute_autocorr_lag1)
        chars['trend_strength'] = grouped.apply(compute_trend_strength)
        chars['seasonal_strength'] = grouped.apply(compute_seasonal_strength)

        # Forecastability score (composite)
        # Higher = easier to forecast
        chars['forecastability_score'] = (
            chars['demand_frequency'] * 0.3 +  # More non-zero = easier
            (1 - chars['cv'].clip(0, 2) / 2) * 0.3 +  # Lower CV = easier
            chars['autocorr_lag1'].abs() * 0.2 +  # Higher autocorr = more predictable
            chars['trend_strength'] * 0.1 +  # Some trend = easier
            chars['seasonal_strength'] * 0.1  # Seasonality = pattern to capture
        ).clip(0, 1)

    # -------------------------------------------------------------------------
    # EXTERNAL FEATURE AGGREGATES (Price, Promo, etc.)
    # These are critical for identifying price-sensitive vs base-demand segments
    # USES CONFIG-DEFINED FEATURE LISTS (not pattern detection)
    # -------------------------------------------------------------------------
    if include_external:
        # Helper function to validate feature before including
        def _validate_feature(col: str, df: pd.DataFrame, is_numeric: bool = True) -> bool:
            """Check if feature has enough variance/diversity to be useful."""
            if col not in df.columns:
                logger.debug(f"Feature '{col}' not found in DataFrame, skipping")
                return False

            col_data = df[col].dropna()
            if len(col_data) == 0:
                logger.debug(f"Feature '{col}' has no valid data, skipping")
                return False

            if is_numeric:
                # Check variance for numeric features
                try:
                    variance = col_data.var()
                    if variance < min_variance_threshold:
                        logger.debug(f"Feature '{col}' has low variance ({variance:.2e}), skipping")
                        return False
                except Exception:
                    return False
            else:
                # Check unique ratio for categorical features
                n_unique = col_data.nunique()
                unique_ratio = n_unique / len(col_data)
                if n_unique <= 1:
                    logger.debug(f"Feature '{col}' has only {n_unique} unique value(s), skipping")
                    return False

            return True

        # ---------------------------------------------------------------------
        # NUMERIC EXTERNAL FEATURES (price, promo amounts, etc.) from config
        # ---------------------------------------------------------------------
        numeric_features_config = {
            'price': price_features_numeric or [],
            'promo': promo_features_numeric or [],
        }

        for feature_type, feature_cols in numeric_features_config.items():
            # Filter to valid features that exist and have sufficient variance
            valid_cols = [col for col in feature_cols if _validate_feature(col, df, is_numeric=True)]

            if valid_cols:
                logger.info(f"Processing {len(valid_cols)} valid {feature_type} numeric features for segmentation")

                for col in valid_cols:
                    try:
                        ext_grouped = df.groupby(group_cols)[col]
                        col_prefix = f"{feature_type}_{col}" if len(valid_cols) > 1 else feature_type

                        # Compute aggregates
                        chars[f'{col_prefix}_mean'] = ext_grouped.mean()
                        chars[f'{col_prefix}_std'] = ext_grouped.std()
                        chars[f'{col_prefix}_cv'] = (
                            chars[f'{col_prefix}_std'] /
                            (chars[f'{col_prefix}_mean'].abs() + 1e-10)
                        ).clip(0, 10)
                        chars[f'{col_prefix}_min'] = ext_grouped.min()
                        chars[f'{col_prefix}_max'] = ext_grouped.max()

                        # For promo, also compute frequency (non-zero ratio)
                        if feature_type == 'promo':
                            chars[f'{col_prefix}_frequency'] = ext_grouped.apply(
                                lambda x: (x > 0).mean() if len(x) > 0 else 0
                            )

                        logger.debug(f"Added {feature_type} aggregates from column '{col}'")
                    except Exception as e:
                        logger.debug(f"Could not compute {feature_type} aggregates for '{col}': {e}")

        # ---------------------------------------------------------------------
        # CATEGORICAL EXTERNAL FEATURES from config
        # These need frequency encoding and aggregation for use in clustering
        # ---------------------------------------------------------------------
        categorical_features_config = {
            'price': price_features_categorical or [],
            'promo': promo_features_categorical or [],
        }

        categorical_encoding_results = {}  # Store encoding maps for downstream use

        for feature_type, feature_cols in categorical_features_config.items():
            # Filter to valid categorical features
            valid_cols = [col for col in feature_cols if _validate_feature(col, df, is_numeric=False)]

            if valid_cols:
                logger.info(f"Processing {len(valid_cols)} valid {feature_type} categorical features for segmentation")

                for col in valid_cols:
                    try:
                        # Compute frequency encoding for the categorical column
                        # This gives each category a value based on its prevalence
                        freq_map = df[col].value_counts(normalize=True).to_dict()
                        df_encoded_col = df[col].map(freq_map).fillna(0)

                        # Store the encoding map
                        categorical_encoding_results[col] = freq_map

                        # Create clean column name for aggregation
                        col_clean = col.replace('.', '_').replace(' ', '_')
                        col_prefix = f"{feature_type}_{col_clean}"

                        # Aggregate encoded values per key
                        cat_grouped = df.assign(**{f'{col}_encoded': df_encoded_col}).groupby(group_cols)

                        # Mean: represents average category frequency for this key
                        chars[f'{col_prefix}_encoded_mean'] = cat_grouped[f'{col}_encoded'].mean()
                        # Max: represents highest frequency category used by this key
                        chars[f'{col_prefix}_encoded_max'] = cat_grouped[f'{col}_encoded'].max()

                        # Dominant category as a string (mode)
                        def get_mode_safe(x):
                            mode_result = x.mode()
                            return mode_result.iloc[0] if len(mode_result) > 0 else None

                        chars[f'{col_prefix}_dominant'] = df.groupby(group_cols)[col].agg(get_mode_safe)

                        # Number of unique categories per key (diversity measure)
                        chars[f'{col_prefix}_diversity'] = df.groupby(group_cols)[col].nunique()

                        logger.debug(f"Added categorical {feature_type} features from column '{col}'")
                    except Exception as e:
                        logger.debug(f"Could not encode categorical {feature_type} feature '{col}': {e}")

        # Store encoding maps in chars.attrs for downstream use
        if categorical_encoding_results:
            try:
                chars.attrs['categorical_encoding_maps'] = categorical_encoding_results
                logger.info(f"Encoded {len(categorical_encoding_results)} categorical external feature columns for segmentation")
            except Exception:
                pass  # attrs not supported in all pandas versions

    return chars.reset_index()


def classify_demand_pattern(
    adi: float,
    cv2: float,
    adi_threshold: float = ADI_THRESHOLD,
    cv2_threshold: float = CV2_THRESHOLD,
) -> str:
    """
    Classify demand pattern using Syntetos-Boylan framework.

    Parameters
    ----------
    adi : float
        Average Demand Interval
    cv2 : float
        Squared Coefficient of Variation
    adi_threshold : float
        ADI threshold (default 1.32)
    cv2_threshold : float
        CV² threshold (default 0.49)

    Returns
    -------
    str
        Demand pattern: 'smooth', 'erratic', 'intermittent', or 'lumpy'

    Example
    -------
    >>> pattern = classify_demand_pattern(1.5, 0.6)
    >>> print(pattern)  # 'lumpy'
    """
    if adi > adi_threshold:
        return 'lumpy' if cv2 > cv2_threshold else 'intermittent'
    else:
        return 'erratic' if cv2 > cv2_threshold else 'smooth'


def add_demand_classification(
    chars_df: pd.DataFrame,
    adi_col: str = 'adi',
    cv2_col: str = 'cv2',
) -> pd.DataFrame:
    """
    Add demand classification to characteristics DataFrame.

    Parameters
    ----------
    chars_df : pd.DataFrame
        DataFrame with ADI and CV² columns
    adi_col : str
        ADI column name
    cv2_col : str
        CV² column name

    Returns
    -------
    pd.DataFrame
        DataFrame with 'intermittency_class' and 'demand_pattern' columns added

    Example
    -------
    >>> chars = add_demand_classification(chars)
    """
    chars_df = chars_df.copy()

    # Compute the Syntetos-Boylan demand pattern classification
    pattern = chars_df.apply(
        lambda r: classify_demand_pattern(r[adi_col], r[cv2_col]),
        axis=1
    )

    # Add BOTH column names for maximum compatibility
    # - 'intermittency_class' is the technical/original name
    # - 'demand_pattern' is used by downstream code (segmentation, feature engineering)
    chars_df['intermittency_class'] = pattern
    chars_df['demand_pattern'] = pattern

    return chars_df


def test_stationarity(
    series: pd.Series,
    adf_significance: float = 0.05,
    kpss_significance: float = 0.05,
) -> Dict[str, Any]:
    """
    Test time series stationarity using ADF and KPSS tests.

    Parameters
    ----------
    series : pd.Series
        Time series to test
    adf_significance : float
        ADF significance level
    kpss_significance : float
        KPSS significance level

    Returns
    -------
    Dict[str, Any]
        Dictionary with test results and stationarity status

    Example
    -------
    >>> result = test_stationarity(df['sales'])
    >>> print(result['status'])
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    # Drop NaN values
    series = series.dropna()

    if len(series) < 10:
        return {
            'status': 'INSUFFICIENT_DATA',
            'adf_statistic': None,
            'adf_pvalue': None,
            'kpss_statistic': None,
            'kpss_pvalue': None,
        }

    try:
        adf_result = adfuller(series, autolag='AIC')
        adf_stat, adf_p = adf_result[0], adf_result[1]
    except Exception:
        adf_stat, adf_p = None, 1.0

    try:
        kpss_result = kpss(series, regression='c', nlags='auto')
        kpss_stat, kpss_p = kpss_result[0], kpss_result[1]
    except Exception:
        kpss_stat, kpss_p = None, 0.0

    # Determine stationarity status
    adf_stationary = adf_p < adf_significance if adf_p else False
    kpss_stationary = kpss_p > kpss_significance if kpss_p else False

    if adf_stationary and kpss_stationary:
        status = 'STATIONARY'
    elif not adf_stationary and not kpss_stationary:
        status = 'NON_STATIONARY'
    elif adf_stationary and not kpss_stationary:
        status = 'TREND_STATIONARY'
    else:
        status = 'DIFFERENCE_STATIONARY'

    return {
        'status': status,
        'adf_statistic': adf_stat,
        'adf_pvalue': adf_p,
        'kpss_statistic': kpss_stat,
        'kpss_pvalue': kpss_p,
        'is_stationary': status == 'STATIONARY',
    }


# =============================================================================
# 5. CLUSTERING UTILITIES
# =============================================================================

def scale_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler_type: str = 'standard',
) -> Tuple[np.ndarray, Any]:
    """
    Scale features for clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_cols : List[str]
        Columns to scale
    scaler_type : str
        Type of scaler: 'standard', 'minmax', 'robust'

    Returns
    -------
    Tuple[np.ndarray, Any]
        Scaled features array and fitted scaler object

    Example
    -------
    >>> X_scaled, scaler = scale_features(df, ['cv', 'mean', 'zero_fraction'])
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    scalers = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler,
    }

    scaler_class = scalers.get(scaler_type, StandardScaler)
    scaler = scaler_class()

    X = df[feature_cols].fillna(0).values
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def cluster_with_hdbscan(
    X_scaled: np.ndarray,
    min_pct: float = 0.10,
    min_samples: Optional[int] = None,
) -> Tuple[np.ndarray, Any]:
    """
    Cluster data using HDBSCAN with minimum cluster size enforcement.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix
    min_pct : float
        Minimum cluster size as percentage of data (default 10%)
    min_samples : int, optional
        HDBSCAN min_samples parameter

    Returns
    -------
    Tuple[np.ndarray, Any]
        Cluster labels and fitted HDBSCAN object

    Example
    -------
    >>> labels, clusterer = cluster_with_hdbscan(X_scaled, min_pct=0.10)
    """
    from hdbscan import HDBSCAN
    from sklearn.neighbors import NearestNeighbors

    n_samples = len(X_scaled)
    min_cluster_size = max(int(n_samples * min_pct), 5)

    if min_samples is None:
        min_samples = max(5, int(n_samples * 0.01))

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom',
        metric='euclidean',
    )

    labels = clusterer.fit_predict(X_scaled)

    # Handle noise points (-1)
    noise_mask = labels == -1
    n_noise = noise_mask.sum()

    if n_noise > 0 and n_noise < n_samples:
        valid_labels = [l for l in set(labels) if l >= 0]
        if valid_labels:
            centroids = np.array([
                X_scaled[labels == l].mean(axis=0) for l in valid_labels
            ])

            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(centroids)
            noise_points = X_scaled[noise_mask]
            _, nearest_idx = nn.kneighbors(noise_points)

            labels[noise_mask] = np.array(valid_labels)[nearest_idx.flatten()]

    # Renumber to consecutive integers
    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])

    return labels, clusterer


def cluster_with_kmeans(
    X_scaled: np.ndarray,
    k_range: List[int] = [2, 3, 4, 5],
    random_state: int = 42,
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Cluster data using K-Means with silhouette-based selection.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix
    k_range : List[int]
        Range of k values to try
    random_state : int
        Random state for reproducibility

    Returns
    -------
    Tuple[np.ndarray, Any, Dict[str, Any]]
        Cluster labels, fitted KMeans object, and selection results

    Example
    -------
    >>> labels, kmeans, results = cluster_with_kmeans(X_scaled, k_range=[2, 3, 4, 5])
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        if len(set(labels)) > 1:
            sil = silhouette_score(X_scaled, labels)
            results.append({
                'k': k,
                'silhouette': sil,
                'inertia': kmeans.inertia_,
                'model': kmeans,
            })

    if not results:
        # Fallback to k=2
        kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        return labels, kmeans, {'k': 2, 'silhouette': 0.0}

    best = max(results, key=lambda x: x['silhouette'])

    return best['model'].labels_, best['model'], {
        'optimal_k': best['k'],
        'silhouette': best['silhouette'],
        'all_results': results,
    }


def compute_segment_profiles(
    df: pd.DataFrame,
    segment_col: str,
    feature_cols: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute segment profiles with relative indices.

    Relative Index = segment_mean / overall_mean
    - > 1.0: segment is above average
    - < 1.0: segment is below average
    - = 1.0: segment matches average

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with segment assignments
    segment_col : str
        Segment column name
    feature_cols : List[str]
        Feature columns to profile

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary with profile for each segment

    Example
    -------
    >>> profiles = compute_segment_profiles(df, 'segment_id', ['cv', 'mean'])
    """
    overall_means = df[feature_cols].mean()
    profiles = {}

    total_count = len(df)

    for seg in sorted(df[segment_col].unique()):
        seg_data = df[df[segment_col] == seg]
        seg_means = seg_data[feature_cols].mean()

        relative_indices = (seg_means / (overall_means + 1e-10)).to_dict()

        profiles[f'segment_{int(seg)}'] = {
            'segment_id': int(seg),
            'size': len(seg_data),
            'pct_of_total': round(100 * len(seg_data) / total_count, 1),
            'relative_indices': relative_indices,
            'means': seg_means.to_dict(),
        }

    return profiles


def validate_segment_sizes(
    df: pd.DataFrame,
    segment_col: str,
    min_pct: float = 0.10,
) -> Dict[str, Any]:
    """
    Validate that all segments meet minimum size requirement.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with segment assignments
    segment_col : str
        Segment column name
    min_pct : float
        Minimum required percentage (default 10%)

    Returns
    -------
    Dict[str, Any]
        Validation results

    Example
    -------
    >>> result = validate_segment_sizes(df, 'segment_id', min_pct=0.10)
    >>> if not result['valid']:
    >>>     print(f"Small segments: {result['small_segments']}")
    """
    seg_dist = df[segment_col].value_counts(normalize=True)

    small_segments = seg_dist[seg_dist < min_pct].index.tolist()

    return {
        'valid': len(small_segments) == 0,
        'n_segments': len(seg_dist),
        'min_pct_required': min_pct,
        'small_segments': small_segments,
        'segment_distribution': seg_dist.to_dict(),
    }


# =============================================================================
# 6. VISUALIZATION UTILITIES
# =============================================================================

def setup_matplotlib():
    """
    Setup matplotlib for non-interactive (batch) use.

    Call this before any plotting in agent code.

    Example
    -------
    >>> setup_matplotlib()
    >>> import matplotlib.pyplot as plt
    """
    import matplotlib
    matplotlib.use('Agg')


def create_bar_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: str,
    horizontal: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    color_threshold: Optional[float] = None,
    color_good: str = 'green',
    color_bad: str = 'orange',
) -> str:
    """
    Create a bar chart and save to file.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    x_col : str
        Column for x-axis (categories)
    y_col : str
        Column for y-axis (values)
    title : str
        Chart title
    output_path : str
        Path to save the chart
    horizontal : bool
        If True, create horizontal bar chart
    figsize : Tuple[int, int]
        Figure size
    color_threshold : float, optional
        Threshold for coloring bars (below=good, above=bad)
    color_good : str
        Color for values below threshold
    color_bad : str
        Color for values above threshold

    Returns
    -------
    str
        Path to saved chart

    Example
    -------
    >>> create_bar_chart(df, 'segment', 'wape', 'WAPE by Segment', 'chart.png')
    """
    setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # Sort data
    data = data.sort_values(y_col, ascending=True)

    # Determine colors
    if color_threshold is not None:
        colors = [color_good if v < color_threshold else color_bad for v in data[y_col]]
    else:
        colors = None

    # Create bar chart
    if horizontal:
        ax.barh(data[x_col].astype(str), data[y_col], color=colors)
        ax.set_xlabel(y_col)
    else:
        ax.bar(data[x_col].astype(str), data[y_col], color=colors)
        ax.set_ylabel(y_col)
        plt.xticks(rotation=45, ha='right')

    ax.set_title(title)
    plt.tight_layout()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def create_heatmap(
    data: Union[pd.DataFrame, np.ndarray],
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    output_path: str,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'RdYlGn',
    center: Optional[float] = None,
    annot: bool = True,
    fmt: str = '.2f',
) -> str:
    """
    Create a heatmap and save to file.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Data to plot
    x_labels : List[str]
        Labels for x-axis (columns)
    y_labels : List[str]
        Labels for y-axis (rows)
    title : str
        Chart title
    output_path : str
        Path to save the chart
    figsize : Tuple[int, int]
        Figure size
    cmap : str
        Colormap name
    center : float, optional
        Value to center the colormap at
    annot : bool
        Whether to annotate cells with values
    fmt : str
        Format string for annotations

    Returns
    -------
    str
        Path to saved chart

    Example
    -------
    >>> create_heatmap(matrix, features, segments, 'Segment Profiles', 'heatmap.png', center=1.0)
    """
    setup_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        data,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=cmap,
        center=center,
        annot=annot,
        fmt=fmt,
        ax=ax,
    )

    ax.set_title(title)
    plt.tight_layout()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def create_histogram(
    data: Union[pd.Series, np.ndarray],
    title: str,
    output_path: str,
    bins: int = 30,
    xlabel: str = 'Value',
    ylabel: str = 'Frequency',
    figsize: Tuple[int, int] = (10, 6),
) -> str:
    """
    Create a histogram and save to file.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        Data to plot
    title : str
        Chart title
    output_path : str
        Path to save the chart
    bins : int
        Number of bins
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    str
        Path to saved chart

    Example
    -------
    >>> create_histogram(df['wape'], 'WAPE Distribution', 'hist.png')
    """
    setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add statistics
    mean_val = np.nanmean(data)
    median_val = np.nanmedian(data)
    ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='blue', linestyle='--', label=f'Median: {median_val:.3f}')
    ax.legend()

    plt.tight_layout()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def create_scatter_plot(
    x: Union[pd.Series, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    title: str,
    output_path: str,
    xlabel: str = 'X',
    ylabel: str = 'Y',
    hue: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (10, 8),
    add_quadrants: bool = False,
    x_threshold: Optional[float] = None,
    y_threshold: Optional[float] = None,
) -> str:
    """
    Create a scatter plot and save to file.

    Parameters
    ----------
    x : pd.Series or np.ndarray
        X values
    y : pd.Series or np.ndarray
        Y values
    title : str
        Chart title
    output_path : str
        Path to save the chart
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    hue : pd.Series, optional
        Color grouping variable
    figsize : Tuple[int, int]
        Figure size
    add_quadrants : bool
        Whether to add quadrant lines
    x_threshold : float, optional
        X threshold for quadrant line
    y_threshold : float, optional
        Y threshold for quadrant line

    Returns
    -------
    str
        Path to saved chart

    Example
    -------
    >>> create_scatter_plot(df['adi'], df['cv2'], 'ADI vs CV²', 'scatter.png',
    ...                     add_quadrants=True, x_threshold=1.32, y_threshold=0.49)
    """
    setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    if hue is not None:
        for label in hue.unique():
            mask = hue == label
            ax.scatter(np.array(x)[mask], np.array(y)[mask], label=label, alpha=0.6)
        ax.legend()
    else:
        ax.scatter(x, y, alpha=0.6)

    if add_quadrants:
        if x_threshold:
            ax.axvline(x_threshold, color='gray', linestyle='--', alpha=0.5)
        if y_threshold:
            ax.axhline(y_threshold, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


# =============================================================================
# 7. CATEGORICAL ENCODING UTILITIES
# =============================================================================

def target_encode(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    alpha: float = 10.0,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Apply target encoding with smoothing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    col : str
        Column to encode
    target_col : str
        Target column
    alpha : float
        Smoothing parameter

    Returns
    -------
    Tuple[pd.Series, Dict[str, float]]
        Encoded series and encoding map

    Example
    -------
    >>> encoded, mapping = target_encode(df, 'category', 'sales')
    """
    global_mean = df[target_col].mean()

    agg = df.groupby(col)[target_col].agg(['mean', 'count'])

    smooth = (agg['count'] * agg['mean'] + alpha * global_mean) / (agg['count'] + alpha)

    encoding_map = smooth.to_dict()
    encoding_map['__global_mean__'] = global_mean

    encoded = df[col].map(smooth).fillna(global_mean)

    return encoded, encoding_map


def ordinal_encode(
    df: pd.DataFrame,
    col: str,
    order_map: Optional[Dict[str, int]] = None,
) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Apply ordinal encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    col : str
        Column to encode
    order_map : Dict[str, int], optional
        Explicit ordering. If None, uses alphabetical order.

    Returns
    -------
    Tuple[pd.Series, Dict[str, int]]
        Encoded series and encoding map

    Example
    -------
    >>> encoded, mapping = ordinal_encode(df, 'tier', {'A': 4, 'B': 3, 'C': 2, 'D': 1})
    """
    if order_map is None:
        unique_vals = sorted(df[col].dropna().unique())
        order_map = {v: i for i, v in enumerate(unique_vals)}

    encoded = df[col].map(order_map).fillna(-1).astype(int)

    return encoded, order_map


def one_hot_encode(
    df: pd.DataFrame,
    col: str,
    drop_first: bool = True,
    prefix: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply one-hot encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    col : str
        Column to encode
    drop_first : bool
        Whether to drop first category (to avoid multicollinearity)
    prefix : str, optional
        Prefix for new column names

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with new columns and list of new column names

    Example
    -------
    >>> df, new_cols = one_hot_encode(df, 'region')
    """
    prefix = prefix or col

    dummies = pd.get_dummies(df[col], prefix=prefix, drop_first=drop_first)
    new_cols = list(dummies.columns)

    df = pd.concat([df, dummies], axis=1)

    return df, new_cols


def save_encoding_specs(
    target_maps: Dict[str, Dict[str, float]],
    ordinal_maps: Dict[str, Dict[str, int]],
    binary_columns: List[str],
    one_hot_columns: List[str],
    interaction_features: Dict[str, Dict[str, Any]],
    output_path: str,
) -> str:
    """
    Save categorical encoding specifications for inference.

    Parameters
    ----------
    target_maps : Dict
        Target encoding maps
    ordinal_maps : Dict
        Ordinal encoding maps
    binary_columns : List[str]
        List of binary columns
    one_hot_columns : List[str]
        List of one-hot encoded columns
    interaction_features : Dict
        Interaction feature specifications
    output_path : str
        Path to save specs

    Returns
    -------
    str
        Path to saved file

    Example
    -------
    >>> save_encoding_specs(target_maps, ordinal_maps, [...], [...], {...}, 'specs.json')
    """
    specs = {
        'target_encoding_maps': target_maps,
        'ordinal_encoding_maps': ordinal_maps,
        'binary_columns': binary_columns,
        'one_hot_columns': one_hot_columns,
        'interaction_features': interaction_features,
    }

    return save_json(specs, output_path)


# =============================================================================
# 8. DIAGNOSTIC UTILITIES
# =============================================================================

def identify_top_bottom_performers(
    df: pd.DataFrame,
    metric_col: str,
    n: int = 10,
    lower_is_better: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify top N and bottom N performers based on a metric.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with metrics
    metric_col : str
        Column containing the metric
    n : int
        Number of top/bottom to return
    lower_is_better : bool
        If True, low values are good (e.g., WAPE)

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Top performers and bottom performers DataFrames

    Example
    -------
    >>> top, bottom = identify_top_bottom_performers(df, 'wape', n=10)
    """
    if lower_is_better:
        top_performers = df.nsmallest(n, metric_col)
        bottom_performers = df.nlargest(n, metric_col)
    else:
        top_performers = df.nlargest(n, metric_col)
        bottom_performers = df.nsmallest(n, metric_col)

    return top_performers, bottom_performers


def analyze_wape_by_categorical(
    df: pd.DataFrame,
    categorical_cols: List[str],
    actual_col: str,
    pred_col: str,
    max_unique: int = 10,
) -> pd.DataFrame:
    """
    Analyze WAPE breakdown by categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actuals and predictions
    categorical_cols : List[str]
        Categorical columns to analyze
    actual_col : str
        Actual values column
    pred_col : str
        Predicted values column
    max_unique : int
        Maximum unique values to analyze per column

    Returns
    -------
    pd.DataFrame
        DataFrame with WAPE by categorical feature values

    Example
    -------
    >>> wape_analysis = analyze_wape_by_categorical(df, ['region', 'category'], 'actual', 'predicted')
    """
    results = []

    for col in categorical_cols:
        if col not in df.columns:
            continue

        n_unique = df[col].nunique()
        if n_unique > max_unique:
            continue

        for val in df[col].unique():
            subset = df[df[col] == val]
            if len(subset) == 0:
                continue

            wape = compute_wape(subset[actual_col], subset[pred_col])

            results.append({
                'feature': col,
                'value': str(val),
                'n_obs': len(subset),
                'wape': round(wape, 4),
                'sum_actual': round(subset[actual_col].sum(), 2),
                'sum_predicted': round(subset[pred_col].sum(), 2),
            })

    return pd.DataFrame(results)


def create_diagnostic_summary(
    overall_metrics: Dict[str, float],
    segment_metrics: pd.DataFrame,
    top_performers: pd.DataFrame,
    bottom_performers: pd.DataFrame,
    model_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a comprehensive diagnostic summary.

    Parameters
    ----------
    overall_metrics : Dict[str, float]
        Overall metrics dictionary
    segment_metrics : pd.DataFrame
        Per-segment metrics
    top_performers : pd.DataFrame
        Top performing keys
    bottom_performers : pd.DataFrame
        Bottom performing keys
    model_info : Dict[str, Any]
        Model information

    Returns
    -------
    Dict[str, Any]
        Comprehensive diagnostic summary

    Example
    -------
    >>> summary = create_diagnostic_summary(metrics, seg_df, top, bottom, model_info)
    """
    return {
        'overall_performance': overall_metrics,
        'segment_summary': {
            'n_segments': len(segment_metrics),
            'best_segment': segment_metrics.nsmallest(1, 'wape').iloc[0].to_dict() if len(segment_metrics) > 0 else None,
            'worst_segment': segment_metrics.nlargest(1, 'wape').iloc[0].to_dict() if len(segment_metrics) > 0 else None,
        },
        'top_performers_summary': {
            'count': len(top_performers),
            'avg_wape': float(top_performers['wape'].mean()) if 'wape' in top_performers else None,
        },
        'bottom_performers_summary': {
            'count': len(bottom_performers),
            'avg_wape': float(bottom_performers['wape'].mean()) if 'wape' in bottom_performers else None,
        },
        'model_info': model_info,
    }


# =============================================================================
# 9. SMART OUTPUT UTILITIES
# =============================================================================

class SmartPrinter:
    """
    Smart printer that limits output to prevent context overflow.

    Example
    -------
    >>> printer = SmartPrinter(max_prints=25)
    >>> printer.print("Message 1")
    >>> printer.print("Message 2")
    """

    def __init__(self, max_prints: int = 25, max_length: int = 500):
        self.max_prints = max_prints
        self.max_length = max_length
        self.print_count = 0

    def print(self, msg: Any) -> None:
        """Print with limits."""
        if self.print_count >= self.max_prints:
            return

        self.print_count += 1
        msg_str = str(msg)[:self.max_length]
        print(msg_str)

    def remaining(self) -> int:
        """Get remaining prints."""
        return self.max_prints - self.print_count

    def reset(self) -> None:
        """Reset print counter."""
        self.print_count = 0


def summarize_dataframe(df: pd.DataFrame, name: str = 'DataFrame') -> str:
    """
    Create a concise summary of a DataFrame (for printing).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to summarize
    name : str
        Name for the DataFrame

    Returns
    -------
    str
        Summary string

    Example
    -------
    >>> print(summarize_dataframe(df, 'Sales Data'))
    """
    return (
        f"{name}: {len(df)} rows × {len(df.columns)} cols | "
        f"Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}"
    )


def summarize_dict(d: Dict[str, Any], name: str = 'Dict', max_items: int = 5) -> str:
    """
    Create a concise summary of a dictionary (for printing).

    Parameters
    ----------
    d : Dict
        Dictionary to summarize
    name : str
        Name for the dictionary
    max_items : int
        Maximum items to show

    Returns
    -------
    str
        Summary string

    Example
    -------
    >>> print(summarize_dict(config, 'Config'))
    """
    keys = list(d.keys())[:max_items]
    more = f" + {len(d) - max_items} more" if len(d) > max_items else ""
    return f"{name}: {len(d)} keys | Keys: {keys}{more}"


# =============================================================================
# 10. CONTEXT FILE UTILITIES
# =============================================================================

def create_context_file(
    content: Dict[str, Any],
    output_path: str,
    source_crew: str,
    target_crew: str,
    purpose: str,
) -> str:
    """
    Create a context file for passing information between crews.

    Parameters
    ----------
    content : Dict[str, Any]
        Content to include in context file
    output_path : str
        Path to save context file
    source_crew : str
        Name of source crew
    target_crew : str
        Name of target crew
    purpose : str
        Purpose of this context file

    Returns
    -------
    str
        Path to saved file

    Example
    -------
    >>> create_context_file(data, 'eda_to_seg.json', 'EDA', 'Segmentation', 'Cluster recommendations')
    """
    context = {
        'metadata': {
            'source_crew': source_crew,
            'target_crew': target_crew,
            'purpose': purpose,
            'created_at': pd.Timestamp.now().isoformat(),
        },
        'content': content,
    }

    return save_json(context, output_path)


def load_context_file(file_path: str) -> Dict[str, Any]:
    """
    Load a context file and return its content.

    Parameters
    ----------
    file_path : str
        Path to context file

    Returns
    -------
    Dict[str, Any]
        Content from context file

    Example
    -------
    >>> context = load_context_file('eda_to_seg.json')
    >>> recommendations = context['cluster_recommendations']
    """
    data = load_json(file_path)
    return data.get('content', data)


def merge_context_files(
    file_paths: List[str],
    output_path: str,
    merged_purpose: str,
) -> str:
    """
    Merge multiple context files into one.

    Parameters
    ----------
    file_paths : List[str]
        List of context file paths to merge
    output_path : str
        Path to save merged file
    merged_purpose : str
        Purpose of merged file

    Returns
    -------
    str
        Path to saved merged file

    Example
    -------
    >>> merge_context_files(['eda_ctx.json', 'seg_ctx.json'], 'merged.json', 'Feature Engineering')
    """
    merged_content = {}
    sources = []

    for path in file_paths:
        if os.path.exists(path):
            data = load_json(path)

            # Extract source info if available
            if 'metadata' in data:
                sources.append(data['metadata'].get('source_crew', os.path.basename(path)))
            else:
                sources.append(os.path.basename(path))

            # Merge content
            content = data.get('content', data)
            merged_content[os.path.basename(path).replace('.json', '')] = content

    return create_context_file(
        content=merged_content,
        output_path=output_path,
        source_crew=', '.join(sources),
        target_crew='Multiple',
        purpose=merged_purpose,
    )


# =============================================================================
# 11. MODEL UTILITIES
# =============================================================================

def save_model(model: Any, file_path: str) -> str:
    """
    Save a trained model using joblib.

    Parameters
    ----------
    model : Any
        Trained model object
    file_path : str
        Path to save model

    Returns
    -------
    str
        Path to saved model

    Example
    -------
    >>> save_model(lgb_model, 'model.pkl')
    """
    import joblib

    os.makedirs(os.path.dirname(file_path), exist_ok=True) if os.path.dirname(file_path) else None
    joblib.dump(model, file_path)

    return file_path


def load_model(file_path: str) -> Any:
    """
    Load a saved model using joblib.

    Parameters
    ----------
    file_path : str
        Path to model file

    Returns
    -------
    Any
        Loaded model object

    Example
    -------
    >>> model = load_model('model.pkl')
    """
    import joblib

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    return joblib.load(file_path)


def date_based_train_val_test_split(
    df: pd.DataFrame,
    date_col: str,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test sets based on date ranges.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_col : str
        Date column name
    train_start, train_end : str
        Training date range
    val_start, val_end : str
        Validation date range
    test_start, test_end : str
        Test date range

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, and test DataFrames

    Example
    -------
    >>> train, val, test = date_based_train_val_test_split(
    ...     df, 'year_week', '202001', '202352', '202401', '202426', '202427', '202452')
    """
    # Handle type mismatch: convert date range params to match df[date_col] dtype
    col_dtype = df[date_col].dtype
    if pd.api.types.is_integer_dtype(col_dtype):
        # Integer column (e.g., year_week as 202521)
        ts = int(train_start) if isinstance(train_start, str) else train_start
        te = int(train_end) if isinstance(train_end, str) else train_end
        vs = int(val_start) if isinstance(val_start, str) else val_start
        ve = int(val_end) if isinstance(val_end, str) else val_end
        tss = int(test_start) if isinstance(test_start, str) else test_start
        tse = int(test_end) if isinstance(test_end, str) else test_end
    elif col_dtype == 'object' or pd.api.types.is_string_dtype(col_dtype):
        ts, te = str(train_start), str(train_end)
        vs, ve = str(val_start), str(val_end)
        tss, tse = str(test_start), str(test_end)
    else:
        ts, te = train_start, train_end
        vs, ve = val_start, val_end
        tss, tse = test_start, test_end

    train_df = df[(df[date_col] >= ts) & (df[date_col] <= te)].copy()
    val_df = df[(df[date_col] >= vs) & (df[date_col] <= ve)].copy()
    test_df = df[(df[date_col] >= tss) & (df[date_col] <= tse)].copy()

    return train_df, val_df, test_df


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data I/O
    'load_csv', 'save_csv', 'load_json', 'save_json', 'merge_dataframes',

    # Metrics
    'compute_wape', 'compute_mae', 'compute_rmse', 'compute_bias', 'compute_smape',
    'compute_all_metrics', 'compute_wape_by_group', 'compute_metrics_at_level',

    # Feature Engineering
    'create_lag_features', 'create_rolling_features', 'create_ewm_features',
    'create_cyclical_features', 'create_calendar_features',
    'create_intermittency_features', 'create_interaction_features',

    # Time Series
    'ADI_THRESHOLD', 'CV2_THRESHOLD',
    'compute_demand_characteristics', 'classify_demand_pattern',
    'add_demand_classification', 'test_stationarity',

    # Clustering
    'scale_features', 'cluster_with_hdbscan', 'cluster_with_kmeans',
    'compute_segment_profiles', 'validate_segment_sizes',

    # Visualization
    'setup_matplotlib', 'create_bar_chart', 'create_heatmap',
    'create_histogram', 'create_scatter_plot',

    # Categorical Encoding
    'target_encode', 'ordinal_encode', 'one_hot_encode', 'save_encoding_specs',

    # Diagnostics
    'identify_top_bottom_performers', 'analyze_wape_by_categorical',
    'create_diagnostic_summary',

    # Smart Output
    'SmartPrinter', 'summarize_dataframe', 'summarize_dict',

    # Context Files
    'create_context_file', 'load_context_file', 'merge_context_files',

    # Model Utilities
    'save_model', 'load_model', 'date_based_train_val_test_split',
]
