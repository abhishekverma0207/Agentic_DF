# utils/smart_output.py
"""
Smart Output Management for Agentic AI Crews.

This module provides utility functions and patterns that agents MUST use
to avoid overwhelming the LLM context window with too much output.

CRITICAL RULES:
1. NEVER print raw data - always compute summaries first
2. ALWAYS check sizes before creating any output
3. SAVE detailed results to files, PRINT only counts and paths
4. Use these helper functions in ALL agent-generated code
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import json
import os


# =============================================================================
# OUTPUT LIMITS - HARD LIMITS FOR AGENT CODE
# =============================================================================
# These limits protect the LLM context window from overflow

MAX_PRINT_ROWS = 10          # Maximum rows to ever print from a DataFrame
MAX_PRINT_DICT_ITEMS = 10    # Maximum dict items to ever print
MAX_PRINT_LIST_ITEMS = 10    # Maximum list items to ever print
MAX_PRINT_CHARS = 5000       # Maximum characters in a single print statement
MAX_TOTAL_PRINTS = 20        # Maximum print statements per code block


def safe_shape(df: pd.DataFrame) -> str:
    """Return shape info without printing any data."""
    return f"({len(df)} rows, {len(df.columns)} cols)"


def safe_describe(df: pd.DataFrame, name: str = "DataFrame") -> str:
    """Return a safe 1-line summary of a DataFrame."""
    null_pct = df.isnull().mean().mean() * 100
    return f"{name}: {safe_shape(df)}, nulls: {null_pct:.1f}%, cols: {list(df.columns)[:5]}..."


def safe_value_counts(series: pd.Series, name: str = "column", top_n: int = 5) -> str:
    """Return top N value counts as a compact string."""
    vc = series.value_counts()
    total = len(series)
    result = [f"{name} distribution (top {min(top_n, len(vc))}/{len(vc)} unique):"]
    for val, count in vc.head(top_n).items():
        result.append(f"  {val}: {count} ({count/total*100:.1f}%)")
    return "\n".join(result)


def safe_dict_summary(d: Dict, name: str = "dict", max_items: int = 5) -> str:
    """Return a summary of a dictionary without printing all items."""
    n_items = len(d)
    sample_keys = list(d.keys())[:max_items]
    return f"{name}: {n_items} items, sample keys: {sample_keys}"


def safe_list_summary(lst: List, name: str = "list", max_items: int = 5) -> str:
    """Return a summary of a list without printing all items."""
    n_items = len(lst)
    sample = lst[:max_items]
    return f"{name}: {n_items} items, sample: {sample}..."


def check_before_print(obj: Any, description: str = "object") -> None:
    """
    Check object size before printing. Raises warning if too large.

    Use this BEFORE every print statement in agent code:

        check_before_print(my_df, "results dataframe")
        if len(my_df) <= MAX_PRINT_ROWS:
            print(my_df)
        else:
            print(safe_describe(my_df))
    """
    if isinstance(obj, pd.DataFrame):
        if len(obj) > MAX_PRINT_ROWS:
            print(f"WARNING: {description} has {len(obj)} rows - use safe_describe() instead of printing")
            return
    elif isinstance(obj, dict):
        if len(obj) > MAX_PRINT_DICT_ITEMS:
            print(f"WARNING: {description} has {len(obj)} items - use safe_dict_summary() instead")
            return
    elif isinstance(obj, (list, tuple)):
        if len(obj) > MAX_PRINT_LIST_ITEMS:
            print(f"WARNING: {description} has {len(obj)} items - use safe_list_summary() instead")
            return


def smart_print_df(df: pd.DataFrame, name: str = "DataFrame", max_rows: int = 5) -> None:
    """
    Smart print for DataFrames - always safe, never overflows.

    - Small df (<=max_rows): prints actual data
    - Large df: prints summary only
    """
    if len(df) <= max_rows:
        print(f"\n{name} ({len(df)} rows):")
        print(df.to_string())
    else:
        print(f"\n{name} summary (too large to print, {len(df)} rows):")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Head (3 rows):")
        print(df.head(3).to_string())


def smart_print_dict(d: Dict, name: str = "dict", max_items: int = 5) -> None:
    """
    Smart print for dicts - always safe, never overflows.
    """
    if len(d) <= max_items:
        print(f"\n{name} ({len(d)} items): {d}")
    else:
        sample = dict(list(d.items())[:max_items])
        print(f"\n{name} ({len(d)} items, showing {max_items}):")
        for k, v in sample.items():
            v_str = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
            print(f"  {k}: {v_str}")
        print(f"  ... and {len(d) - max_items} more items")


def save_and_summarize_df(
    df: pd.DataFrame,
    filepath: str,
    name: str = "data"
) -> str:
    """
    Save DataFrame to file and return a summary string (NOT the data).

    This is the CORRECT pattern for agent code:
        summary = save_and_summarize_df(results, 'output.csv', 'results')
        print(summary)  # Safe - only prints summary
    """
    df.to_csv(filepath, index=False)

    summary_lines = [
        f"Saved {name} to: {filepath}",
        f"  Rows: {len(df):,}",
        f"  Columns: {len(df.columns)}",
    ]

    # Add distribution info for key columns if present
    for col in ['intermittency_class', 'segment_id', 'cluster', 'model_group']:
        if col in df.columns:
            vc = df[col].value_counts()
            summary_lines.append(f"  {col} distribution: {dict(vc)}")

    return "\n".join(summary_lines)


def save_and_summarize_json(
    data: Union[Dict, List],
    filepath: str,
    name: str = "data"
) -> str:
    """
    Save data to JSON file and return a summary string (NOT the data).
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    if isinstance(data, dict):
        summary = f"Saved {name} to: {filepath}\n  Keys ({len(data)}): {list(data.keys())[:10]}"
    else:
        summary = f"Saved {name} to: {filepath}\n  Items: {len(data)}"

    return summary


def compute_then_summarize(
    df: pd.DataFrame,
    groupby_cols: List[str],
    agg_dict: Dict[str, str],
    summary_name: str = "aggregation"
) -> tuple:
    """
    Compute aggregation and return both result AND summary.

    Returns:
        (aggregated_df, summary_string)

    Usage in agent code:
        agg_df, summary = compute_then_summarize(df, ['category'], {'value': 'mean'}, 'category means')
        print(summary)  # Safe
        agg_df.to_csv('output.csv')  # Save full results
    """
    agg_df = df.groupby(groupby_cols).agg(agg_dict).reset_index()

    summary = f"{summary_name}: {len(agg_df)} groups from {len(df):,} rows"
    if len(agg_df) <= MAX_PRINT_ROWS:
        summary += f"\n{agg_df.to_string()}"
    else:
        summary += f"\n  Top 5 groups:\n{agg_df.head().to_string()}"

    return agg_df, summary


# =============================================================================
# STANDARD OUTPUT PATTERN - COPY THIS INTO AGENT CODE
# =============================================================================

AGENT_OUTPUT_PATTERN = '''
# =============================================================================
# SMART OUTPUT PATTERN - PREVENTS CONTEXT OVERFLOW
# =============================================================================
# RULE 1: ALWAYS check sizes before printing
# RULE 2: NEVER print raw DataFrames/dicts/lists directly
# RULE 3: SAVE to files first, then print ONLY the summary

def safe_output(msg):
    """Print only if message is under 500 chars."""
    if len(str(msg)) > 500:
        print(f"[OUTPUT TOO LONG - {len(str(msg))} chars, truncated]")
        print(str(msg)[:500] + "...")
    else:
        print(msg)

# For DataFrames:
# BAD:  print(df)  # Could be thousands of rows!
# GOOD: print(f"DataFrame: {len(df)} rows, {len(df.columns)} cols")
#       df.to_csv('output.csv')
#       print(f"Saved to output.csv")

# For dicts:
# BAD:  print(my_dict)  # Could be huge!
# GOOD: print(f"Dict with {len(my_dict)} keys: {list(my_dict.keys())[:5]}...")
#       with open('output.json', 'w') as f: json.dump(my_dict, f)
#       print(f"Saved to output.json")

# For loops:
# BAD:  for key in keys: print(f"Processing {key}...")  # Could be 10000 prints!
# GOOD: for i, key in enumerate(keys):
#           if i % 1000 == 0: print(f"Progress: {i}/{len(keys)}")

# At end of script:
print(f"\\n=== SUMMARY ===")
print(f"Processed: X items")
print(f"Output files: Y")
print(f"Key metrics: Z")
# =============================================================================
'''


def get_agent_output_instructions() -> str:
    """
    Returns the standardized output instructions that should be included
    in every agent's backstory.
    """
    return """
## CRITICAL OUTPUT RULES (VIOLATION = CONTEXT OVERFLOW FAILURE)
These rules are NON-NEGOTIABLE. Breaking them causes the entire pipeline to fail.

### RULE 1: CHECK SIZES BEFORE ANY OUTPUT
```python
# ALWAYS do this before printing:
print(f"Data shape: {len(df)} rows")  # Check size first
if len(df) > 10:
    print("Data too large to print - saving to file")
    df.to_csv('output.csv')
else:
    print(df)  # Only print if small
```

### RULE 2: SAVE TO FILES, PRINT SUMMARIES
```python
# CORRECT PATTERN:
results_df.to_csv('results.csv', index=False)
print(f"Saved results.csv: {len(results_df)} rows, {len(results_df.columns)} cols")

# Distribution summary (not raw data):
for col in ['segment', 'cluster']:
    if col in results_df.columns:
        counts = results_df[col].value_counts().to_dict()
        print(f"  {col} distribution: {counts}")
```

### RULE 3: AGGREGATE BEFORE PRINTING
```python
# BAD - prints raw data:
for key, group in df.groupby('key'):
    print(f"{key}: {group['value'].mean()}")  # Could be 10000 lines!

# GOOD - aggregate first:
summary = df.groupby('key')['value'].mean()
print(f"Computed means for {len(summary)} keys")
summary.to_csv('key_means.csv')
print(f"Saved to key_means.csv")
```

### RULE 4: PROGRESS EVERY 1000, NOT EVERY 1
```python
# BAD:
for i, item in enumerate(items):
    print(f"Processing {item}")  # 10000 prints!

# GOOD:
for i, item in enumerate(items):
    if i % 1000 == 0:
        print(f"Progress: {i:,}/{len(items):,}")
print(f"Completed: {len(items):,} items")
```

### FINAL SUMMARY FORMAT (END OF EVERY SCRIPT)
```python
print("\\n" + "="*50)
print("EXECUTION SUMMARY")
print("="*50)
print(f"Files created: {len(output_files)}")
for f in output_files[:5]:  # Max 5 files listed
    print(f"  - {f}")
print(f"Total records processed: {total_records:,}")
print(f"Key insight: [one line summary]")
print("="*50)
```
"""
