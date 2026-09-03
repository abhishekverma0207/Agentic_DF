# utils/code_execution_tool.py

"""
Final, fully Pydantic-v2-compatible CodeExecutionTool for CrewAI v1.4.1.

CRITICAL: This tool enforces strict output limits to prevent context window overflow.
Output that exceeds limits is TRUNCATED with a clear error message.

CONTEXT WINDOW MANAGEMENT:
==========================
The main cause of context overflow is NOT single-output size, but ACCUMULATED
outputs across many tool calls (100+ calls = 100+ outputs in conversation history).

This tool now implements:
1. AGGRESSIVE output compression (much smaller limits)
2. Smart summarization of successful outputs
3. File-based state tracking to reduce repeated output
4. Execution counter to warn agents about context accumulation

Agents MUST:
1. NEVER print raw data, DataFrames, or large collections
2. Process data in chunks for large datasets
3. Save all results to FILES, not print statements
4. Only print: progress counts, file paths, completion messages
"""

import builtins
import contextlib
import io
import os
import textwrap
import traceback
import logging
import json
from typing import Optional, Any, Dict, List, Type

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

logger = logging.getLogger(__name__)

# =============================================================================
# EXECUTION TRACKING - Monitor cumulative context usage
# =============================================================================
_execution_count = 0
_cumulative_output_chars = 0

def reset_execution_tracking():
    """Reset execution tracking counters (call at start of each crew)."""
    global _execution_count, _cumulative_output_chars
    _execution_count = 0
    _cumulative_output_chars = 0
    logger.info("Code execution tracking reset")

def get_execution_stats() -> Dict[str, Any]:
    """Get current execution statistics."""
    return {
        'execution_count': _execution_count,
        'cumulative_output_chars': _cumulative_output_chars,
        'estimated_tokens': _cumulative_output_chars // 4,
    }

# =============================================================================
# OUTPUT LIMITS - VERY GENEROUS (Context boundaries handle summarization)
# =============================================================================
# With automatic context summarization enabled, we can afford much larger
# output limits. The system handles context management automatically.
#
# These limits are now primarily for sanity checks, not context management.

MAX_OUTPUT_CHARS = 100000   # ~25K tokens - very generous for detailed output
MAX_OUTPUT_LINES = 1000     # Max 1000 lines - plenty for comprehensive output
MAX_RESULT_CHARS = 20000    # Result variable - allows larger dicts

# Warning thresholds (relaxed since context is auto-summarized)
WARN_EXECUTION_COUNT = 50   # Warn agent after 50 tool calls
CRITICAL_EXECUTION_COUNT = 100  # Critical warning at 100 calls
MAX_CUMULATIVE_CHARS = 1000000  # ~250K tokens cumulative (auto-summarized)


# =============================================================================
# Pydantic Args Schema
# =============================================================================

class CodeExecutionToolSchema(BaseModel):
    """
    Arguments for python_code_executor tool.
    """

    code: str = Field(
        ...,
        description=(
            "Python code to execute. Write complete, runnable code. "
            "Output limits are generous (100K chars / 1000 lines) since context "
            "is automatically summarized. However, best practice is still to:\n"
            "1. Save large results to FILES (CSV, JSON)\n"
            "2. Print meaningful summaries rather than raw data\n"
            "3. Avoid assigning large DataFrames to `result` variable"
        ),
    )

    working_dir: Optional[str] = Field(
        default=None,
        description=("Optional working directory to temporarily chdir into."),
    )


# Resolve forward annotations (important for Pydantic v2)
CodeExecutionToolSchema.model_rebuild()


def _truncate_output(text: str, max_chars: int, max_lines: int, label: str) -> tuple[str, bool]:
    """
    Truncate output to prevent context window overflow.

    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    if not text:
        return text, False

    original_chars = len(text)
    original_lines = text.count('\n') + 1

    truncated = False
    truncation_reason = []

    # Truncate by lines first
    lines = text.split('\n')
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        text = '\n'.join(lines)
        truncated = True
        truncation_reason.append(f"{original_lines} lines → {max_lines}")

    # Then truncate by characters
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
        truncation_reason.append(f"{original_chars} chars → {max_chars}")

    if truncated:
        reason = ", ".join(truncation_reason)
        error_msg = (
            f"\n\n{'='*60}\n"
            f"🚨 OUTPUT OVERFLOW: {label} TRUNCATED ({reason})\n"
            f"{'='*60}\n"
            f"YOUR CODE IS PRINTING RAW DATA INSTEAD OF SUMMARIES!\n\n"
            f"REQUIRED PATTERN:\n"
            f"  1. SAVE to file: df.to_csv('output.csv')\n"
            f"  2. PRINT summary: print(f'Saved: output.csv ({{len(df)}} rows)')\n\n"
            f"INCORRECT (what you're doing):\n"
            f"  print(df)          # NEVER print DataFrames\n"
            f"  print(results)     # NEVER print raw dicts/lists\n"
            f"  for x in items: print(x)  # NEVER print per-item\n\n"
            f"CORRECT:\n"
            f"  df.to_csv('file.csv')\n"
            f"  print(f'Rows: {{len(df)}}, Cols: {{len(df.columns)}}')\n"
            f"{'='*60}\n"
        )
        logger.error(
            f"Code output truncated ({label}): {original_chars} chars, "
            f"{original_lines} lines. Agent code is printing too much data!"
        )
        return text + error_msg, True

    return text, False


# =============================================================================
# Tool Implementation
# =============================================================================

class CodeExecutionTool(BaseTool):
    """
    CrewAI tool that executes arbitrary Python code and returns stdout/result/error.

    Output limits are generous (100K chars / 1000 lines) since context is
    automatically summarized by the system. This allows for detailed output
    while still preventing runaway executions.

    Best practices (still recommended for clarity):
    - Save large datasets to FILES (CSV, JSON, pickle)
    - Print meaningful summaries for human readability
    - Use `result` variable for small return values

    Write Protection:
    - Pass protected_paths=['path/to/dir'] to prevent LLM code from writing
      to those directories (e.g., protect EDA outputs from segmentation agent).
    - Reads are always allowed; only writes (mode 'w', 'a', 'x') are blocked.
    """

    name: str = "python_code_executor"
    description: str = (
        "Executes Python code in a controlled local environment. "
        "Returns stdout, `result` variable, and traceback on error. "
        "Output limits are generous (100K chars / 1000 lines). "
        "Best practice: save large data to files, print summaries."
    )

    args_schema: Type[CodeExecutionToolSchema] = CodeExecutionToolSchema
    _globals: Dict[str, Any] = {}
    # Directories that are READ-ONLY - LLM code cannot write files here
    protected_paths: List[str] = Field(default_factory=list, description="Directories where writes are blocked")

    def _run(self, code: str, working_dir: Optional[str] = None) -> str:
        global _execution_count, _cumulative_output_chars
        _execution_count += 1

        # DEBUG: Log that the tool is being called
        logger.info(f"[CodeExecutionTool] Execution #{_execution_count} starting")

        # BUGFIX: Validate working_dir is a valid path string, not a number or other type
        # This can happen due to CrewAI/Pydantic serialization issues
        if working_dir is not None:
            # Convert to string if it's not already
            working_dir = str(working_dir)
            # Check if it looks like a valid path (not just a number)
            if working_dir.isdigit() or not os.path.exists(working_dir):
                logger.warning(
                    f"[CodeExecutionTool] Invalid working_dir '{working_dir}' - "
                    f"ignoring and using current directory"
                )
                working_dir = None

        logger.info(f"[CodeExecutionTool] Working dir: {working_dir or os.getcwd()}")
        logger.info(f"[CodeExecutionTool] Code length: {len(code)} chars")
        logger.info(f"[CodeExecutionTool] Code preview: {code[:200]}...")

        code_to_exec = textwrap.dedent(code).strip()

        if not self._globals:
            self._globals = {}

        # =====================================================================
        # WRITE PROTECTION: Prevent LLM code from writing to protected dirs
        # =====================================================================
        # Resolve protected paths once (handles symlinks, relative paths)
        resolved_protected = []
        for p in self.protected_paths:
            try:
                resolved_protected.append(os.path.realpath(p))
            except Exception:
                resolved_protected.append(os.path.abspath(p))

        if resolved_protected:
            _original_open = builtins.open

            def _guarded_open(file, mode='r', *args, **kwargs):
                """Wrapper that blocks writes to protected directories."""
                # Only intercept write modes
                is_write = any(c in str(mode) for c in ('w', 'a', 'x'))
                if is_write and file is not None:
                    try:
                        real_path = os.path.realpath(str(file))
                        for pdir in resolved_protected:
                            if real_path.startswith(pdir + os.sep) or real_path == pdir:
                                msg = (
                                    f"WRITE BLOCKED: Cannot write to protected directory.\n"
                                    f"  File: {file}\n"
                                    f"  Protected dir: {pdir}\n"
                                    f"  This directory is READ-ONLY for this agent."
                                )
                                logger.warning(f"[CodeExecutionTool] {msg}")
                                raise PermissionError(msg)
                    except PermissionError:
                        raise
                    except Exception:
                        pass  # If path resolution fails, allow the write
                return _original_open(file, mode, *args, **kwargs)

            # Monkey-patch builtins.open so ALL code (including imported modules
            # like pandas to_csv, json.dump) also gets write protection.
            # This is restored in the finally block below.
            builtins.open = _guarded_open

            # Also inject into exec globals for direct open() calls in the LLM code
            self._globals['__builtins__'] = {
                **(__builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)),
                'open': _guarded_open,
            }
            logger.info(
                f"[CodeExecutionTool] Write protection active for {len(resolved_protected)} dir(s): "
                f"{[os.path.basename(p) for p in resolved_protected]}"
            )

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        old_cwd = os.getcwd()
        has_error = False
        try:
            if working_dir:
                os.chdir(working_dir)

            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                try:
                    exec(code_to_exec, self._globals, self._globals)  # nosec: trusted env
                except Exception:
                    has_error = True
                    traceback.print_exc()

            stdout_val = stdout_buf.getvalue().strip()
            stderr_val = stderr_buf.getvalue().strip()

            # Apply AGGRESSIVE truncation
            stdout_val, stdout_truncated = _truncate_output(
                stdout_val, MAX_OUTPUT_CHARS, MAX_OUTPUT_LINES, "STDOUT"
            )
            stderr_val, stderr_truncated = _truncate_output(
                stderr_val, MAX_OUTPUT_CHARS, MAX_OUTPUT_LINES, "STDERR"
            )

            result_val = None
            if "result" in self._globals:
                try:
                    result_val = repr(self._globals["result"])
                    if len(result_val) > MAX_RESULT_CHARS:
                        result_val = (
                            result_val[:MAX_RESULT_CHARS] +
                            f"\n[TRUNCATED: {len(result_val)} → {MAX_RESULT_CHARS}]"
                        )
                except Exception:
                    result_val = "<unrepresentable>"

            # Build output with context warnings
            sections = []

            # Add context accumulation warning if needed
            warning_msg = _get_context_warning(_execution_count, _cumulative_output_chars)
            if warning_msg:
                sections.append(warning_msg)

            if stdout_val:
                sections.append("[STDOUT]\n" + stdout_val)
            if stderr_val:
                sections.append("[ERROR]\n" + stderr_val)
            if result_val is not None:
                sections.append("[RESULT]\n" + result_val)
            if not sections or (len(sections) == 1 and warning_msg):
                sections.append("[OK] Code executed successfully.")

            output = "\n\n".join(sections)

            # Track cumulative output
            _cumulative_output_chars += len(output)

            # DEBUG: Log the result
            logger.info(f"[CodeExecutionTool] Execution #{_execution_count} completed")
            logger.info(f"[CodeExecutionTool] Has error: {has_error}")
            logger.info(f"[CodeExecutionTool] Output preview: {output[:500]}...")

            return output

        finally:
            # Restore builtins.open if we monkey-patched it
            if resolved_protected:
                try:
                    builtins.open = _original_open
                except Exception:
                    pass
            try:
                os.chdir(old_cwd)
            except Exception:
                pass


def _get_context_warning(exec_count: int, cumulative_chars: int) -> Optional[str]:
    """Generate context accumulation warning if thresholds exceeded."""
    warnings = []

    if exec_count >= CRITICAL_EXECUTION_COUNT:
        warnings.append(
            f"🚨 CRITICAL: {exec_count} tool calls! Context overflow imminent!\n"
            f"STOP iterating. Save progress to file. Use BATCH approach for remaining work."
        )
    elif exec_count >= WARN_EXECUTION_COUNT:
        warnings.append(
            f"⚠️ WARNING: {exec_count} tool calls. Approaching context limits.\n"
            f"Consolidate remaining work into fewer, larger code blocks."
        )

    if cumulative_chars >= MAX_CUMULATIVE_CHARS:
        warnings.append(
            f"🚨 CRITICAL: {cumulative_chars:,} chars cumulative output!\n"
            f"Print ONLY: 'Saved to: filepath' - nothing else."
        )

    if warnings:
        return "[CONTEXT WARNING]\n" + "\n".join(warnings)
    return None
