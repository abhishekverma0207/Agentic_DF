# utils/output_suppressor.py
"""
Output Suppression Utility for Context Overflow Prevention.

This module provides utilities to capture and limit print output from
autonomous functions to prevent context overflow in CrewAI pipelines.

Usage:
    from utils.output_suppressor import suppress_verbose_output, get_output_summary

    # Option 1: As context manager
    with suppress_verbose_output(max_lines=10):
        run_some_verbose_function()

    # Option 2: Get summary
    summary = get_output_summary(output_lines)
"""

from __future__ import annotations

import io
import sys
from contextlib import contextmanager
from typing import Generator, List, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_OUTPUT_LINES = 10  # Maximum lines to capture
MAX_LINE_LENGTH = 200  # Maximum characters per line
SUMMARY_TEMPLATE = "Output: {captured}/{total} lines ({truncated} truncated)"


# =============================================================================
# OUTPUT CAPTURE
# =============================================================================

class OutputCapture:
    """
    Captures stdout/stderr and limits output size.
    """

    def __init__(
        self,
        max_lines: int = MAX_OUTPUT_LINES,
        max_line_length: int = MAX_LINE_LENGTH,
        silent: bool = False,
    ):
        self.max_lines = max_lines
        self.max_line_length = max_line_length
        self.silent = silent

        self.captured_lines: List[str] = []
        self.total_lines = 0
        self.truncated_count = 0

        self._original_stdout = None
        self._original_stderr = None
        self._capture_buffer = None

    def start(self) -> None:
        """Start capturing output."""
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._capture_buffer = io.StringIO()

        # Create a custom stdout that both captures and optionally prints
        sys.stdout = _TeeWriter(
            self._capture_buffer,
            self._original_stdout if not self.silent else None,
        )
        sys.stderr = _TeeWriter(
            self._capture_buffer,
            self._original_stderr if not self.silent else None,
        )

    def stop(self) -> List[str]:
        """Stop capturing and return captured lines."""
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
        if self._original_stderr is not None:
            sys.stderr = self._original_stderr

        if self._capture_buffer is not None:
            all_output = self._capture_buffer.getvalue()
            all_lines = all_output.splitlines()
            self.total_lines = len(all_lines)

            # Process lines with truncation
            for i, line in enumerate(all_lines):
                if i >= self.max_lines:
                    self.truncated_count = self.total_lines - self.max_lines
                    break

                # Truncate long lines
                if len(line) > self.max_line_length:
                    line = line[:self.max_line_length] + "..."

                self.captured_lines.append(line)

        return self.captured_lines

    def get_summary(self) -> str:
        """Get a summary of captured output."""
        return SUMMARY_TEMPLATE.format(
            captured=len(self.captured_lines),
            total=self.total_lines,
            truncated=self.truncated_count,
        )


class _TeeWriter:
    """Writer that optionally writes to two destinations."""

    def __init__(self, buffer: io.StringIO, original: Optional[io.TextIOWrapper]):
        self.buffer = buffer
        self.original = original

    def write(self, text: str) -> int:
        self.buffer.write(text)
        if self.original is not None:
            self.original.write(text)
        return len(text)

    def flush(self) -> None:
        self.buffer.flush()
        if self.original is not None:
            self.original.flush()


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

@contextmanager
def suppress_verbose_output(
    max_lines: int = MAX_OUTPUT_LINES,
    max_line_length: int = MAX_LINE_LENGTH,
    silent: bool = False,
) -> Generator[OutputCapture, None, None]:
    """
    Context manager to capture and limit output.

    Parameters
    ----------
    max_lines : int
        Maximum lines to capture (default: 10)
    max_line_length : int
        Maximum characters per line (default: 200)
    silent : bool
        If True, suppress all output to console

    Yields
    ------
    OutputCapture
        Object containing captured output

    Example
    -------
    >>> with suppress_verbose_output(max_lines=5) as capture:
    ...     print("Line 1")
    ...     print("Line 2")
    ...     # ... many more lines ...
    >>> print(capture.get_summary())
    Output: 5/100 lines (95 truncated)
    """
    capture = OutputCapture(
        max_lines=max_lines,
        max_line_length=max_line_length,
        silent=silent,
    )
    try:
        capture.start()
        yield capture
    finally:
        capture.stop()


@contextmanager
def minimal_output() -> Generator[OutputCapture, None, None]:
    """
    Context manager for minimal output (10 lines, silent mode).

    This is the recommended wrapper for autonomous functions in agents.
    """
    capture = OutputCapture(
        max_lines=10,
        max_line_length=150,
        silent=True,  # Don't print to console
    )
    try:
        capture.start()
        yield capture
    finally:
        capture.stop()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_output_summary(
    output_lines: List[str],
    max_lines: int = MAX_OUTPUT_LINES,
) -> str:
    """
    Get a compact summary of output lines.

    Parameters
    ----------
    output_lines : List[str]
        Lines of output
    max_lines : int
        Maximum lines to include

    Returns
    -------
    str
        Compact summary
    """
    if len(output_lines) <= max_lines:
        return "\n".join(output_lines)

    # Include first few and last few lines
    half = max_lines // 2
    summary_lines = output_lines[:half]
    summary_lines.append(f"... ({len(output_lines) - max_lines} lines omitted) ...")
    summary_lines.extend(output_lines[-half:])

    return "\n".join(summary_lines)


def format_minimal_result(
    files_created: int = 0,
    step_name: str = "",
    success: bool = True,
) -> str:
    """
    Format a minimal result string for agent output.

    Parameters
    ----------
    files_created : int
        Number of files created
    step_name : str
        Name of the step completed
    success : bool
        Whether the step succeeded

    Returns
    -------
    str
        Minimal result string (under 50 chars)
    """
    status = "Done" if success else "Failed"
    if step_name:
        return f"{step_name}: {status}. Files: {files_created}"
    return f"{status}. Files: {files_created}"


# =============================================================================
# PRINT WRAPPER
# =============================================================================

class MinimalPrint:
    """
    A print replacement that only outputs critical information.

    Usage:
        print = MinimalPrint(max_lines=10)
        print("Starting...")  # This will be captured
        # After max_lines, additional prints are silently ignored
    """

    def __init__(self, max_lines: int = 10):
        self.max_lines = max_lines
        self.line_count = 0
        self.critical_patterns = [
            "error",
            "fail",
            "exception",
            "warning",
            "done",
            "complete",
            "saved",
            "files:",
        ]

    def __call__(self, *args, **kwargs) -> None:
        """Print with line limit."""
        if self.line_count >= self.max_lines:
            # Check if this is a critical message
            message = " ".join(str(a) for a in args).lower()
            is_critical = any(p in message for p in self.critical_patterns)
            if not is_critical:
                return  # Silently skip non-critical messages

        self.line_count += 1
        print(*args, **kwargs)

    def reset(self) -> None:
        """Reset line counter."""
        self.line_count = 0


# Global minimal print instance for import
minimal_print = MinimalPrint()
