# utils/trace_logging.py
"""
CrewAI Trace Logging Utility
============================

This module provides comprehensive trace logging for CrewAI agents and crews.
It captures all agent interactions, tool calls, LLM responses, and context flow.

Features:
- Detailed trace logs with timestamps
- Separate log files for each crew/agent
- Context size tracking
- Tool call logging
- LLM response capture
- JSON-structured trace output for analysis

Usage:
    from utils.trace_logging import setup_crewai_trace_logging, get_trace_log_path

    # Call this BEFORE running any crews
    setup_crewai_trace_logging(artifact_base_path="/path/to/artifacts", log_level="DEBUG")

    # Get path to trace logs after execution
    trace_path = get_trace_log_path(artifact_base_path)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any, List
import functools


# =============================================================================
# Constants
# =============================================================================

TRACE_LOG_DIR_NAME = "trace_logs"
TRACE_LOG_FILENAME = "crewai_trace.log"
TRACE_JSON_FILENAME = "crewai_trace.jsonl"  # JSON Lines format for structured traces
MAX_LOG_SIZE_MB = 50
BACKUP_COUNT = 5


# =============================================================================
# Custom Trace Logger
# =============================================================================

class CrewAITraceHandler(logging.Handler):
    """
    Custom logging handler that writes structured trace events to a JSON Lines file.
    Each line is a valid JSON object for easy parsing and analysis.
    """

    def __init__(self, trace_file_path: str):
        super().__init__()
        self.trace_file_path = trace_file_path
        self._ensure_dir_exists()

    def _ensure_dir_exists(self):
        os.makedirs(os.path.dirname(self.trace_file_path), exist_ok=True)

    def emit(self, record: logging.LogRecord):
        try:
            trace_event = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "funcName": record.funcName,
                "lineno": record.lineno,
            }

            # Add any extra attributes that might be set
            for key in ["agent", "crew", "task", "tool", "context_size", "event_type"]:
                if hasattr(record, key):
                    trace_event[key] = getattr(record, key)

            with open(self.trace_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_event) + "\n")

        except Exception:
            self.handleError(record)


class CrewAITraceLogger:
    """
    Singleton trace logger for CrewAI operations.
    Provides methods to log specific event types with structured data.
    """

    _instance: Optional["CrewAITraceLogger"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, trace_dir: Optional[str] = None):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._initialized = True
        self.trace_dir = trace_dir or "."
        self.events: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

    def configure(self, trace_dir: str):
        """Configure or reconfigure the trace directory."""
        self.trace_dir = trace_dir
        os.makedirs(trace_dir, exist_ok=True)

    def log_crew_start(self, crew_name: str, agents: List[str], tasks: List[str]):
        """Log when a crew starts execution."""
        event = {
            "event_type": "crew_start",
            "timestamp": datetime.now().isoformat(),
            "crew_name": crew_name,
            "agents": agents,
            "tasks": tasks,
            "agent_count": len(agents),
            "task_count": len(tasks),
        }
        self.events.append(event)
        self._write_event(event)

    def log_crew_end(self, crew_name: str, success: bool, duration_seconds: float, error: Optional[str] = None):
        """Log when a crew finishes execution."""
        event = {
            "event_type": "crew_end",
            "timestamp": datetime.now().isoformat(),
            "crew_name": crew_name,
            "success": success,
            "duration_seconds": duration_seconds,
            "error": error,
        }
        self.events.append(event)
        self._write_event(event)

    def log_agent_start(self, agent_name: str, task_name: str, task_description: str):
        """Log when an agent starts working on a task."""
        event = {
            "event_type": "agent_start",
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "task_name": task_name,
            "task_description_preview": task_description[:500] if task_description else "",
        }
        self.events.append(event)
        self._write_event(event)

    def log_agent_end(self, agent_name: str, task_name: str, output_preview: str, duration_seconds: float):
        """Log when an agent finishes a task."""
        event = {
            "event_type": "agent_end",
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "task_name": task_name,
            "output_preview": output_preview[:1000] if output_preview else "",
            "duration_seconds": duration_seconds,
        }
        self.events.append(event)
        self._write_event(event)

    def log_tool_call(self, agent_name: str, tool_name: str, tool_input: str, tool_output: str, success: bool):
        """Log a tool call made by an agent."""
        event = {
            "event_type": "tool_call",
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "tool_name": tool_name,
            "tool_input_preview": tool_input[:500] if tool_input else "",
            "tool_output_preview": tool_output[:1000] if tool_output else "",
            "success": success,
        }
        self.events.append(event)
        self._write_event(event)

    def log_llm_call(self, agent_name: str, prompt_tokens: int, completion_tokens: int, model: str):
        """Log an LLM API call."""
        event = {
            "event_type": "llm_call",
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "model": model,
        }
        self.events.append(event)
        self._write_event(event)

    def log_context_transfer(self, from_crew: str, to_crew: str, context_file: str, size_bytes: int):
        """Log when context is passed between crews."""
        event = {
            "event_type": "context_transfer",
            "timestamp": datetime.now().isoformat(),
            "from_crew": from_crew,
            "to_crew": to_crew,
            "context_file": context_file,
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 2),
        }
        self.events.append(event)
        self._write_event(event)

    def log_custom(self, event_type: str, **kwargs):
        """Log a custom event with arbitrary data."""
        event = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.events.append(event)
        self._write_event(event)

    def _write_event(self, event: Dict[str, Any]):
        """Write a single event to the trace file."""
        trace_file = os.path.join(self.trace_dir, TRACE_JSON_FILENAME)
        try:
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logging.warning(f"Failed to write trace event: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all traced events."""
        summary = {
            "total_events": len(self.events),
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "event_counts": {},
            "crews_executed": [],
            "total_tool_calls": 0,
            "total_llm_calls": 0,
        }

        for event in self.events:
            event_type = event.get("event_type", "unknown")
            summary["event_counts"][event_type] = summary["event_counts"].get(event_type, 0) + 1

            if event_type == "crew_start":
                summary["crews_executed"].append(event.get("crew_name"))
            elif event_type == "tool_call":
                summary["total_tool_calls"] += 1
            elif event_type == "llm_call":
                summary["total_llm_calls"] += 1

        return summary

    def save_summary(self, output_path: Optional[str] = None):
        """Save the trace summary to a JSON file."""
        if output_path is None:
            output_path = os.path.join(self.trace_dir, "trace_summary.json")

        summary = self.get_summary()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return output_path


# =============================================================================
# Global Trace Logger Instance
# =============================================================================

_trace_logger: Optional[CrewAITraceLogger] = None


def get_trace_logger() -> CrewAITraceLogger:
    """Get the global trace logger instance."""
    global _trace_logger
    if _trace_logger is None:
        _trace_logger = CrewAITraceLogger()
    return _trace_logger


# =============================================================================
# Setup Functions
# =============================================================================

def setup_crewai_trace_logging(
    artifact_base_path: str,
    log_level: str = "DEBUG",
    also_log_to_console: bool = False,
    max_log_size_mb: int = MAX_LOG_SIZE_MB,
) -> str:
    """
    Configure comprehensive trace logging for CrewAI.

    This function:
    1. Creates a trace_logs directory under artifact_base_path
    2. Sets up file handlers for both text logs and JSON traces
    3. Configures CrewAI's internal loggers to write traces to FILES ONLY
    4. Does NOT affect console output - CrewAI verbose output still goes to console
    5. Returns the path to the trace log directory

    Parameters
    ----------
    artifact_base_path : str
        Base path for artifacts (trace_logs will be created here)
    log_level : str
        Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
    also_log_to_console : bool
        Whether to also print traces to console (default: False - file only)
    max_log_size_mb : int
        Maximum size of each log file before rotation

    Returns
    -------
    str
        Path to the trace log directory
    """
    trace_dir = os.path.join(artifact_base_path, TRACE_LOG_DIR_NAME)
    os.makedirs(trace_dir, exist_ok=True)

    # Configure the global trace logger
    trace_logger = get_trace_logger()
    trace_logger.configure(trace_dir)

    # Text log file path
    text_log_path = os.path.join(trace_dir, TRACE_LOG_FILENAME)
    json_log_path = os.path.join(trace_dir, TRACE_JSON_FILENAME)

    # Get numeric log level
    numeric_level = getattr(logging, log_level.upper(), logging.DEBUG)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create handlers - FILE ONLY (no console by default)
    file_handler = RotatingFileHandler(
        text_log_path,
        maxBytes=max_log_size_mb * 1024 * 1024,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(detailed_formatter)

    # JSON trace handler for structured logging (file only)
    json_handler = CrewAITraceHandler(json_log_path)
    json_handler.setLevel(numeric_level)

    # Configure CrewAI-specific loggers to write to FILE ONLY
    # We do NOT touch the root logger to avoid affecting console output
    crewai_loggers = [
        "crewai",
        "crewai.agent",
        "crewai.crew",
        "crewai.task",
        "crewai.tools",
        "crewai.llm",
        "langchain",
        "langchain_core",
        "langchain_anthropic",
        "langchain_aws",
        "litellm",
    ]

    for logger_name in crewai_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(numeric_level)

        # Remove any existing file handlers to avoid duplicates
        for handler in logger.handlers[:]:
            if isinstance(handler, (RotatingFileHandler, CrewAITraceHandler)):
                logger.removeHandler(handler)

        # Add file handlers only
        logger.addHandler(file_handler)
        logger.addHandler(json_handler)

        # IMPORTANT: Set propagate=False so logs don't bubble up to root logger
        # This prevents trace logs from appearing in the console
        logger.propagate = False

    # Optionally add console handler if explicitly requested
    if also_log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        for logger_name in crewai_loggers:
            logging.getLogger(logger_name).addHandler(console_handler)

    # Log initialization to the main logger (not the crewai loggers)
    # This will appear in console as part of normal pipeline logging
    main_logger = logging.getLogger(__name__)
    main_logger.info(f"CrewAI trace logging initialized: {trace_dir}")

    return trace_dir


def get_trace_log_path(artifact_base_path: str) -> str:
    """Get the path to the trace log directory."""
    return os.path.join(artifact_base_path, TRACE_LOG_DIR_NAME)


def get_trace_files(artifact_base_path: str) -> Dict[str, str]:
    """Get paths to all trace files."""
    trace_dir = get_trace_log_path(artifact_base_path)
    return {
        "trace_dir": trace_dir,
        "text_log": os.path.join(trace_dir, TRACE_LOG_FILENAME),
        "json_trace": os.path.join(trace_dir, TRACE_JSON_FILENAME),
        "summary": os.path.join(trace_dir, "trace_summary.json"),
    }


# =============================================================================
# Trace Analysis Utilities
# =============================================================================

def analyze_trace(artifact_base_path: str) -> Dict[str, Any]:
    """
    Analyze the trace log and return a summary.

    Returns
    -------
    dict
        Analysis summary including:
        - Total events
        - Events by type
        - Crew execution times
        - Tool usage statistics
        - Token usage (if available)
    """
    trace_file = os.path.join(artifact_base_path, TRACE_LOG_DIR_NAME, TRACE_JSON_FILENAME)

    if not os.path.exists(trace_file):
        return {"error": f"Trace file not found: {trace_file}"}

    events = []
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    analysis = {
        "total_events": len(events),
        "events_by_type": {},
        "crews": {},
        "tools": {},
        "total_tokens": 0,
        "errors": [],
    }

    for event in events:
        event_type = event.get("event_type", event.get("level", "unknown"))
        analysis["events_by_type"][event_type] = analysis["events_by_type"].get(event_type, 0) + 1

        if event_type == "crew_start":
            crew_name = event.get("crew_name", "unknown")
            analysis["crews"][crew_name] = {"start": event.get("timestamp"), "agents": event.get("agents", [])}

        elif event_type == "crew_end":
            crew_name = event.get("crew_name", "unknown")
            if crew_name in analysis["crews"]:
                analysis["crews"][crew_name]["end"] = event.get("timestamp")
                analysis["crews"][crew_name]["duration"] = event.get("duration_seconds", 0)
                analysis["crews"][crew_name]["success"] = event.get("success", False)

        elif event_type == "tool_call":
            tool_name = event.get("tool_name", "unknown")
            if tool_name not in analysis["tools"]:
                analysis["tools"][tool_name] = {"calls": 0, "successes": 0, "failures": 0}
            analysis["tools"][tool_name]["calls"] += 1
            if event.get("success"):
                analysis["tools"][tool_name]["successes"] += 1
            else:
                analysis["tools"][tool_name]["failures"] += 1

        elif event_type == "llm_call":
            analysis["total_tokens"] += event.get("total_tokens", 0)

        elif event.get("level") == "ERROR":
            analysis["errors"].append({
                "timestamp": event.get("timestamp"),
                "message": event.get("message", ""),
            })

    return analysis


def print_trace_summary(artifact_base_path: str):
    """Print a human-readable summary of the trace."""
    analysis = analyze_trace(artifact_base_path)

    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return

    print("\n" + "="*70)
    print("CREWAI TRACE SUMMARY")
    print("="*70)

    print(f"\nTotal Events: {analysis['total_events']}")

    print("\nEvents by Type:")
    for event_type, count in sorted(analysis["events_by_type"].items()):
        print(f"  {event_type}: {count}")

    print("\nCrew Execution:")
    for crew_name, crew_data in analysis["crews"].items():
        status = "SUCCESS" if crew_data.get("success") else "FAILED"
        duration = crew_data.get("duration", 0)
        print(f"  {crew_name}: {status} ({duration:.2f}s)")
        if crew_data.get("agents"):
            print(f"    Agents: {', '.join(crew_data['agents'])}")

    if analysis["tools"]:
        print("\nTool Usage:")
        for tool_name, tool_data in sorted(analysis["tools"].items()):
            success_rate = tool_data["successes"] / tool_data["calls"] * 100 if tool_data["calls"] > 0 else 0
            print(f"  {tool_name}: {tool_data['calls']} calls ({success_rate:.0f}% success)")

    if analysis["total_tokens"] > 0:
        print(f"\nTotal Tokens Used: {analysis['total_tokens']:,}")

    if analysis["errors"]:
        print(f"\nErrors ({len(analysis['errors'])}):")
        for error in analysis["errors"][:5]:  # Show first 5 errors
            print(f"  [{error['timestamp']}] {error['message'][:100]}")
        if len(analysis["errors"]) > 5:
            print(f"  ... and {len(analysis['errors']) - 5} more errors")

    print("\n" + "="*70)


# =============================================================================
# Environment Variables for CrewAI Logging
# =============================================================================

def set_crewai_logging_env_vars(console_verbose: bool = False):
    """
    Set environment variables for CrewAI logging.

    By default, this does NOT enable verbose console logging from LiteLLM
    or LangChain to keep the console clean. The trace files will still
    capture all details.

    Parameters
    ----------
    console_verbose : bool
        If True, also enables verbose console output from LiteLLM/LangChain.
        Default is False (file logging only, console stays clean).
    """
    if console_verbose:
        # Enable verbose mode for LiteLLM (used by CrewAI) - prints to console
        os.environ["LITELLM_LOG"] = "DEBUG"
        # Enable LangChain debugging - prints to console
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    else:
        # Suppress verbose console output from these libraries
        # They will still log to our file handlers
        os.environ["LITELLM_LOG"] = "ERROR"  # Only errors to console
        os.environ.pop("LANGCHAIN_TRACING_V2", None)  # Disable tracing
