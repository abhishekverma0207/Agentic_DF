# utils/cost_tracking.py
"""
Token and Cost Tracking Utility for CrewAI
==========================================

This module provides utilities to track token usage and calculate costs
for LLM API calls made during crew execution.

IMPORTANT: This module uses LiteLLM callbacks to intercept token usage
at the source, bypassing CrewAI's limitations in exposing token data.

Features:
- Track input/output tokens per crew via LiteLLM callbacks
- Calculate costs based on model pricing
- Save cost reports as JSON artifacts
- Aggregate costs across the entire pipeline
- Works with ANY LLM provider (Bedrock, OpenAI, Azure, custom LLMs)

Usage:
    from utils.cost_tracking import (
        get_cost_tracker,
        setup_litellm_callbacks,
        cleanup_litellm_callbacks
    )

    # Setup callbacks BEFORE creating crews
    setup_litellm_callbacks()

    # Get the global tracker
    tracker = get_cost_tracker()

    # Start tracking a crew
    tracker.start_crew("EDA Crew")

    # ... crew executes ...

    # End tracking and save report
    tracker.end_crew("EDA Crew", output_dir="/path/to/eda_output")

    # When done with all crews
    cleanup_litellm_callbacks()
"""

from __future__ import annotations

import json
import os
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Thread lock for thread-safe token recording
_token_lock = threading.Lock()


# =============================================================================
# Model Pricing (per 1M tokens as of Dec 2024)
# =============================================================================

# AWS Bedrock Claude pricing (US regions)
# https://aws.amazon.com/bedrock/pricing/
MODEL_PRICING = {
    # Claude Sonnet 4.5 (latest)
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
    },
    "anthropic.claude-sonnet-4-5-20250929-v1:0": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
    },
    "claude-sonnet-4-5-20250929": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
    },
    # Claude 3.5 Sonnet
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
    },
    "anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
    },
    "claude-3-5-sonnet-20241022": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
    },
    # Claude 3.5 Haiku
    "anthropic.claude-3-5-haiku-20241022-v1:0": {
        "input_per_1m": 0.80,
        "output_per_1m": 4.00,
    },
    "claude-3-5-haiku-20241022": {
        "input_per_1m": 0.80,
        "output_per_1m": 4.00,
    },
    # Claude 3 Opus
    "anthropic.claude-3-opus-20240229-v1:0": {
        "input_per_1m": 15.00,
        "output_per_1m": 75.00,
    },
    "claude-3-opus-20240229": {
        "input_per_1m": 15.00,
        "output_per_1m": 75.00,
    },
    # Claude 3 Sonnet
    "anthropic.claude-3-sonnet-20240229-v1:0": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
    },
    # Claude 3 Haiku
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "input_per_1m": 0.25,
        "output_per_1m": 1.25,
    },
    # OpenAI models (for future flexibility)
    "gpt-4-turbo": {
        "input_per_1m": 10.00,
        "output_per_1m": 30.00,
    },
    "gpt-4o": {
        "input_per_1m": 2.50,
        "output_per_1m": 10.00,
    },
    "gpt-4o-mini": {
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
    },
    # Default fallback (assume Claude 3.5 Sonnet pricing)
    "default": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
    },
}


def get_model_pricing(model_id: str) -> Dict[str, float]:
    """Get pricing for a specific model."""
    # Try exact match first
    if model_id in MODEL_PRICING:
        return MODEL_PRICING[model_id]

    # Try partial match
    for key in MODEL_PRICING:
        if key in model_id or model_id in key:
            return MODEL_PRICING[key]

    # Return default
    return MODEL_PRICING["default"]


def calculate_cost(input_tokens: int, output_tokens: int, model_id: str = "default") -> float:
    """Calculate cost in USD for given token counts."""
    pricing = get_model_pricing(model_id)
    input_cost = (input_tokens / 1_000_000) * pricing["input_per_1m"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_1m"]
    return input_cost + output_cost


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LLMCall:
    """Record of a single LLM API call."""
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    agent_name: Optional[str] = None
    task_name: Optional[str] = None


@dataclass
class CrewCostReport:
    """Cost report for a single crew execution."""
    crew_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    model: str
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    llm_call_count: int
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "crew_name": self.crew_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "model": self.model,
            "tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "total": self.total_tokens,
            },
            "cost_usd": round(self.total_cost_usd, 6),
            "llm_call_count": self.llm_call_count,
            "calls": self.calls,
        }


@dataclass
class PipelineCostReport:
    """Aggregated cost report for the entire pipeline."""
    start_time: str
    end_time: str
    duration_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    crew_count: int
    crews: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pipeline_summary": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration_seconds": round(self.duration_seconds, 2),
                "duration_minutes": round(self.duration_seconds / 60, 2),
            },
            "total_tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "total": self.total_tokens,
            },
            "total_cost_usd": round(self.total_cost_usd, 6),
            "crew_count": self.crew_count,
            "crews": self.crews,
        }


# =============================================================================
# Cost Tracker Class
# =============================================================================

class CostTracker:
    """
    Tracks token usage and costs across crew executions.

    This is a singleton class that maintains state across the entire pipeline.
    """

    _instance: Optional["CostTracker"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._initialized = True
        self.pipeline_start_time: Optional[datetime] = None
        self.current_crew: Optional[str] = None
        self.current_crew_start: Optional[datetime] = None
        self.current_crew_calls: List[LLMCall] = []
        self.crew_reports: List[CrewCostReport] = []
        self.model_id: str = "default"

    def reset(self):
        """Reset all tracking state."""
        self.pipeline_start_time = None
        self.current_crew = None
        self.current_crew_start = None
        self.current_crew_calls = []
        self.crew_reports = []

    def start_pipeline(self):
        """Mark the start of the pipeline."""
        self.reset()
        self.pipeline_start_time = datetime.now()
        logger.info("Cost tracking: Pipeline started")

    def set_model(self, model_id: str):
        """Set the model ID for cost calculations."""
        self.model_id = model_id
        logger.info(f"Cost tracking: Model set to {model_id}")

    def start_crew(self, crew_name: str):
        """Start tracking a new crew."""
        self.current_crew = crew_name
        self.current_crew_start = datetime.now()
        self.current_crew_calls = []
        logger.info(f"Cost tracking: Started crew '{crew_name}'")

    def record_llm_call(
        self,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None,
        agent_name: Optional[str] = None,
        task_name: Optional[str] = None,
    ):
        """Record a single LLM API call."""
        model = model or self.model_id
        total_tokens = input_tokens + output_tokens
        cost = calculate_cost(input_tokens, output_tokens, model)

        call = LLMCall(
            timestamp=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            agent_name=agent_name,
            task_name=task_name,
        )
        self.current_crew_calls.append(call)

    def end_crew(self, crew_name: str, output_dir: str) -> CrewCostReport:
        """
        End tracking for a crew and save the cost report.

        Parameters
        ----------
        crew_name : str
            Name of the crew
        output_dir : str
            Directory to save the cost report JSON

        Returns
        -------
        CrewCostReport
            The cost report for this crew
        """
        end_time = datetime.now()
        start_time = self.current_crew_start or end_time

        # Aggregate totals
        total_input = sum(c.input_tokens for c in self.current_crew_calls)
        total_output = sum(c.output_tokens for c in self.current_crew_calls)
        total_tokens = total_input + total_output
        total_cost = sum(c.cost_usd for c in self.current_crew_calls)

        report = CrewCostReport(
            crew_name=crew_name,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=(end_time - start_time).total_seconds(),
            model=self.model_id,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            llm_call_count=len(self.current_crew_calls),
            calls=[asdict(c) for c in self.current_crew_calls],
        )

        self.crew_reports.append(report)

        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        cost_file = os.path.join(output_dir, f"{crew_name.lower().replace(' ', '_')}_cost.json")
        with open(cost_file, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Cost tracking: Crew '{crew_name}' - {total_tokens:,} tokens, ${total_cost:.4f}")
        logger.info(f"Cost report saved to: {cost_file}")

        # Reset current crew state
        self.current_crew = None
        self.current_crew_start = None
        self.current_crew_calls = []

        return report

    def get_pipeline_report(self, output_dir: Optional[str] = None) -> PipelineCostReport:
        """
        Get the aggregated pipeline cost report.

        Parameters
        ----------
        output_dir : Optional[str]
            If provided, saves the report to this directory

        Returns
        -------
        PipelineCostReport
            The aggregated cost report
        """
        end_time = datetime.now()
        start_time = self.pipeline_start_time or end_time

        total_input = sum(r.total_input_tokens for r in self.crew_reports)
        total_output = sum(r.total_output_tokens for r in self.crew_reports)
        total_tokens = total_input + total_output
        total_cost = sum(r.total_cost_usd for r in self.crew_reports)

        report = PipelineCostReport(
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=(end_time - start_time).total_seconds(),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            crew_count=len(self.crew_reports),
            crews=[r.to_dict() for r in self.crew_reports],
        )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            cost_file = os.path.join(output_dir, "pipeline_cost_summary.json")
            with open(cost_file, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info(f"Pipeline cost summary saved to: {cost_file}")

        return report

    def print_summary(self):
        """Print a summary of costs to the console."""
        if not self.crew_reports:
            print("No cost data recorded.")
            return

        total_input = sum(r.total_input_tokens for r in self.crew_reports)
        total_output = sum(r.total_output_tokens for r in self.crew_reports)
        total_tokens = total_input + total_output
        total_cost = sum(r.total_cost_usd for r in self.crew_reports)

        print("\n" + "=" * 70)
        print("PIPELINE COST SUMMARY")
        print("=" * 70)
        print(f"\n{'Crew':<35} {'Input':>12} {'Output':>12} {'Cost':>10}")
        print("-" * 70)

        for r in self.crew_reports:
            print(f"{r.crew_name:<35} {r.total_input_tokens:>12,} {r.total_output_tokens:>12,} ${r.total_cost_usd:>8.4f}")

        print("-" * 70)
        print(f"{'TOTAL':<35} {total_input:>12,} {total_output:>12,} ${total_cost:>8.4f}")
        print("=" * 70)
        print(f"\nTotal Tokens: {total_tokens:,}")
        print(f"Total Cost: ${total_cost:.4f}")
        print("=" * 70 + "\n")


# =============================================================================
# Global Tracker Instance
# =============================================================================

_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


# =============================================================================
# CrewAI Integration - Token Callback
# =============================================================================

def create_token_callback(crew_name: str):
    """
    Create a callback function for tracking tokens from CrewAI/LiteLLM.

    This can be used with LiteLLM's success_callback or similar mechanisms.
    """
    tracker = get_cost_tracker()

    def callback(kwargs, completion_response, start_time, end_time):
        """Callback invoked after each LLM completion."""
        try:
            usage = completion_response.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            model = kwargs.get("model", "unknown")

            tracker.record_llm_call(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
            )
        except Exception as e:
            logger.warning(f"Failed to record token usage: {e}")

    return callback


def extract_tokens_from_crew_result(crew_result: Any) -> Dict[str, int]:
    """
    Extract token usage from a CrewAI crew result.

    CrewAI may store token usage in different places depending on version.
    This function tries multiple approaches.
    """
    tokens = {"input": 0, "output": 0, "total": 0}

    try:
        # Try to get usage from crew result attributes
        if hasattr(crew_result, "token_usage"):
            usage = crew_result.token_usage
            if isinstance(usage, dict):
                tokens["input"] = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                tokens["output"] = usage.get("completion_tokens", usage.get("output_tokens", 0))
                tokens["total"] = tokens["input"] + tokens["output"]

        # Try tasks_output if available
        elif hasattr(crew_result, "tasks_output"):
            for task_output in crew_result.tasks_output:
                if hasattr(task_output, "token_usage"):
                    usage = task_output.token_usage
                    if isinstance(usage, dict):
                        tokens["input"] += usage.get("prompt_tokens", usage.get("input_tokens", 0))
                        tokens["output"] += usage.get("completion_tokens", usage.get("output_tokens", 0))
            tokens["total"] = tokens["input"] + tokens["output"]

        # Try raw attribute
        elif hasattr(crew_result, "raw"):
            raw = crew_result.raw
            if isinstance(raw, dict) and "usage" in raw:
                usage = raw["usage"]
                tokens["input"] = usage.get("prompt_tokens", 0)
                tokens["output"] = usage.get("completion_tokens", 0)
                tokens["total"] = tokens["input"] + tokens["output"]

    except Exception as e:
        logger.warning(f"Could not extract tokens from crew result: {e}")

    return tokens


def record_crew_tokens(
    crew_name: str,
    crew_result: Any,
    output_dir: str,
    model_id: str = "default",
) -> Dict[str, Any]:
    """
    Record token usage from a completed crew and save cost report.

    Parameters
    ----------
    crew_name : str
        Name of the crew
    crew_result : Any
        The result returned by crew.kickoff()
    output_dir : str
        Directory to save the cost report
    model_id : str
        Model ID for cost calculation

    Returns
    -------
    Dict[str, Any]
        Cost report dictionary
    """
    tracker = get_cost_tracker()
    tracker.set_model(model_id)

    # Extract tokens from crew result
    tokens = extract_tokens_from_crew_result(crew_result)

    # If we got tokens, record them
    if tokens["total"] > 0:
        tracker.record_llm_call(
            input_tokens=tokens["input"],
            output_tokens=tokens["output"],
            model=model_id,
        )

    # Generate and save report
    report = tracker.end_crew(crew_name, output_dir)
    return report.to_dict()


# =============================================================================
# LiteLLM Callback Integration - Universal Token Tracking
# =============================================================================

# Global reference to our callback handler (needed for cleanup)
_litellm_callback_handler = None
_callbacks_initialized = False


class LiteLLMTokenTracker:
    """
    Custom LiteLLM callback handler that intercepts ALL LLM calls
    and records token usage to our CostTracker.

    This works with ANY LLM provider that LiteLLM supports:
    - AWS Bedrock (Claude, Titan, etc.)
    - OpenAI (GPT-4, GPT-3.5, etc.)
    - Azure OpenAI
    - Anthropic Direct API
    - Google Vertex AI
    - Custom/local LLMs
    """

    def __init__(self):
        self.tracker = get_cost_tracker()

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Called by LiteLLM after EVERY successful LLM completion.
        This is where we capture token usage.
        """
        try:
            with _token_lock:
                # Extract model name (normalize from different formats)
                model = self._extract_model_name(kwargs, response_obj)

                # Extract token usage from response
                input_tokens, output_tokens = self._extract_token_usage(
                    kwargs, response_obj
                )

                # Record the call
                if input_tokens > 0 or output_tokens > 0:
                    self.tracker.record_llm_call(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        model=model,
                    )
                    logger.debug(
                        f"LiteLLM callback: Recorded {input_tokens} input, "
                        f"{output_tokens} output tokens for model {model}"
                    )
        except Exception as e:
            logger.warning(f"LiteLLM callback error: {e}")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Async version of log_success_event for async LLM calls."""
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Called when an LLM call fails."""
        logger.debug(f"LiteLLM call failed: {kwargs.get('model', 'unknown')}")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Async version of log_failure_event."""
        self.log_failure_event(kwargs, response_obj, start_time, end_time)

    def _extract_model_name(self, kwargs: Dict, response_obj: Any) -> str:
        """Extract and normalize the model name from various sources."""
        model = "unknown"

        # Try kwargs first (most reliable)
        if "model" in kwargs:
            model = kwargs["model"]
        # Try response object
        elif hasattr(response_obj, "model"):
            model = response_obj.model
        # Try litellm_params
        elif "litellm_params" in kwargs:
            model = kwargs["litellm_params"].get("model", model)

        # Normalize bedrock model names (remove bedrock/ prefix for pricing lookup)
        if model.startswith("bedrock/"):
            model = model[8:]  # Remove "bedrock/" prefix

        return model

    def _extract_token_usage(
        self, kwargs: Dict, response_obj: Any
    ) -> tuple[int, int]:
        """
        Extract token usage from the response.
        Returns (input_tokens, output_tokens).
        """
        input_tokens = 0
        output_tokens = 0

        # Method 1: Try response object's usage attribute (most common)
        if hasattr(response_obj, "usage"):
            usage = response_obj.usage
            if usage:
                # Handle both dict and object forms
                if isinstance(usage, dict):
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)
                else:
                    input_tokens = getattr(usage, "prompt_tokens", 0)
                    output_tokens = getattr(usage, "completion_tokens", 0)

        # Method 2: Try kwargs response_cost (LiteLLM calculates this)
        if input_tokens == 0 and output_tokens == 0:
            if "response_cost" in kwargs:
                # We have cost but not tokens, estimate from cost
                # This is a fallback, not ideal but better than nothing
                logger.debug("Token count not available, only cost recorded")

        # Method 3: Try to extract from streaming response
        if input_tokens == 0 and output_tokens == 0:
            if hasattr(response_obj, "_hidden_params"):
                hidden = response_obj._hidden_params
                if isinstance(hidden, dict) and "usage" in hidden:
                    usage = hidden["usage"]
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)

        return input_tokens, output_tokens


def setup_litellm_callbacks() -> bool:
    """
    Setup LiteLLM callbacks to intercept ALL LLM calls for token tracking.

    MUST be called BEFORE creating any CrewAI crews.

    Returns True if setup was successful, False otherwise.

    Usage:
        from utils.cost_tracking import setup_litellm_callbacks

        # Call this at the start of your pipeline
        setup_litellm_callbacks()

        # Then create and run your crews...
    """
    global _litellm_callback_handler, _callbacks_initialized

    if _callbacks_initialized:
        logger.info("LiteLLM callbacks already initialized")
        return True

    try:
        import litellm

        # Create our callback handler
        _litellm_callback_handler = LiteLLMTokenTracker()

        # Register with LiteLLM
        # LiteLLM supports multiple callback patterns:
        # 1. litellm.callbacks = [handler] for class-based handlers
        # 2. litellm.success_callback = [func] for function callbacks

        # Initialize callbacks list if needed
        if not hasattr(litellm, "callbacks") or litellm.callbacks is None:
            litellm.callbacks = []

        # Add our handler (avoid duplicates)
        if _litellm_callback_handler not in litellm.callbacks:
            litellm.callbacks.append(_litellm_callback_handler)

        _callbacks_initialized = True
        logger.info("LiteLLM token tracking callbacks initialized successfully")
        return True

    except ImportError:
        logger.warning(
            "LiteLLM not installed. Token tracking via callbacks not available. "
            "Install with: pip install litellm"
        )
        return False
    except Exception as e:
        logger.error(f"Failed to setup LiteLLM callbacks: {e}")
        return False


def cleanup_litellm_callbacks():
    """
    Remove our callback from LiteLLM.

    Call this when done with all crews to clean up.
    """
    global _litellm_callback_handler, _callbacks_initialized

    if not _callbacks_initialized:
        return

    try:
        import litellm

        if (
            hasattr(litellm, "callbacks")
            and litellm.callbacks
            and _litellm_callback_handler in litellm.callbacks
        ):
            litellm.callbacks.remove(_litellm_callback_handler)

        _litellm_callback_handler = None
        _callbacks_initialized = False
        logger.info("LiteLLM token tracking callbacks cleaned up")

    except Exception as e:
        logger.warning(f"Error cleaning up LiteLLM callbacks: {e}")


def is_token_tracking_active() -> bool:
    """Check if LiteLLM token tracking is currently active."""
    return _callbacks_initialized


# =============================================================================
# Context Manager for Easy Token Tracking
# =============================================================================

class TokenTrackingContext:
    """
    Context manager for easy token tracking of crew executions.

    Usage:
        with TokenTrackingContext("EDA Crew", output_dir="/path/to/output") as tracker:
            # Run your crew
            result = crew.kickoff()
        # Report is automatically saved when exiting context
    """

    def __init__(self, crew_name: str, output_dir: str, model_id: str = "default"):
        self.crew_name = crew_name
        self.output_dir = output_dir
        self.model_id = model_id
        self.tracker = get_cost_tracker()
        self.report = None

    def __enter__(self):
        # Ensure LiteLLM callbacks are set up
        setup_litellm_callbacks()

        # Start tracking
        self.tracker.set_model(self.model_id)
        self.tracker.start_crew(self.crew_name)
        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        # End tracking and save report
        self.report = self.tracker.end_crew(self.crew_name, self.output_dir)

        # Don't suppress exceptions
        return False

    def get_report(self) -> Optional[CrewCostReport]:
        """Get the cost report (available after exiting context)."""
        return self.report
