# config/llm_config.py

from __future__ import annotations
import os
import logging
import time
from typing import Any, Dict, List, Optional, Union

import requests
from requests.exceptions import HTTPError

from crewai import LLM, BaseLLM

logger = logging.getLogger(__name__)


# ==============================================================================
# DATABRICKS / AIF GPT-5 LLM CONFIGURATION
# ==============================================================================

def _is_databricks_environment() -> bool:
    """
    Detect if we're running inside a Databricks environment.
    Checks for Databricks-specific environment variables and runtime.
    """
    # Check for common Databricks environment indicators
    databricks_indicators = [
        "DATABRICKS_RUNTIME_VERSION",
        "SPARK_HOME",
        "DB_HOME",
        "DATABRICKS_HOST",
    ]
    for indicator in databricks_indicators:
        if os.getenv(indicator):
            return True

    # Also check if dbutils is available
    try:
        # This import only works in Databricks
        from pyspark.dbutils import DBUtils  # noqa: F401
        return True
    except ImportError:
        pass

    return False


def _get_databricks_secret(scope: str, key: str) -> Optional[str]:
    """
    Safely get a secret from Databricks secret scope.
    Returns None if not in Databricks environment.
    """
    try:
        # Import dbutils - only available in Databricks
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        from pyspark.dbutils import DBUtils
        dbutils = DBUtils(spark)
        return dbutils.secrets.get(scope, key)
    except Exception as e:
        logger.warning(f"Could not get Databricks secret {scope}/{key}: {e}")
        return None


def _get_aif_access_token(
    url: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    scope: Optional[str] = None,
) -> str:
    """Obtain an AAD bearer token for the AIF scope."""
    if url is None:
        url = os.getenv("AIF_TOKEN_URL")
    if client_id is None:
        client_id = os.getenv("AIF_CLIENT_ID")
    if client_secret is None:
        client_secret = os.getenv("AIF_CLIENT_SECRET")
    if scope is None:
        scope = os.getenv("AIF_SCOPE")

    if not all([url, client_id, client_secret, scope]):
        raise RuntimeError(
            "Missing AIF credentials. Required env vars: "
            "AIF_TOKEN_URL, AIF_CLIENT_ID, AIF_CLIENT_SECRET, AIF_SCOPE"
        )

    response = requests.post(
        url,
        data={
            "grant_type": "client_credentials",
            "scope": scope,
        },
        auth=(client_id, client_secret),
    )
    response.raise_for_status()
    return response.json()["access_token"]


def _run_aif_model(
    url: str,
    token: str,
    subscription_key: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 50000,
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Call the AIF GPT-5 chat endpoint with OpenAI-style messages.

    Args:
        url: AIF model endpoint URL
        token: Bearer token for authentication
        subscription_key: APIM subscription key
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Maximum tokens in response
        tools: Optional list of tool definitions (OpenAI function calling format)
        temperature: Optional temperature setting

    Returns:
        API response as dict
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
    }

    # Add tools if provided (enables function calling)
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"  # Let the model decide when to use tools
        logger.debug(f"[AIF] Passing {len(tools)} tools to API")

    # Add temperature if provided
    if temperature is not None:
        payload["temperature"] = temperature

    logger.info(f"[_run_aif_model] Calling API: {url}")
    logger.info(f"[_run_aif_model] Payload keys: {payload.keys()}")
    logger.info(f"[_run_aif_model] Number of messages: {len(messages)}")
    if tools:
        logger.info(f"[_run_aif_model] Number of tools: {len(tools)}")

    response = requests.post(url, headers=headers, json=payload)

    logger.info(f"[_run_aif_model] Response status: {response.status_code}")
    logger.info(f"[_run_aif_model] Response length: {len(response.text)} chars")

    if response.status_code >= 400:
        logger.error(f"AIF error status: {response.status_code}")
        logger.error(f"AIF error body: {response.text}")

    response.raise_for_status()

    result = response.json()
    logger.info(f"[_run_aif_model] Parsed JSON keys: {result.keys() if result else 'None'}")
    return result


class AIFChatLLM(BaseLLM):
    """
    CrewAI-native custom LLM that wraps the AIF GPT-5 chat endpoint.
    Used for Databricks deployments with Azure AD OAuth2 authentication.

    IMPORTANT: This LLM handles tool calling internally because CrewAI doesn't
    pass tools/available_functions to custom BaseLLM implementations. Tools must
    be registered directly with the register_tool() method before running crews.

    Required environment variables:
        AIF_TOKEN_URL      - Azure AD token endpoint
        AIF_SCOPE          - OAuth2 scope
        AIF_MODEL_URL      - Azure API Management endpoint
        AIF_CLIENT_ID      - Service principal client ID
        AIF_CLIENT_SECRET  - Service principal secret
        AIF_SUBSCRIPTION_KEY - APIM subscription key
    """

    def __init__(
        self,
        model: str = "az_openai_gpt-5_chat",
        temperature: Optional[float] = 0.2,
        max_tokens: int = 50000,
    ):
        # BaseLLM requires a model string and common params
        super().__init__(model=model, temperature=temperature)

        self.max_tokens = max_tokens
        self.model_url: str = os.getenv("AIF_MODEL_URL") or ""
        self.subscription_key: str = os.getenv("AIF_SUBSCRIPTION_KEY") or ""
        self.token: Optional[str] = None

        # Tool registration - CrewAI doesn't pass tools to custom LLMs,
        # so we register them directly and handle function calling ourselves
        self._registered_tools: Dict[str, Any] = {}
        self._tool_schemas: List[Dict[str, Any]] = []

        # Initialize usage tracking attributes
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.successful_requests = 0
        self._last_call_usage: Dict[str, int] = {}
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        self._last_total_tokens = 0

        if not self.model_url or not self.subscription_key:
            raise RuntimeError(
                "AIF_MODEL_URL or AIF_SUBSCRIPTION_KEY is not set. "
                "Please configure these environment variables for Databricks LLM."
            )

    def register_tool(self, name: str, description: str, func: Any, parameters: Dict[str, Any]) -> None:
        """
        Register a tool that this LLM can call.

        Since CrewAI doesn't pass tools to custom BaseLLM.call(), we need to
        register tools directly with the LLM and handle function calling internally.

        Args:
            name: Tool name (e.g., "python_code_executor")
            description: Tool description for the LLM
            func: Callable to execute when tool is called
            parameters: JSON Schema for tool parameters
        """
        self._registered_tools[name] = func
        self._tool_schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        })
        logger.info(f"[AIFChatLLM] Registered tool: {name}")

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        **kwargs,  # CrewAI passes from_task, from_agent, response_model, etc.
    ) -> str:
        """
        Main entry point CrewAI uses. Returns a plain text string.

        Handles tool calling internally since CrewAI doesn't pass tools/available_functions
        to custom BaseLLM implementations. Uses registered tools from register_tool().

        Flow:
        1. Pass registered tools to GPT-5 API
        2. If GPT-5 returns tool_calls, execute them using registered functions
        3. Send tool results back to API
        4. Loop until we get a final text response
        """
        import json

        # Normalise messages to [{role, content}, ...]
        if isinstance(messages, str):
            msg_list: List[Dict[str, Any]] = [{"role": "user", "content": messages}]
        else:
            msg_list = list(messages)  # Make a copy to avoid modifying original

        # Lazy token fetch
        if self.token is None:
            self.token = _get_aif_access_token()

        # Use registered tools (CrewAI doesn't pass tools to custom LLMs)
        tools_to_use = self._tool_schemas if self._registered_tools else None
        funcs_to_use = self._registered_tools if self._registered_tools else None

        logger.info(f"[AIFChatLLM] call() - messages: {len(msg_list)}, registered_tools: {len(self._registered_tools)}")

        max_attempts = 3
        base_backoff = 2.0
        last_err: Optional[Exception] = None

        # Tool call loop - may need multiple iterations if model calls tools
        max_tool_iterations = 10  # Prevent infinite loops
        for iteration in range(max_tool_iterations):
            logger.info(f"[AIFChatLLM] Iteration {iteration + 1}/{max_tool_iterations}")

            for attempt in range(1, max_attempts + 1):
                try:
                    # Call API with registered tools
                    resp_json = _run_aif_model(
                        self.model_url,
                        self.token,
                        self.subscription_key,
                        msg_list,
                        max_tokens=self.max_tokens,
                        tools=tools_to_use,  # Pass registered tools
                    )

                    # Update usage tracking
                    self._update_usage(resp_json.get("usage", {}))

                    # Extract the response message
                    if not resp_json or "choices" not in resp_json or not resp_json["choices"]:
                        logger.error(f"[AIFChatLLM] Invalid API response: {resp_json}")
                        raise RuntimeError(f"Invalid API response - no choices: {resp_json}")

                    response_message = resp_json["choices"][0]["message"]
                    if not response_message:
                        logger.error(f"[AIFChatLLM] No message in choice: {resp_json['choices'][0]}")
                        raise RuntimeError(f"Invalid API response - no message in choice")

                    content = response_message.get("content", "") or ""
                    tool_calls = response_message.get("tool_calls", [])

                    logger.info(f"[AIFChatLLM] Content: {len(content)} chars, Tool calls: {len(tool_calls)}")

                    # ===== HANDLE TOOL CALLS =====
                    if tool_calls and funcs_to_use:
                        logger.info(f"[AIFChatLLM] Processing {len(tool_calls)} tool call(s)...")

                        # Add assistant's response (with tool_calls) to messages
                        msg_list.append(response_message)

                        # Process each tool call
                        for tool_call in tool_calls:
                            function_name = tool_call["function"]["name"]
                            function_args_str = tool_call["function"]["arguments"]
                            tool_call_id = tool_call["id"]

                            logger.info(f"[AIFChatLLM] Executing tool: {function_name}")

                            # Parse arguments
                            try:
                                function_args = json.loads(function_args_str)
                            except json.JSONDecodeError:
                                function_args = {}

                            # Execute the function
                            if function_name in funcs_to_use:
                                try:
                                    func = funcs_to_use[function_name]
                                    result = func(**function_args)
                                    logger.info(f"[AIFChatLLM] Tool {function_name} returned {len(str(result))} chars")
                                except Exception as e:
                                    result = f"Error executing {function_name}: {str(e)}"
                                    logger.error(f"[AIFChatLLM] Tool error: {result}")
                            else:
                                result = f"Function {function_name} not found in registered tools"
                                logger.warning(f"[AIFChatLLM] {result}")

                            # Add tool result to messages
                            msg_list.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": function_name,
                                "content": str(result),
                            })

                        # Continue the loop to get the model's response after tool execution
                        break  # Break from attempt loop, continue tool iteration loop

                    # ===== NO TOOL CALLS - RETURN CONTENT =====
                    return content

                except HTTPError as e:
                    status = e.response.status_code

                    # 401: token expired → refresh and retry
                    if status == 401 and attempt < max_attempts:
                        logger.warning("[AIFChatLLM] 401 Unauthorized. Refreshing token...")
                        self.token = _get_aif_access_token()
                        continue

                    # 429: rate limit → backoff
                    if status == 429 and attempt < max_attempts:
                        retry_after = e.response.headers.get("Retry-After")
                        try:
                            wait = float(retry_after)
                        except (TypeError, ValueError):
                            wait = base_backoff * attempt
                        logger.warning(
                            f"[AIFChatLLM] 429 Too Many Requests. Sleeping {wait:.1f}s "
                            f"(attempt {attempt}/{max_attempts})..."
                        )
                        time.sleep(wait)
                        continue

                    last_err = e
                    break

                except Exception as e:
                    last_err = e
                    break

            else:
                # All attempts exhausted without break - we got a final response
                # This shouldn't happen since we return inside the loop
                if last_err:
                    raise last_err
                continue  # Continue to next iteration

            # Check if we should continue (tool calls were processed)
            if not tool_calls:
                break  # No tool calls processed, we're done

        # If we get here after max_tool_iterations, raise error
        if last_err:
            raise last_err
        raise RuntimeError("Max tool iterations reached without final response")

    def _update_usage(self, usage: Dict[str, int]) -> None:
        """Update usage tracking from API response."""
        if usage:
            prompt_tok = usage.get("prompt_tokens", 0)
            completion_tok = usage.get("completion_tokens", 0)
            total_tok = usage.get("total_tokens", 0)

            self._last_call_usage = {
                "prompt_tokens": prompt_tok,
                "completion_tokens": completion_tok,
                "total_tokens": total_tok,
            }
            self._last_prompt_tokens = prompt_tok
            self._last_completion_tokens = completion_tok
            self._last_total_tokens = total_tok

            self.prompt_tokens += prompt_tok
            self.completion_tokens += completion_tok
            self.total_tokens += total_tok
            self.successful_requests += 1
        else:
            self._last_call_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            self._last_prompt_tokens = 0
            self._last_completion_tokens = 0
            self._last_total_tokens = 0
            self.successful_requests += 1

    def get_usage_metrics(self) -> Dict[str, int]:
        """Helper method to get current usage statistics."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "successful_requests": self.successful_requests,
        }

    def get_num_tokens(self, text: str) -> int:
        """
        Estimate token count for text. CrewAI may call this.
        Simple approximation: ~4 chars per token for English text.
        """
        return len(text) // 4

    def get_num_tokens_from_messages(self, messages: List[Dict[str, str]]) -> int:
        """Estimate token count for messages. CrewAI may call this."""
        total = 0
        for msg in messages:
            total += self.get_num_tokens(msg.get("content", ""))
        return total

    def reset_usage_metrics(self):
        """Reset usage counters (useful between different crew runs)."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.successful_requests = 0
        self._last_call_usage = {}
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        self._last_total_tokens = 0

    @property
    def last_call_usage(self) -> Dict[str, int]:
        """Property that CrewAI can check for usage from the last call."""
        return self._last_call_usage

    def _get_usage_from_last_call(self) -> Dict[str, int]:
        """Alternative method for CrewAI to get usage data."""
        return self._last_call_usage

    def get_token_usage(self) -> Dict[str, int]:
        """Explicit method for telemetry systems to get cumulative usage."""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "successful_requests": self.successful_requests,
        }

    # BaseLLM capabilities
    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 50000


def _setup_databricks_aif_environment():
    """
    Setup AIF environment variables for Databricks.
    This function sets up the required env vars, fetching secrets from Databricks if needed.

    Can be customized by setting environment variables before calling, or will use defaults.
    """
    # Set defaults if not already configured
    if not os.getenv("AIF_TOKEN_URL"):
        os.environ["AIF_TOKEN_URL"] = "https://login.microsoftonline.com/f66fae02-5d36-495b-bfe0-78a6ff9f8e6e/oauth2/v2.0/token"

    if not os.getenv("AIF_SCOPE"):
        os.environ["AIF_SCOPE"] = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"

    if not os.getenv("AIF_MODEL_URL"):
        os.environ["AIF_MODEL_URL"] = "https://bnlwe-ai03-q-931039-apim-01.azure-api.net/openai5/az_openai_gpt-5_chat"

    if not os.getenv("AIF_CLIENT_ID"):
        os.environ["AIF_CLIENT_ID"] = "c6d73674-8d75-47c0-9f78-927c687ce20a"

    if not os.getenv("AIF_SUBSCRIPTION_KEY"):
        os.environ["AIF_SUBSCRIPTION_KEY"] = "6e4956c3b8854fb191a43ea4b7a0c063"

    # For client secret, try to get from Databricks secrets if not already set
    if not os.getenv("AIF_CLIENT_SECRET"):
        if _is_databricks_environment():
            secret = _get_databricks_secret(
                "scope-databrickskv",
                "svc-b-da-d-931272-ina-aadprincipal"
            )
            if secret:
                os.environ["AIF_CLIENT_SECRET"] = secret
            else:
                logger.warning(
                    "AIF_CLIENT_SECRET not set and could not fetch from Databricks secrets. "
                    "Please set AIF_CLIENT_SECRET environment variable."
                )
        else:
            logger.warning(
                "AIF_CLIENT_SECRET not set. Please set this environment variable "
                "for Databricks/AIF authentication."
            )


def register_code_execution_tool(llm: AIFChatLLM) -> None:
    """
    Register the CodeExecutionTool with an AIFChatLLM instance.

    This is required because CrewAI doesn't pass tools to custom BaseLLM implementations.
    We need to register tools directly with the LLM so it can handle function calling internally.

    The CodeExecutionTool instance is stored on `llm._code_execution_tool` so that
    crew runners can set `protected_paths` before kicking off a crew, preventing
    LLM agents from overwriting upstream pipeline output files.

    Args:
        llm: An AIFChatLLM instance to register the tool with
    """
    from utils.code_execution_tool import CodeExecutionTool

    # Create a code execution tool instance and store it on the LLM
    # so crew runners can update protected_paths before each crew run
    code_tool = CodeExecutionTool()
    llm._code_execution_tool = code_tool

    # Register the tool with the LLM
    llm.register_tool(
        name="python_code_executor",
        description=(
            "Executes Python code in a controlled local environment. "
            "Returns stdout, `result` variable, and traceback on error. "
            "Use this to run any Python code for data analysis, file operations, etc."
        ),
        func=code_tool._run,
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Write complete, runnable code."
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory to run the code in."
                }
            },
            "required": ["code"]
        }
    )
    logger.info("[register_code_execution_tool] CodeExecutionTool registered with LLM")


def create_databricks_aif_llm(register_tools: bool = True) -> AIFChatLLM:
    """
    Create an AIFChatLLM instance for Databricks deployment.
    Sets up required environment and returns the LLM with tools registered.

    IMPORTANT: This automatically registers the CodeExecutionTool because CrewAI
    doesn't pass tools to custom BaseLLM implementations. The LLM handles
    function calling internally using registered tools.

    Args:
        register_tools: If True (default), register CodeExecutionTool with the LLM.
                       Set to False if you want to register tools manually.

    Environment variables (set before calling or use defaults):
        AIF_TOKEN_URL      - Azure AD token endpoint
        AIF_SCOPE          - OAuth2 scope
        AIF_MODEL_URL      - Azure API Management endpoint
        AIF_CLIENT_ID      - Service principal client ID
        AIF_CLIENT_SECRET  - Service principal secret (required, fetched from Databricks secrets if available)
        AIF_SUBSCRIPTION_KEY - APIM subscription key
    """
    _setup_databricks_aif_environment()

    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "50000"))

    llm = AIFChatLLM(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    logger.info(f"Created Databricks AIF LLM with endpoint: {llm.model_url}")
    logger.info(f"Temperature: {temperature}, Max Tokens: {max_tokens}")

    # Register CodeExecutionTool so agents can execute Python code
    if register_tools:
        register_code_execution_tool(llm)

    return llm


# ==============================================================================
# AWS BEDROCK LLM CONFIGURATION
# ==============================================================================


def create_bedrock_claude_llm() -> LLM:
    """
    Returns a real CrewAI LLM object pointing to Claude Sonnet on Bedrock
    using LiteLLM under the hood.

    You MUST have the following env vars set:

        AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY
        AWS_REGION   (or AWS_DEFAULT_REGION)

    And optionally:

        BEDROCK_MODEL_ID   (e.g. "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
        LLM_TEMPERATURE
        LLM_MAX_TOKENS

    CrewAI 1.7.2+ uses LiteLLM for model routing. For AWS Bedrock, the model ID
    format is: "bedrock/<model-id>"

    Claude Sonnet 4.5 Model IDs (cross-region inference):
      - eu.anthropic.claude-sonnet-4-5-20250929-v1:0   (EU cross-region)
      - us.anthropic.claude-sonnet-4-5-20250929-v1:0   (US cross-region)
      - global.anthropic.claude-sonnet-4-5-20250929-v1:0 (Global cross-region)

    Claude 3.5 Sonnet Model IDs (fallback):
      - anthropic.claude-3-5-sonnet-20241022-v2:0  (Claude 3.5 Sonnet v2)
      - eu.anthropic.claude-3-5-sonnet-20241022-v2:0 (EU cross-region)
      - us.anthropic.claude-3-5-sonnet-20241022-v2:0 (US cross-region)

    IMPORTANT: For cross-region inference profiles:
      - LiteLLM uses bedrock/converse/ endpoint by default for Claude models
      - Make sure you have enabled model access in AWS Bedrock console
      - Anthropic models require one-time First-Time Usage form submission
    """
    # Get the region for potential cross-region prefix
    region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-west-1"))

    # Default model ID - Claude Sonnet 4.5 (cross-region EU)
    # Change to us. prefix if using US region, or global. for global routing
    default_model = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"

    # Get model ID from environment, with sensible default
    raw_model_id = os.getenv("BEDROCK_MODEL_ID", default_model)

    if raw_model_id:
        # User provided a model ID
        if raw_model_id.startswith("bedrock/"):
            model_id = raw_model_id
        else:
            # Prepend bedrock/ if not present
            model_id = f"bedrock/{raw_model_id}"
    else:
        # Use default with bedrock/ prefix
        model_id = f"bedrock/{default_model}"

    temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    # 16000 tokens needed for agents generating extensive code
    # (8000 was causing truncation in Segmentation/Feature crews)
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "60000"))

    # Log the model being used for debugging
    logger.info(f"Creating Bedrock LLM with model: {model_id}")
    logger.info(f"AWS Region: {region}")
    logger.info(f"Temperature: {temperature}, Max Tokens: {max_tokens}")

    # This is the CREWAI LLM object
    # IMPORTANT: CrewAI 1.8.0+ has native Bedrock support. When it detects "bedrock/" prefix,
    # it uses its own Bedrock client instead of LiteLLM. We must explicitly set the region
    # to match the cross-region inference profile (eu. prefix requires eu-west-1).
    llm = LLM(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        region_name=region,  # Explicitly set region for native Bedrock client
    )

    return llm


# ==============================================================================
# UNIFIED LLM FACTORY - CONFIG-DRIVEN PROVIDER SELECTION
# ==============================================================================

# Global cache for the LLM provider setting (read from config once)
_cached_llm_provider: Optional[str] = None


def _get_llm_provider_from_config(config_path: Optional[str] = None) -> str:
    """
    Read llm_provider from YAML config file.

    Args:
        config_path: Path to config YAML file. If None, tries default locations.

    Returns:
        LLM provider string ('bedrock' or 'databricks')
    """
    import yaml
    from pathlib import Path

    # Try to find the config file
    if config_path:
        yaml_path = Path(config_path).expanduser().resolve()
    else:
        # Try default locations
        possible_paths = [
            Path(__file__).parent / "config.yaml",  # config/config.yaml
            Path.cwd() / "config" / "config.yaml",  # ./config/config.yaml
            Path.cwd() / "config.yaml",  # ./config.yaml
        ]
        yaml_path = None
        for p in possible_paths:
            if p.exists():
                yaml_path = p
                break

    if yaml_path and yaml_path.exists():
        try:
            with yaml_path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            provider = raw.get("llm_provider", "bedrock")
            logger.info(f"Read llm_provider='{provider}' from {yaml_path}")
            return provider.lower().strip()
        except Exception as e:
            logger.warning(f"Could not read llm_provider from {yaml_path}: {e}")

    # Fallback to environment variable
    provider = os.getenv("LLM_PROVIDER", "bedrock")
    logger.info(f"Using llm_provider='{provider}' from environment variable (config file not found)")
    return provider.lower().strip()


def set_llm_provider(provider: str) -> None:
    """
    Explicitly set the LLM provider (useful for programmatic control).

    Args:
        provider: 'bedrock' or 'databricks'
    """
    global _cached_llm_provider
    _cached_llm_provider = provider.lower().strip()
    logger.info(f"LLM provider explicitly set to: {_cached_llm_provider}")


def get_llm(config_path: Optional[str] = None) -> Union[LLM, AIFChatLLM]:
    """
    Unified LLM factory function that returns the appropriate LLM based on configuration.

    This is the recommended entry point for getting an LLM instance. It automatically
    selects the correct provider based on the llm_provider setting in config.yaml.

    Provider Selection Priority:
        1. Explicitly set via set_llm_provider() (cached)
        2. Read from config.yaml (llm_provider field at top of file)
        3. Environment variable LLM_PROVIDER (fallback)
        4. Default: 'bedrock'

    Args:
        config_path: Optional path to config.yaml. If not provided, searches
                     default locations (config/config.yaml, ./config.yaml).

    For Bedrock (AWS):
        - Uses create_bedrock_claude_llm()
        - Requires AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)

    For Databricks (AIF GPT-5):
        - Uses create_databricks_aif_llm()
        - Requires AIF credentials (AIF_TOKEN_URL, AIF_CLIENT_ID, etc.)

    Returns:
        LLM instance compatible with CrewAI agents

    Example:
        # In config.yaml:
        # llm_provider: "bedrock"  # or "databricks"

        # In your code:
        from config.llm_config import get_llm
        llm = get_llm()  # Reads from config.yaml

        # Or with explicit config path:
        llm = get_llm(config_path="config/config_uk_fabric.yaml")

        # Or programmatic override:
        from config.llm_config import set_llm_provider, get_llm
        set_llm_provider("databricks")
        llm = get_llm()
    """
    global _cached_llm_provider

    # Use cached value if set programmatically
    if _cached_llm_provider is not None:
        provider = _cached_llm_provider
    else:
        provider = _get_llm_provider_from_config(config_path)

    logger.info(f"LLM Provider selected: {provider}")

    if provider == "databricks":
        logger.info("Creating Databricks AIF LLM...")
        return create_databricks_aif_llm()
    elif provider == "bedrock":
        logger.info("Creating AWS Bedrock Claude LLM...")
        return create_bedrock_claude_llm()
    else:
        logger.warning(
            f"Unknown llm_provider '{provider}'. "
            f"Valid options are 'bedrock' or 'databricks'. Defaulting to 'bedrock'."
        )
        return create_bedrock_claude_llm()


# Backward compatibility: also expose get_shared_llm as an alias
# This matches the function name from the original databricks config
def get_shared_llm(config_path: Optional[str] = None) -> Union[LLM, AIFChatLLM]:
    """Alias for get_llm() for backward compatibility."""
    return get_llm(config_path)
