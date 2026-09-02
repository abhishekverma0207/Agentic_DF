# Databricks notebook: modules/00_llm_aif_gpt5

import os
import json
import time
from typing import Any, Dict, List, Optional, Union

import requests
from requests.exceptions import HTTPError

from crewai import BaseLLM

# --------------------------------------------------
# 1) Environment / endpoint configuration
# --------------------------------------------------

# These are your AIF / Azure AD endpoints and model URL
os.environ["AIF_TOKEN_URL"] = "https://login.microsoftonline.com/f66fae02-5d36-495b-bfe0-78a6ff9f8e6e/oauth2/v2.0/token"
os.environ["AIF_SCOPE"] = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"
os.environ["AIF_MODEL_URL"] = "https://bnlwe-ai03-q-931039-apim-01.azure-api.net/openai5/az_openai_gpt-54_chat"

# Client credentials for OAuth2 token
os.environ["AIF_CLIENT_ID"] = "c6d73674-8d75-47c0-9f78-927c687ce20a"
os.environ["AIF_CLIENT_SECRET"] = dbutils.secrets.get(
    "scope-databrickskv",
    "svc-b-da-d-931272-ina-aadprincipal",
)

# APIM subscription key (you may want to move this to a secret too)
os.environ["AIF_SUBSCRIPTION_KEY"] = "6e4956c3b8854fb191a43ea4b7a0c063"

print("MODEL_URL:", os.getenv("AIF_MODEL_URL"))


# --------------------------------------------------
# 2) Token + raw HTTP helper
# --------------------------------------------------

def get_access_token(
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


def run_model(
    url: str,
    token: str,
    subscription_key: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 50000,
) -> Dict[str, Any]:
    """
    Call the AIF GPT-5 chat endpoint with OpenAI-style messages.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Content-Type": "application/json",
    }
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code >= 400:
        print("AIF error status:", response.status_code)
        print("AIF error body:", response.text)

    response.raise_for_status()
    return response.json()


# --------------------------------------------------
# 3) CrewAI-compatible LLM wrapper
# --------------------------------------------------

class AIFChatLLM(BaseLLM):
    """
    CrewAI-native custom LLM that wraps your AIF GPT-5 chat endpoint.
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

        # Initialize usage tracking attributes
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.successful_requests = 0
        self._last_call_usage = {}
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        self._last_total_tokens = 0

        if not self.model_url or not self.subscription_key:
            raise RuntimeError("AIF_MODEL_URL or AIF_SUBSCRIPTION_KEY is not set")

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
        """
        # Normalise messages to [{role, content}, ...]
        if isinstance(messages, str):
            msg_list: List[Dict[str, str]] = [{"role": "user", "content": messages}]
        else:
            msg_list = messages

        # Lazy token fetch
        if self.token is None:
            self.token = get_access_token()

        max_attempts = 3
        base_backoff = 2.0
        last_err: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                resp_json = run_model(
                    self.model_url,
                    self.token,
                    self.subscription_key,
                    msg_list,
                    max_tokens=self.max_tokens,
                )
                
                # Extract content
                content = resp_json["choices"][0]["message"]["content"]
                
                # ===== EXTRACT AND REPORT USAGE TO CREWAI =====
                usage = resp_json.get("usage", {})
                if usage:
                    prompt_tok = usage.get("prompt_tokens", 0)
                    completion_tok = usage.get("completion_tokens", 0)
                    total_tok = usage.get("total_tokens", 0)
                    
                    # Store the last call's usage (CrewAI checks this)
                    self._last_call_usage = {
                        "prompt_tokens": prompt_tok,
                        "completion_tokens": completion_tok,
                        "total_tokens": total_tok,
                    }
                    
                    # Set individual attributes that CrewAI telemetry might check
                    self._last_prompt_tokens = prompt_tok
                    self._last_completion_tokens = completion_tok
                    self._last_total_tokens = total_tok
                    
                    # Also accumulate for our own tracking
                    self.prompt_tokens += prompt_tok
                    self.completion_tokens += completion_tok
                    self.total_tokens += total_tok
                    self.successful_requests += 1
                else:
                    # Fallback if no usage returned
                    self._last_call_usage = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                    self._last_prompt_tokens = 0
                    self._last_completion_tokens = 0
                    self._last_total_tokens = 0
                    self.successful_requests += 1
                # =============================================
                
                return content

            except HTTPError as e:
                status = e.response.status_code

                # 401: token expired → refresh and retry
                if status == 401 and attempt < max_attempts:
                    print("[AIFChatLLM] 401 Unauthorized. Refreshing token...")
                    self.token = get_access_token()
                    continue

                # 429: rate limit → backoff
                if status == 429 and attempt < max_attempts:
                    retry_after = e.response.headers.get("Retry-After")
                    try:
                        wait = float(retry_after)
                    except (TypeError, ValueError):
                        wait = base_backoff * attempt
                    print(
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

        raise last_err if last_err else RuntimeError("Unknown error calling AIF GPT-5 endpoint")

    def get_usage_metrics(self) -> Dict[str, int]:
        """
        Helper method to get current usage statistics.
        """
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
        """
        Estimate token count for messages. CrewAI may call this.
        """
        total = 0
        for msg in messages:
            total += self.get_num_tokens(msg.get("content", ""))
        return total

    def reset_usage_metrics(self):
        """
        Reset usage counters (useful between different crew runs).
        """
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
        """
        Property that CrewAI can check for usage from the last call.
        """
        return self._last_call_usage

    def _get_usage_from_last_call(self) -> Dict[str, int]:
        """
        Alternative method for CrewAI to get usage data.
        """
        return self._last_call_usage
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        Explicit method for telemetry systems to get cumulative usage.
        """
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


# --------------------------------------------------
# 4) Shared factory for all agents
# --------------------------------------------------

def get_shared_llm() -> AIFChatLLM:
    llm = AIFChatLLM()
    print("Using AIFChatLLM with endpoint:", llm.model_url)
    return llm