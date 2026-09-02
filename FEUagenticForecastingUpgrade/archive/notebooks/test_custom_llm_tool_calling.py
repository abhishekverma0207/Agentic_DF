# Databricks notebook source
# MAGIC %md
# MAGIC # Test Custom LLM with Tool Calling
# MAGIC
# MAGIC This notebook tests our custom AIFChatLLM implementation with CrewAI to verify:
# MAGIC 1. The LLM can make API calls successfully
# MAGIC 2. Tool calling works (the agent actually executes Python code)
# MAGIC 3. The agent can read files from Unity Catalog Volumes

# COMMAND ----------

# MAGIC %pip install crewai==0.108.0 crewai-tools requests pydantic --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import logging

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Setup AIF Environment Variables

# COMMAND ----------

# Get client secret from Databricks secrets
client_secret = dbutils.secrets.get(scope="scope-databrickskv", key="svc-b-da-d-931272-ina-aadprincipal")

# Set environment variables for AIF
os.environ["AIF_TOKEN_URL"] = "https://login.microsoftonline.com/f66fae02-5d36-495b-bfe0-78a6ff9f8e6e/oauth2/v2.0/token"
os.environ["AIF_SCOPE"] = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"
os.environ["AIF_MODEL_URL"] = "https://bnlwe-ai03-q-931039-apim-01.azure-api.net/openai5/az_openai_gpt-5_chat"
os.environ["AIF_CLIENT_ID"] = "c6d73674-8d75-47c0-9f78-927c687ce20a"
os.environ["AIF_CLIENT_SECRET"] = client_secret
os.environ["AIF_SUBSCRIPTION_KEY"] = "6e4956c3b8854fb191a43ea4b7a0c063"

print("Environment variables set!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Define the Custom LLM (AIFChatLLM)

# COMMAND ----------

import time
import json
import requests
from requests.exceptions import HTTPError
from typing import Any, Dict, List, Optional, Union

from crewai import BaseLLM

def _get_aif_access_token() -> str:
    """Obtain an AAD bearer token for the AIF scope."""
    url = os.getenv("AIF_TOKEN_URL")
    client_id = os.getenv("AIF_CLIENT_ID")
    client_secret = os.getenv("AIF_CLIENT_SECRET")
    scope = os.getenv("AIF_SCOPE")

    response = requests.post(
        url,
        data={"grant_type": "client_credentials", "scope": scope},
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
) -> Dict[str, Any]:
    """Call the AIF GPT-5 chat endpoint."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {"messages": messages, "max_tokens": max_tokens}

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        print(f"[API] Passing {len(tools)} tools to API")

    print(f"[API] Calling: {url}")
    print(f"[API] Messages: {len(messages)}, Tools: {len(tools) if tools else 0}")

    response = requests.post(url, headers=headers, json=payload)
    print(f"[API] Response status: {response.status_code}")

    if response.status_code >= 400:
        print(f"[API] Error: {response.text}")
    response.raise_for_status()

    result = response.json()
    print(f"[API] Response keys: {result.keys()}")
    return result


class AIFChatLLM(BaseLLM):
    """Custom LLM for GPT-5 via AIF with tool calling support."""

    def __init__(self, model: str = "az_openai_gpt-5_chat", temperature: float = 0.2, max_tokens: int = 50000):
        super().__init__(model=model, temperature=temperature)
        self.max_tokens = max_tokens
        self.model_url = os.getenv("AIF_MODEL_URL")
        self.subscription_key = os.getenv("AIF_SUBSCRIPTION_KEY")
        self.token = None

        # Usage tracking
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.successful_requests = 0
        self._last_call_usage = {}

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Main entry point for LLM calls."""

        # Normalize messages
        if isinstance(messages, str):
            msg_list = [{"role": "user", "content": messages}]
        else:
            msg_list = list(messages)

        # Get token if needed
        if self.token is None:
            print("[LLM] Fetching access token...")
            self.token = _get_aif_access_token()
            print("[LLM] Token obtained!")

        # Log what we received
        print(f"[LLM] call() invoked:")
        print(f"[LLM]   - tools: {len(tools) if tools else 0}")
        print(f"[LLM]   - available_functions: {list(available_functions.keys()) if available_functions else None}")
        print(f"[LLM]   - messages: {len(msg_list)}")

        max_tool_iterations = 10
        for iteration in range(max_tool_iterations):
            print(f"\n[LLM] === Iteration {iteration + 1} ===")

            # Call API - pass tools if we have available_functions to execute them
            resp_json = _run_aif_model(
                self.model_url,
                self.token,
                self.subscription_key,
                msg_list,
                max_tokens=self.max_tokens,
                tools=tools if available_functions else None,  # Only pass tools if we can execute them
            )

            # Extract response
            response_message = resp_json["choices"][0]["message"]
            content = response_message.get("content", "") or ""
            tool_calls = response_message.get("tool_calls", [])

            print(f"[LLM] Content length: {len(content)}")
            print(f"[LLM] Content preview: {content[:200] if content else '(empty)'}...")
            print(f"[LLM] Tool calls: {len(tool_calls) if tool_calls else 0}")

            # Handle tool calls
            if tool_calls and available_functions:
                print(f"[LLM] Processing {len(tool_calls)} tool call(s)...")
                msg_list.append(response_message)

                for tool_call in tool_calls:
                    func_name = tool_call["function"]["name"]
                    func_args_str = tool_call["function"]["arguments"]
                    tool_call_id = tool_call["id"]

                    print(f"[LLM] Executing tool: {func_name}")
                    print(f"[LLM] Arguments: {func_args_str[:200]}...")

                    try:
                        func_args = json.loads(func_args_str)
                    except json.JSONDecodeError:
                        func_args = {}

                    if func_name in available_functions:
                        try:
                            func = available_functions[func_name]
                            result = func(**func_args)
                            print(f"[LLM] Tool result length: {len(str(result))}")
                            print(f"[LLM] Tool result preview: {str(result)[:300]}...")
                        except Exception as e:
                            result = f"Error: {str(e)}"
                            print(f"[LLM] Tool error: {result}")
                    else:
                        result = f"Function {func_name} not found"
                        print(f"[LLM] {result}")

                    msg_list.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": func_name,
                        "content": str(result),
                    })

                # Continue loop to get final response
                continue

            # No tool calls - return content
            self._update_usage(resp_json.get("usage", {}))
            return content

        raise RuntimeError("Max tool iterations reached")

    def _update_usage(self, usage: Dict[str, int]) -> None:
        if usage:
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
            self.successful_requests += 1
            self._last_call_usage = usage

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 50000

print("AIFChatLLM class defined!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Test Basic LLM Call (No Tools)

# COMMAND ----------

print("=" * 60)
print("TEST 1: Basic LLM call without tools")
print("=" * 60)

llm = AIFChatLLM()

response = llm.call("What is 2 + 2? Reply with just the number.")
print(f"\nFinal Response: {response}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Test LLM with Tool Calling (Direct)

# COMMAND ----------

print("=" * 60)
print("TEST 2: LLM call WITH tools and available_functions")
print("=" * 60)

# Define a simple tool
test_tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    }
]

# Define the function that executes the code
def execute_code(code: str) -> str:
    import io
    import sys
    print(f"\n>>> EXECUTING CODE <<<\n{code}\n>>> END CODE <<<\n")
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    try:
        exec(code, {"__builtins__": __builtins__})
        output = mystdout.getvalue()
    except Exception as e:
        output = f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout
    return output if output else "Code executed successfully (no output)"

available_functions = {"execute_code": execute_code}

# Test message
test_messages = [
    {"role": "system", "content": "You are a helpful assistant. When asked to run Python code, use the execute_code tool."},
    {"role": "user", "content": "Please run this Python code: print('Hello from tool test!')"}
]

response = llm.call(
    messages=test_messages,
    tools=test_tools,
    available_functions=available_functions
)

print(f"\n{'=' * 60}")
print(f"Final Response: {response}")
print(f"{'=' * 60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Test with CrewAI Agent (The Real Test)

# COMMAND ----------

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import textwrap
import io
import contextlib

# Define a simple code execution tool
class CodeExecutionToolSchema(BaseModel):
    code: str = Field(..., description="Python code to execute")

class SimpleCodeTool(BaseTool):
    name: str = "python_code_executor"
    description: str = "Executes Python code and returns the output. Use this to run any Python code."
    args_schema: Type[CodeExecutionToolSchema] = CodeExecutionToolSchema
    _globals: dict = {}

    def _run(self, code: str) -> str:
        print(f"\n{'='*60}")
        print(f"[TOOL] python_code_executor CALLED!")
        print(f"[TOOL] Code:\n{code[:500]}...")
        print(f"{'='*60}\n")

        code_to_exec = textwrap.dedent(code).strip()
        if not self._globals:
            self._globals = {}

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(code_to_exec, self._globals, self._globals)

            stdout_val = stdout_buf.getvalue().strip()
            stderr_val = stderr_buf.getvalue().strip()

            sections = []
            if stdout_val:
                sections.append(f"[STDOUT]\n{stdout_val}")
            if stderr_val:
                sections.append(f"[STDERR]\n{stderr_val}")
            if "result" in self._globals:
                sections.append(f"[RESULT]\n{repr(self._globals['result'])}")
            if not sections:
                sections.append("[OK] Code executed successfully.")

            output = "\n\n".join(sections)
            print(f"[TOOL] Output preview: {output[:300]}...")
            return output
        except Exception as e:
            import traceback
            return f"[ERROR]\n{traceback.format_exc()}"

print("SimpleCodeTool defined!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test: Simple Agent that reads a CSV file

# COMMAND ----------

# Create the LLM
llm = AIFChatLLM()

# Create the tool
code_tool = SimpleCodeTool()

# Create a simple agent
agent = Agent(
    name="data_reader",
    role="Data Reader",
    goal="Read CSV files and report their contents",
    backstory=(
        "You are a data reader agent. Your job is to use the python_code_executor tool "
        "to read CSV files and report what you find. You MUST use the tool to execute code - "
        "do not make assumptions about file contents."
    ),
    tools=[code_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# Create a task - using a file path you know exists
DATA_PATH = "/Volumes/pds_feu_931272_dev/data_science_team/us_run/HOMECARE/sourcedata/Home_Care_US.csv"

task = Task(
    description=f"""
    Use the python_code_executor tool to:
    1. Check if this file exists: {DATA_PATH}
    2. If it exists, read it with pandas and print the shape and first 5 rows
    3. Report what you found

    You MUST use the python_code_executor tool to run Python code. Do not guess or assume.
    """,
    expected_output="A report of the file contents including shape and first 5 rows",
    agent=agent,
)

# Create and run the crew
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
)

print("Starting crew...")
result = crew.kickoff()
print(f"\n{'='*60}")
print("CREW RESULT:")
print(f"{'='*60}")
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook tests:
# MAGIC 1. **Basic LLM call** - Does the API work at all?
# MAGIC 2. **Direct tool calling** - Does our `call()` method handle tools correctly when we pass `available_functions`?
# MAGIC 3. **CrewAI integration** - Does CrewAI pass tools and available_functions to our LLM, and does the agent actually use the tool?
# MAGIC
# MAGIC If Test 2 works but Test 3 doesn't, the issue is with how CrewAI calls our LLM.
# MAGIC If Test 2 doesn't work, the issue is with our tool calling implementation.
