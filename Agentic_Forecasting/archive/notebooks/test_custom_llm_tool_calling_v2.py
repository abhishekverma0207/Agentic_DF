# Databricks notebook source
# MAGIC %md
# MAGIC # Test Custom LLM with Tool Calling - V2
# MAGIC
# MAGIC This notebook tests different approaches to make tool calling work with our GPT-5 AIF endpoint.
# MAGIC
# MAGIC **Key Finding from V1:** CrewAI doesn't pass `tools`/`available_functions` to custom `BaseLLM.call()`.
# MAGIC It relies on LiteLLM's built-in function calling OR text-based ReAct parsing.
# MAGIC
# MAGIC **This notebook tests:**
# MAGIC 1. Using LiteLLM's custom provider mechanism
# MAGIC 2. Using `function_calling_llm` parameter
# MAGIC 3. Forcing tool execution in our custom LLM by intercepting the ReAct output

# COMMAND ----------

# MAGIC %pip install crewai crewai-tools requests pydantic litellm --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Setup AIF Environment Variables

# COMMAND ----------

client_secret = dbutils.secrets.get(scope="scope-databrickskv", key="svc-b-da-d-931272-ina-aadprincipal")

os.environ["AIF_TOKEN_URL"] = "https://login.microsoftonline.com/f66fae02-5d36-495b-bfe0-78a6ff9f8e6e/oauth2/v2.0/token"
os.environ["AIF_SCOPE"] = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"
os.environ["AIF_MODEL_URL"] = "https://bnlwe-ai03-q-931039-apim-01.azure-api.net/openai5/az_openai_gpt-5_chat"
os.environ["AIF_CLIENT_ID"] = "c6d73674-8d75-47c0-9f78-927c687ce20a"
os.environ["AIF_CLIENT_SECRET"] = client_secret
os.environ["AIF_SUBSCRIPTION_KEY"] = "6e4956c3b8854fb191a43ea4b7a0c063"

print("Environment variables set!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Helper Functions for AIF API

# COMMAND ----------

import requests
import json
import time
from typing import Any, Dict, List, Optional

_cached_token = None
_token_expiry = 0

def get_aif_token() -> str:
    """Get AIF access token with caching."""
    global _cached_token, _token_expiry

    if _cached_token and time.time() < _token_expiry - 60:
        return _cached_token

    response = requests.post(
        os.environ["AIF_TOKEN_URL"],
        data={"grant_type": "client_credentials", "scope": os.environ["AIF_SCOPE"]},
        auth=(os.environ["AIF_CLIENT_ID"], os.environ["AIF_CLIENT_SECRET"]),
    )
    response.raise_for_status()
    data = response.json()
    _cached_token = data["access_token"]
    _token_expiry = time.time() + data.get("expires_in", 3600)
    return _cached_token


def call_aif_api(messages: List[Dict], tools: Optional[List[Dict]] = None, max_tokens: int = 4096) -> Dict:
    """Call the AIF GPT-5 API directly."""
    token = get_aif_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Ocp-Apim-Subscription-Key": os.environ["AIF_SUBSCRIPTION_KEY"],
        "Content-Type": "application/json",
    }

    payload = {"messages": messages, "max_tokens": max_tokens}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    print(f"[AIF API] Calling with {len(messages)} messages, {len(tools) if tools else 0} tools")

    response = requests.post(os.environ["AIF_MODEL_URL"], headers=headers, json=payload)

    if response.status_code >= 400:
        print(f"[AIF API] Error {response.status_code}: {response.text}")
    response.raise_for_status()

    return response.json()

print("Helper functions defined!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Test Direct API Call with Tools

# COMMAND ----------

print("=" * 60)
print("TEST: Direct API call WITH tools parameter")
print("=" * 60)

test_tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code and return output",
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

test_messages = [
    {"role": "system", "content": "You are a helpful assistant. Use the execute_code tool to run Python code."},
    {"role": "user", "content": "Run this Python code: print('Hello from API test!')"}
]

result = call_aif_api(test_messages, tools=test_tools)

msg = result["choices"][0]["message"]
print(f"\nContent: {msg.get('content', '(none)')}")
print(f"Tool calls: {msg.get('tool_calls', '(none)')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Test Direct API WITHOUT tools (ReAct style)

# COMMAND ----------

print("=" * 60)
print("TEST: Direct API call WITHOUT tools (expecting ReAct format)")
print("=" * 60)

react_messages = [
    {"role": "system", "content": """You are a helpful assistant with access to a python_code_executor tool.

When you need to execute Python code, use this EXACT format:
Thought: [your reasoning]
Action: python_code_executor
Action Input: {"code": "[your python code here]"}

After the tool executes, you will receive an Observation with the result.
When you have your final answer, respond with:
Thought: [your reasoning]
Final Answer: [your response]

IMPORTANT: You must use the tool to execute code - do not make up results."""},
    {"role": "user", "content": "Run this Python code and tell me the result: print('Hello from ReAct test!')"}
]

result = call_aif_api(react_messages, tools=None)

msg = result["choices"][0]["message"]
content = msg.get('content', '')
print(f"\nContent:\n{content}")
print(f"\nTool calls: {msg.get('tool_calls', '(none)')}")

# Check if it's using ReAct format
if "Action:" in content and "Action Input:" in content:
    print("\n✅ LLM is outputting ReAct format!")
else:
    print("\n❌ LLM is NOT using ReAct format")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Custom LLM that Handles Tool Execution Internally
# MAGIC
# MAGIC Since CrewAI doesn't pass tools to BaseLLM, we need to:
# MAGIC 1. Pass tools to the API ourselves
# MAGIC 2. Execute tool calls ourselves
# MAGIC 3. Loop until we get a final text response

# COMMAND ----------

from crewai import BaseLLM
from typing import Union
import re
import textwrap
import io
import contextlib

class AIFChatLLMWithTools(BaseLLM):
    """
    Custom LLM that handles tool execution internally.

    Since CrewAI doesn't pass tools/available_functions to BaseLLM.call(),
    we register tools directly and handle function calling ourselves.
    """

    def __init__(self, model: str = "gpt-5-aif", temperature: float = 0.2, max_tokens: int = 4096):
        super().__init__(model=model, temperature=temperature)
        self.max_tokens = max_tokens
        self._registered_tools: Dict[str, Any] = {}
        self._tool_schemas: List[Dict] = []

        # Usage tracking
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def register_tool(self, name: str, description: str, func: callable, parameters: Dict):
        """Register a tool that this LLM can call."""
        self._registered_tools[name] = func
        self._tool_schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        })
        print(f"[LLM] Registered tool: {name}")

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Execute LLM call with internal tool handling."""

        # Normalize messages
        if isinstance(messages, str):
            msg_list = [{"role": "user", "content": messages}]
        else:
            msg_list = list(messages)

        # Use our registered tools since CrewAI doesn't pass them
        tools_to_use = self._tool_schemas if self._registered_tools else None
        funcs_to_use = self._registered_tools if self._registered_tools else None

        print(f"\n[LLM] call() - messages: {len(msg_list)}, registered_tools: {len(self._registered_tools)}")

        max_iterations = 10
        for iteration in range(max_iterations):
            print(f"[LLM] Iteration {iteration + 1}")

            # Call API
            result = call_aif_api(msg_list, tools=tools_to_use, max_tokens=self.max_tokens)

            # Update usage
            usage = result.get("usage", {})
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)

            # Extract response
            response_message = result["choices"][0]["message"]
            content = response_message.get("content", "") or ""
            tool_calls = response_message.get("tool_calls", [])

            print(f"[LLM] Content: {len(content)} chars, Tool calls: {len(tool_calls)}")

            # Handle tool calls
            if tool_calls and funcs_to_use:
                print(f"[LLM] Processing {len(tool_calls)} tool call(s)...")
                msg_list.append(response_message)

                for tool_call in tool_calls:
                    func_name = tool_call["function"]["name"]
                    func_args_str = tool_call["function"]["arguments"]
                    tool_call_id = tool_call["id"]

                    print(f"[LLM] Executing: {func_name}")

                    try:
                        func_args = json.loads(func_args_str)
                    except:
                        func_args = {}

                    if func_name in funcs_to_use:
                        try:
                            result_str = funcs_to_use[func_name](**func_args)
                            print(f"[LLM] Tool result: {len(str(result_str))} chars")
                        except Exception as e:
                            result_str = f"Error: {str(e)}"
                    else:
                        result_str = f"Unknown tool: {func_name}"

                    msg_list.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": func_name,
                        "content": str(result_str),
                    })

                continue  # Get next response

            # No tool calls - return content
            return content

        return "Max iterations reached without final response"

    def supports_function_calling(self) -> bool:
        return True


def execute_python_code(code: str) -> str:
    """Execute Python code and return output."""
    print(f"\n{'='*50}")
    print("[TOOL] python_code_executor CALLED!")
    print(f"[TOOL] Code:\n{code[:300]}...")
    print(f"{'='*50}\n")

    code_to_exec = textwrap.dedent(code).strip()
    _globals = {}

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(code_to_exec, _globals, _globals)

        stdout_val = stdout_buf.getvalue().strip()
        stderr_val = stderr_buf.getvalue().strip()

        sections = []
        if stdout_val:
            sections.append(f"[STDOUT]\n{stdout_val}")
        if stderr_val:
            sections.append(f"[STDERR]\n{stderr_val}")
        if "result" in _globals:
            sections.append(f"[RESULT]\n{repr(_globals['result'])[:1000]}")
        if not sections:
            sections.append("[OK] Code executed successfully (no output)")

        return "\n\n".join(sections)
    except Exception as e:
        import traceback
        return f"[ERROR]\n{traceback.format_exc()}"


print("AIFChatLLMWithTools defined!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Test Custom LLM with Registered Tools (Direct)

# COMMAND ----------

print("=" * 60)
print("TEST: Custom LLM with registered tools (direct call)")
print("=" * 60)

llm = AIFChatLLMWithTools()

# Register the code execution tool
llm.register_tool(
    name="python_code_executor",
    description="Execute Python code and return the output. Use this for any Python code execution.",
    func=execute_python_code,
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python code to execute"}
        },
        "required": ["code"]
    }
)

# Test direct call
response = llm.call("Please run this Python code and tell me the result: print('Hello from direct test!')")

print(f"\n{'='*60}")
print(f"Response:\n{response}")
print(f"{'='*60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Test with CrewAI Agent

# COMMAND ----------

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

# We still need to define a CrewAI tool (even though our LLM handles it)
# This is because CrewAI injects tool descriptions into the prompt

class CodeToolSchema(BaseModel):
    code: str = Field(..., description="Python code to execute")

class CodeTool(BaseTool):
    name: str = "python_code_executor"
    description: str = "Execute Python code and return the output. Use this to run any Python code."
    args_schema: Type[CodeToolSchema] = CodeToolSchema

    def _run(self, code: str) -> str:
        # This should be called by CrewAI's tool execution
        # But since CrewAI doesn't properly call it for custom LLMs,
        # our AIFChatLLMWithTools handles it internally
        return execute_python_code(code)


# Create our custom LLM with registered tool
llm = AIFChatLLMWithTools()
llm.register_tool(
    name="python_code_executor",
    description="Execute Python code and return the output",
    func=execute_python_code,
    parameters={
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python code to execute"}},
        "required": ["code"]
    }
)

# Create agent
agent = Agent(
    name="data_reader",
    role="Data Reader",
    goal="Read CSV files and report their contents using Python code",
    backstory="You are a data reader. Use the python_code_executor tool to run code.",
    tools=[CodeTool()],  # CrewAI needs this for prompt injection
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# Test file path
DATA_PATH = "/Volumes/pds_feu_931272_dev/data_science_team/us_run/HOMECARE/sourcedata/Home_Care_US.csv"

task = Task(
    description=f"""
    Use Python code to:
    1. Check if this file exists: {DATA_PATH}
    2. If it exists, read it with pandas and print the shape and first 5 rows

    You MUST use the python_code_executor tool to run the code.
    """,
    expected_output="A report with file existence, shape, and first 5 rows if file exists",
    agent=agent,
)

crew = Crew(agents=[agent], tasks=[task], verbose=True)

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
# MAGIC **Approach:** Since CrewAI doesn't pass `tools`/`available_functions` to `BaseLLM.call()`,
# MAGIC we created `AIFChatLLMWithTools` that:
# MAGIC 1. Registers tools directly with the LLM instance
# MAGIC 2. Passes tools to the GPT-5 API ourselves
# MAGIC 3. Handles tool_calls internally and executes them
# MAGIC 4. Loops until we get a final text response
# MAGIC
# MAGIC This bypasses CrewAI's broken tool execution for custom LLMs.
