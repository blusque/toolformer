# SPDX-License-Identifier: Apache-2.0
from argparse import Namespace
from email import message
import os
from typing import Never

from openai import OpenAI
from .toolformer_prompt import toolformer_answer_prompt, toolformer_generation_prompt, toolformer_error_prompt, toolformer_summary_prompt

from .tools import Calendar, Calculator

tools_map = {
    "Calendar": Calendar,
    "Calculator": Calculator
}

dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "sk-ba7b3e19b37d49ca9bea6ec40bb8077a")

def fit_prompt(prompt: str, **kwargs) -> dict:
    """
    Fit the prompt with the input.
    """
    content = prompt.format(**kwargs)
    return {"role": "user", "content": content}

def get_tool_call_args(args: str) -> list:
    """
    Extract the arguments from the tool call string.
    """
    # In case there are strings with commas, we first get the strings
    # and then split other args by commas.
    return [arg.strip() for arg in args.split(",")] if args else []

def invoke_tool(content: str, used_tools: set) -> tuple[str, bool]:
    """
    Replace tool calls in the message with a specific format.
    """
    # This function is a placeholder for actual tool call replacement logic.
    # In a real implementation, you would parse the message and replace tool calls accordingly.
    tool_call_start = content.find("[")
    tool_call_end = content.find("->")
    if tool_call_start != -1 and tool_call_end != -1:
        tool_call = content[tool_call_start + 1:tool_call_end].split("(")
        tool_call_name = tool_call[0].strip()
        tool_call_args = get_tool_call_args(tool_call[1].rstrip(')') if len(tool_call[1]) > 1 else "")
        print(f"Tool call detected: {tool_call_name} with args {tool_call_args}")
        try:
            result = tools_map[tool_call_name](*tool_call_args)
        except Exception as e:
            raise ValueError(f"Error calling tool {tool_call_name} with args {tool_call_args}: {e}")
        used_tools.add(tool_call_name)
        return content[:tool_call_start] + f"{result}", False
    elif content.index("<eos>") != -1:
        return content[:content.index("<eos>")], True
    else:
        print("No tool call detected.")
        return content, False

def setup_openai_client(api_key: str) -> OpenAI:
    """
    Set up the OpenAI client with the provided API key.
    """
    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key
    )
    return client

def chat_completion(client: OpenAI, messages: list, max_tokens: int, temperature: float, top_p: float, top_k: float) -> tuple[str, list] | Never:
    """
    Send a chat completion request to the OpenAI API.
    """
    finished = False
    output = ''
    previous_output = None
    tolerance = 3
    same_count = 0
    used_tools = set()
    while not finished:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body={
                "top_k": top_k,
                "stop": ["]"],
            }
        )
        # print(messages)
        # Replace tool calls in the response
        if response.choices[0].message.content:
            try:
                output, finished = invoke_tool(response.choices[0].message.content)
            except ValueError as e:
                print(f"Error processing tool calls: {e}")
                messages.append({"role": "assistant", "content": output})
                messages.append(fit_prompt(toolformer_error_prompt, error=str(e), previous_answer=output))
                continue
        print(f"Processed Output: {output}")
        if previous_output is not None and previous_output == output:
            print("No change in output, stopping. count:", same_count, "tolerance:", tolerance)
            same_count += 1
            if same_count >= tolerance:
                print("Reached tolerance limit, stopping.")
                finished = True
        else:
            same_count = 0
        previous_output = output
        if finished:
            print("Finished processing the input.")
        else:
            print("Continuing to process the input with tool calls.")
        messages.append({"role": "assistant", "content": output})
        messages.append(fit_prompt(toolformer_generation_prompt, input=output))
        # print(f"Updated messages: {messages}")
        print("-" * 40)

    # Send the final response
    messages.append(fit_prompt(toolformer_summary_prompt))
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    if response.choices[0].message.content is not None:
        return response.choices[0].message.content, list(used_tools)  # Return the final summarized response
    else:
        raise ValueError("No content in the final response from the model.")


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     # Add engine args
#     parser.add_argument("--max-tokens", type=int, default=256)
#     parser.add_argument("--temperature", type=float, default=0.7)
#     parser.add_argument("--top-p", type=float, default=1.0)
#     parser.add_argument("--top-k", type=int, default=50)
#     args = parser.parse_args()