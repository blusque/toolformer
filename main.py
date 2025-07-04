# SPDX-License-Identifier: Apache-2.0
from argparse import Namespace
from email import message
import os

from openai import OpenAI
from toolformer_prompt import toolformer_answer_prompt, toolformer_generation_prompt, toolformer_error_prompt, toolformer_thinking_prompt, toolformer_thinking_completion_prompt, toolformer_thinking_error_prompt

from tools import Calendar, Calculator

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

def invoke_tool_calls(content: str) -> tuple[str, bool]:
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
        return content[:tool_call_start] + f"{result}", False
    elif content.endswith("<eos>"):
        # If the content ends with <eos>, it means the answer is finished.
        return content[:-5], True
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

def main(args: Namespace):
    # Pop arguments not used by LLM
    max_tokens = args.max_tokens
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k

    inputs = [
        "I planned a trip to Paris on Christmas day this year, can you tell me how many days until then?",
        "Can you tell me the current date?",
        "Can you solve 2*8+3?",
        "Can you calculate 1+1?",
    ]

    client = setup_openai_client(api_key=dashscope_api_key)

    # for input_text in inputs:
    #     tool_called = False
    #     messages = [
    #         {"role": "system", "content": toolformer_thinking_prompt},
    #         {"role": "user", "content": input_text},
    #     ]
    
    #     reasoning_content: str = ""
    #     answer_content: str = ""
    #     is_answering = False

    #     while True:
    #         stream = client.chat.completions.create(
    #             model="qwen-plus",
    #             messages=messages,
    #             max_tokens=max_tokens,
    #             temperature=temperature,
    #             top_p=top_p,
    #             extra_body={
    #                 "top_k": top_k,
    #                 "enable_thinking": True
    #             },
    #             stream=True
    #         )
    #         for chunk in stream:
    #             # 处理流式输出
    #             if not chunk.choices:
    #                 print("\nUsage:")
    #                 print(chunk.usage)
    #                 continue

    #             delta = chunk.choices[0].delta

    #             # 只收集思考内容
    #             if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
    #                 if not is_answering:
    #                     reasoning_content += delta.reasoning_content
    #                     print(reasoning_content, end="", flush=True)
    #                 # # 处理思考内容
    #                 try:
    #                     reasoning_content, tool_called = invoke_tool_calls(reasoning_content)
    #                     prompt = fit_prompt(toolformer_thinking_completion_prompt, input=reasoning_content)
    #                 except ValueError as e:
    #                     # print(f"Error processing tool calls: {e}")
    #                     prompt = fit_prompt(toolformer_thinking_error_prompt, error=str(e), previous_answer=reasoning_content)
    #                 if tool_called:
    #                     # 重新开始流式输出
    #                     messages.append({"role": "assistant", "content": reasoning_content})
    #                     messages.append(prompt)
    #                     stream.close()
    #                     break

    #             # 收到content，开始进行回复
    #             elif hasattr(delta, "content") and delta.content:
    #                 if not is_answering:
    #                     print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
    #                     is_answering = True
    #                 print(delta.content, end="", flush=True)
    #                 answer_content += delta.content
    #         messages.append({"role": "assistant", "content": answer_content})
    #         print('\n')

    for input_text in inputs:
        finished = False
        output = ''
        previous_output = None
        tolerance = 3
        same_count = 0
        messages=[
            {"role": "system", "content": toolformer_answer_prompt},
            {"role": "user", "content": input_text},
        ]
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
            print(response.choices[0])
            exit(0)
            print(f"Question: {input_text}")
            print(f"previous output: {output}")
            print(f"Output: {response.choices[0].message.content}")
            # print(messages)
            # Replace tool calls in the response
            if response.choices[0].message.content:
                try:
                    output, finished = replace_tool_calls(response.choices[0].message.content)
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Add engine args
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    main(args)