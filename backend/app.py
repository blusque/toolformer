from email import message
import re
import token
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import sys
import ollama


app = FastAPI(title="Toolformer Chat API", version="1.0.0")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    message: str
    tools_used: Optional[List[str]] = []

# client = None  # Placeholder for the LLM, replace with actual model loading if needed
template = None  # Placeholder for the template, replace with actual template loading if needed

def initialize_client():
    """
    Initialize the LLM if needed.
    This function can be used to load the model and tokenizer.
    """
    global template

    # success = os.system("vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 8000 --device 0")
    # if success != 0:
    #     print("Failed to start vLLM server. Please ensure vLLM is installed and configured correctly.")
    #     raise RuntimeError("Failed to start vLLM server")

    # if client is None:
    #     client = InferenceClient(
    #         provider="nscale",
    #         api_key=os.getenv("HF_TOKEN"),  # Replace with your Hugging Face API key
    #     )
    if template is None:
        template = "You are a helpful assistant. Respond to the user's message using the tools provided. If you use a tool, mention it in square brackets like [tool_name(param1, param2, ...)]. User message: {content}"

# Mock toolformer function - replace with actual implementation
def toolformer_chat(messages: List[dict], max_input_tokens: int = 2048, max_tokens: int = 150, temperature: float = 0.7) -> dict:
    """
    Replace this with your actual Toolformer implementation
    """
    global template
    # global client, template

    if template is None:
        try:
            initialize_client()
        except RuntimeError as e:
            print(f"Error initializing client: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    messages[-1]["content"] = template.format(content=messages[-1]["content"])
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
    # Ensure the input tokens do not exceed the limit
    token_count = sum(len(msg["content"].split()) for msg in messages)
    if token_count > max_input_tokens:
        summerize_template = "You are a helpful assistant, and you are good at summarizing the conversation and retrieving the import information. The conversation will be given in the format of 'role: content', and there are three roles so far: system, user and assistant. You should summarize the conversation within {num_words} words.Summarize the conversation so far: {content}"
        conversation = ""
        conversation_template = "{role}: {content}\n"
        for msg in messages:
            for key, value in msg.items():
                if isinstance(value, str):
                    value = value.strip()
                conversation += conversation_template.format(role=key, content=value)
        summerize_message = summerize_template.format(num_words=max_input_tokens // 2, content=conversation)
        # response = client.chat.completions.create(
        #     model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        #     messages=[{"role": "user", "content": summerize_message}],
        #     max_tokens=max_input_tokens,
        #     temperature=temperature,
        #     stop=["\n"]
        # )
        response: ollama.ChatResponse = ollama.chat(
            model="qwen2.5:1.5b",
            messages=[{"role": "user", "content": summerize_message}]
        )
        summary = response.message.content.strip()
        messages = [{"role": "user", "content": summary}]

    # response from the LLM
    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=messages
    )

    last_message = response.message.content.strip()
    messages.append({
        "role": "assistant",
        "content": last_message
    })
    
    # parser for tools used
    if re.search(r'\[\w+\(.*?\)\]', last_message):
        tools_used = re.findall(r'\[(\w+)\(.*?\)\]', last_message)
        tools_used = [tool.split("(")[0].split("[")[1] for tool in tools_used]
    else:
        tools_used = []

    return {
        "response": f"Toolformer response to: {last_message}",
        "tools_used": tools_used
    }

# Mount static files
app.mount("/static", StaticFiles(directory="../template"), name="static")

@app.get("/")
async def serve_home():
    return FileResponse("../template/index.html")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        print(f"Received messages: {messages_dict}")
        
        result = toolformer_chat(
            messages=messages_dict,
            max_tokens=request.max_tokens, # type: ignore
            temperature=request.temperature # type: ignore
        )
        
        return ChatResponse(
            message=result["response"],
            tools_used=result.get("tools_used", [])
        )
    
    except Exception as e:
        print(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2025)