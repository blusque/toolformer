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
from openai import OpenAI
from fastapi.responses import StreamingResponse
import json
from backend.toolformer import chat_completion
from toolformer_prompt import toolformer_answer_prompt, toolformer_generation_prompt, toolformer_error_prompt, toolformer_summary_prompt


app = FastAPI(title="Toolformer Chat API", version="1.0.0")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50

class ChatResponse(BaseModel):
    message: str
    tools_used: Optional[List[str]] = []

# Mount static files
app.mount("/static", StaticFiles(directory="../template"), name="static")

@app.get("/")
async def serve_home():
    return FileResponse("../template/index.html")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        messages = [{"role": "system", "content": toolformer_answer_prompt}]
        messages.extend([{"role": msg.role, "content": msg.content} for msg in request.messages])

        result, used_tools = chat_completion(
            messages=messages,
            max_tokens=request.max_tokens, # type: ignore
            temperature=request.temperature, # type: ignore
            top_p=request.top_p, # type: ignore
            top_k=request.top_k # type: ignore
        )
        
        return ChatResponse(
            message=result,
            tools_used=used_tools
        )
    
    except Exception as e:
        print(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# @app.post("/chat/stream")
# async def chat_stream(request: ChatRequest):
#     try:
#         messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
#         def generate_response():
#             global client, template
            
#             if client is None or template is None:
#                 try:
#                     initialize_client()
#                 except RuntimeError as e:
#                     yield f"data: {json.dumps({'error': str(e)})}\n\n"
#                     return
            
#             messages_dict[-1]["content"] = template.format(content=messages_dict[-1]["content"])
#             messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages_dict]
            
#             try:
#                 response = client.chat.completions.create(
#                     model="dmayhem93/toolformer_v0_epoch2",
#                     messages=messages,
#                     temperature=request.temperature,
#                     stop=["\n"],
#                     stream=True
#                 )
                
#                 full_content = ""
#                 for chunk in response:
#                     if chunk.choices[0].delta.content:
#                         content = chunk.choices[0].delta.content
#                         full_content += content
#                         yield f"data: {json.dumps({'content': content, 'type': 'chunk'})}\n\n"
                
#                 # Extract tools used from full content
#                 tools_used = []
#                 if re.search(r'\[\w+\(.*?\)\]', full_content):
#                     tools_used = re.findall(r'\[(\w+)\(.*?\)\]', full_content)
#                     tools_used = [tool.split("(")[0] for tool in tools_used]
                
#                 yield f"data: {json.dumps({'type': 'done', 'tools_used': tools_used})}\n\n"
                
#             except Exception as e:
#                 yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
#         return StreamingResponse(
#             generate_response(),
#             media_type="text/plain",
#             headers={
#                 "Cache-Control": "no-cache",
#                 "Connection": "keep-alive",
#                 "Content-Type": "text/event-stream"
#             }
#         )
        
#     except Exception as e:
#         print(f"Error during streaming chat: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2025)