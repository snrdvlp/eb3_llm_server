from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import httpx

app = FastAPI()
VLLM_SERVER = "http://localhost:8000/v1/chat/completions"

class LLMRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    max_new_tokens: Optional[int] = None

@app.post("/chat")
async def chat(req: LLMRequest):
    # Qwen expects message role format
    messages = [
        {"role": "system", "content": req.system_prompt},
        {"role": "user", "content": req.user_prompt}
    ]
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": messages,
        "max_tokens": req.max_new_tokens or 1024,
        "temperature": 0.7
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(VLLM_SERVER, json=payload, timeout=120)
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return {"response": text.strip()}