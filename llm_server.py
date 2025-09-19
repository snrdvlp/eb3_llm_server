from fastapi import FastAPI, Body
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

MODEL_DIR = "Qwen/Qwen2.5-14B-Instruct"  # Change to your actual model path

app = FastAPI()

class LLMRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    max_new_tokens: int = 1024

# Load model and tokenizer at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)
try:
    gen_config = GenerationConfig.from_pretrained(MODEL_DIR)
except Exception:
    gen_config = None

@app.post("/chat")
async def chat_endpoint(req: LLMRequest):
    prompt = f"{req.system_prompt}\n\n{req.user_prompt}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    gen_args = {
        "input_ids": input_ids,
        "max_new_tokens": req.max_new_tokens,
    }
    if gen_config:
        gen_args["generation_config"] = gen_config
    generated_ids = model.generate(**gen_args)
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # Optionally strip prompt from output
    response = output[len(prompt):].strip() if output.startswith(prompt) else output.strip()
    return {"response": response}