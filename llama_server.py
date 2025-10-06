from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load model
model_name = "meta-llama/Llama-2-7b-hf"

# Specify `use_auth_token=True` if needed
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype="auto",
    use_auth_token=True
)

class Request(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.post("/generate")
def generate(req: Request):
    inputs = tokenizer(req.prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=req.max_tokens)
    return {"text": tokenizer.decode(outputs[0], skip_special_tokens=True)}

# Test endpoint
@app.get("/")
def root():
    return {"status": "LLaMA-2 7B API running"}
