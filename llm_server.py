from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import json

MODEL_DIR = "Qwen/Qwen2.5-14B-Instruct"

app = FastAPI()

class LLMRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    max_new_tokens: int = 1024
# Load model + tokenizer at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    load_in_8bit=True,

)
model.eval()

try:
    gen_config = GenerationConfig.from_pretrained(MODEL_DIR)
except Exception:
    gen_config = None

@app.post("/chat")
async def chat_endpoint(req: LLMRequest):
    try:
        messages = [
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": req.user_prompt}
        ]

        # Build prompt for assistant reply
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True   # 👈 ensures assistant role is ready
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0
        )

        # Decode & clean response
        response = tokenizer.decode(
            generated_ids[0][model_inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True   # 👈 removes <|im_start|>, <|im_end|>, <|endoftext|>
        ).strip()

        return {"response": response}

    except Exception as e:
        return {"error": str(e)}