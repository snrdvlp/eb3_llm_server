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
    device_map="auto"
)

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

health_model_id = "aashish12/eb3_quantized"

# Load tokenizer
health_tokenizer = AutoTokenizer.from_pretrained(health_model_id)

# Load quantized model
health_model = AutoModelForCausalLM.from_pretrained(
    health_model_id,
    device_map="auto",        # automatically put on GPU if available
    dtype=torch.float16, # or torch.float16 depending on GPU
    trust_remote_code=True,
    attn_implementation="flash_attention_2"

)

@app.post("/chat_health")
async def generate_text(prompt: str = "hi"):
    inputs = health_tokenizer(prompt, return_tensors="pt").to(health_model.device)
    outputs = health_model.generate( **inputs, max_new_tokens=300,  temperature=0.0, top_p=1.0, do_sample=False,  use_cache=True)
    response_text = health_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Try to extract JSON only
    start = response_text.find("{")
    end = response_text.rfind("}")

    if start != -1 and end != -1:
        json_text = response_text[start:end+1]
        try:
            data= json.loads(json_text)
            return data  # returns Python dict
        except json.JSONDecodeError as e:
            return {"insurance_details": response_text}

    print("\n\n\n\nraw response---->>", response_text)

    return {"insurance_details": response_text}
