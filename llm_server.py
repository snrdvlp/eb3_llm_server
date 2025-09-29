import os
import logging

from fastapi import FastAPI, HTTPException
from vllm import LLM, SamplingParams

from pydantic import BaseModel
from transformers import AutoTokenizer
from typing import Optional

# --- Configuration ---
MODEL_DIR = os.environ.get("MODEL_DIR", "Qwen/Qwen2.5-7B-Instruct")
# Tune these depending on your prompts/latency tradeoffs:
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", 4096))
TENSOR_PARALLEL_SIZE = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", 1))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("DEFAULT_MAX_OUTPUT_TOKENS", 2048))
TORCH_DTYPE = os.environ.get("VLLM_TORCH_DTYPE", "float16")  # or "bfloat16" if supported

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_server_vllm")

app = FastAPI(title="vLLM FastAPI Server")

# Will be set on startup
llm_engine: Optional[LLM] = None
tokenizer = None

class LLMRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    max_new_tokens: Optional[int] = None

# -----------------------
# Startup / Shutdown
# -----------------------
@app.on_event("startup")
def on_startup():
    global llm_engine, tokenizer
    try:
        logger.info("Loading tokenizer from %s", MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

        logger.info("Starting vLLM engine (this may take a while)...")
        # Create the engine. Keep args minimal: vLLM handles defaults (no use_cache/continuous_batching).
        llm_engine = LLM(
            MODEL_DIR,
            dtype=TORCH_DTYPE,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        )

        logger.info("vLLM engine started successfully.")
        # optional quick warmup: a tiny generation to ensure everything initialized
        try:
            warm_prompt = "Hello."
            sp = SamplingParams(temperature=0.0, max_tokens=1, stop=["</s>"])
            out = next(llm_engine.generate([warm_prompt], sampling_params=sp))
            logger.info("Warmup OK.")
        except Exception:
            logger.warning("Warmup generation failed or is not necessary; continuing.")
    except Exception as e:
        logger.exception("Failed to initialize vLLM engine: %s", e)
        # Fail startup hard so operator sees it quickly
        raise

@app.on_event("shutdown")
def on_shutdown():
    global llm_engine
    try:
        if llm_engine is not None:
            logger.info("Closing vLLM engine...")
            # llm_engine.close()
    except Exception:
        logger.exception("Error closing vLLM engine.")

# -----------------------
# Helper: build chat text
# -----------------------
def build_chat_text(system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Use tokenizer helper to apply chat template then return plain string
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# -----------------------
# Main endpoint
# -----------------------
@app.post("/chat")
def chat(req: LLMRequest):
    global llm_engine, tokenizer
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM engine not ready")

    text = build_chat_text(req.system_prompt, req.user_prompt)

    # Estimate output tokens if not set
    max_out = req.max_new_tokens or DEFAULT_MAX_OUTPUT_TOKENS

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_out,
        stop=["</s>"]
    )

    try:
        # vLLM generate returns generator of GenerationResult objects
        outputs = llm_engine.generate([text], sampling_params=sampling_params)
        # take first result (single prompt)
        gen = outputs[0]
        return {"response": gen.outputs[0].text}
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))