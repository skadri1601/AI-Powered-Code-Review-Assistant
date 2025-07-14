# service/app.py

import os
import hashlib
from functools import lru_cache
from typing import List

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ─── Config ───────────────────────────────────────────────────────────
MODEL_DIR = os.getenv("REVIEW_MODEL_PATH", "../checkpoints/codet5-finetuned/final")
CACHE_SIZE = 128

# ─── Load Model & Tokenizer ────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1  # CPU; if GPU available, set device=0
    )

# ─── Request/Response Schemas ───────────────────────────────────────────
class ReviewRequest(BaseModel):
    diff_hunks: List[str]

class ReviewResponse(BaseModel):
    suggestions: List[str]

# ─── App & Endpoints ─────────────────────────────────────────────────────
app = FastAPI(title="AI Code Review Assistant")

@app.post("/review", response_model=ReviewResponse)
async def review(req: ReviewRequest):
    gen = get_pipeline()
    suggestions = []

    for hunk in req.diff_hunks:
        # Use a simple cache key per hunk
        key = hashlib.sha256(hunk.encode()).hexdigest()
        # NOTE: lru_cache can’t cache inside loop; for production swap in Redis or similar
        output = gen(hunk, max_length=64, num_beams=4, early_stopping=True)
        # pipeline returns a list of dicts: [{"generated_text": "..."}]
        suggestions.append(output[0]["generated_text"].strip())

    return ReviewResponse(suggestions=suggestions)

# ─── Healthcheck ─────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}
