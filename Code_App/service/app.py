# service/app.py

import os
import hashlib
import logging
from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import uvicorn

from .github_app import GitHubAppClient, CodeReviewBot

# ─── Logging Setup ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────
MODEL_DIR = os.getenv("REVIEW_MODEL_PATH", "../checkpoints/codet5-finetuned/final")
CACHE_SIZE = 128
WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")

# ─── Load Model & Tokenizer ────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_pipeline():
    """Load and cache the ML pipeline."""
    logger.info(f"Loading model from {MODEL_DIR}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1  # CPU; if GPU available, set device=0
        )
        logger.info("Model loaded successfully")
        return pipe
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# ─── Initialize GitHub App Client ───────────────────────────────────────
try:
    github_client = GitHubAppClient()
    review_bot = CodeReviewBot(github_client, "http://localhost:8000")
    logger.info("GitHub App client initialized")
except Exception as e:
    logger.warning(f"GitHub App not configured: {e}")
    github_client = None
    review_bot = None

# ─── Request/Response Schemas ───────────────────────────────────────────
class ReviewRequest(BaseModel):
    diff_hunks: List[str]
    max_length: Optional[int] = 64
    num_beams: Optional[int] = 4

class ReviewResponse(BaseModel):
    suggestions: List[str]
    model_info: dict

class WebhookPayload(BaseModel):
    action: str
    pull_request: Optional[dict] = None
    repository: Optional[dict] = None
    installation: Optional[dict] = None

# ─── App & Middleware ─────────────────────────────────────────────────────
app = FastAPI(
    title="AI Code Review Assistant",
    description="AI-powered code review suggestions using fine-tuned CodeT5",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── API Endpoints ────────────────────────────────────────────────────────

@app.post("/review", response_model=ReviewResponse)
async def review(req: ReviewRequest):
    """Generate code review suggestions for given diff hunks."""
    try:
        gen = get_pipeline()
        suggestions = []

        logger.info(f"Processing {len(req.diff_hunks)} diff hunks")
        
        for i, hunk in enumerate(req.diff_hunks):
            logger.debug(f"Processing hunk {i+1}/{len(req.diff_hunks)}")
            
            # Truncate very long hunks
            if len(hunk) > 512:
                hunk = hunk[:512] + "..."
            
            try:
                output = gen(
                    hunk, 
                    max_length=req.max_length,
                    num_beams=req.num_beams,
                    early_stopping=True,
                    do_sample=False  # Deterministic output
                )
                suggestion = output[0]["generated_text"].strip()
                suggestions.append(suggestion)
            except Exception as e:
                logger.error(f"Error processing hunk {i}: {e}")
                suggestions.append("Unable to generate suggestion for this code change.")

        model_info = {
            "model_path": MODEL_DIR,
            "num_hunks_processed": len(req.diff_hunks),
            "parameters": {
                "max_length": req.max_length,
                "num_beams": req.num_beams
            }
        }

        return ReviewResponse(suggestions=suggestions, model_info=model_info)
    
    except Exception as e:
        logger.error(f"Error in review endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle GitHub webhook events."""
    if not review_bot:
        raise HTTPException(status_code=501, detail="GitHub App not configured")
    
    try:
        # Verify signature
        signature = request.headers.get("X-Hub-Signature-256")
        if not signature:
            raise HTTPException(status_code=400, detail="Missing signature")
        
        body = await request.body()
        if not github_client.verify_webhook_signature(body, signature):
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        # Parse payload
        import json
        payload = json.loads(body)
        
        event_type = request.headers.get("X-GitHub-Event")
        logger.info(f"Received {event_type} event for PR #{payload.get('pull_request', {}).get('number', 'unknown')}")
        
        if event_type == "pull_request":
            # Process in background to avoid webhook timeout
            background_tasks.add_task(review_bot.handle_pull_request, payload)
        
        return {"status": "ok"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        # Test model loading
        get_pipeline()
        model_status = "healthy"
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        model_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "ok",
        "model_status": model_status,
        "github_app_configured": review_bot is not None,
        "model_path": MODEL_DIR
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI Code Review Assistant",
        "version": "1.0.0",
        "endpoints": {
            "review": "/review - POST - Generate review suggestions",
            "webhook": "/webhook - POST - GitHub webhook endpoint",
            "health": "/health - GET - Health check",
            "docs": "/docs - GET - API documentation"
        }
    }

# ─── Development Server ───────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "service.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )