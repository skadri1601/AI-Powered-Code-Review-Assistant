# service/monitoring.py

"""
Monitoring and metrics collection for the AI Code Review Assistant.
"""

import time
import logging
from functools import wraps
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import redis
from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
REVIEW_COUNT = Counter('reviews_generated_total', 'Total reviews generated')
REVIEW_DURATION = Histogram('review_generation_seconds', 'Review generation time')
MODEL_LOAD_TIME = Gauge('model_load_seconds', 'Time taken to load model')
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')

# Redis client for caching (optional)
try:
    redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
    redis_available = redis_client.ping()
except:
    redis_client = None
    redis_available = False

logger = logging.getLogger(__name__)

class MetricsMiddleware:
    """Middleware for collecting HTTP metrics."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        start_time = time.time()
        
        ACTIVE_REQUESTS.inc()
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=status_code
                ).inc()
                
                REQUEST_DURATION.observe(time.time() - start_time)
                ACTIVE_REQUESTS.dec()
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

def monitor_review_generation(func):
    """Decorator to monitor review generation performance."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            REVIEW_COUNT.inc()
            REVIEW_DURATION.observe(time.time() - start_time)
            return result
        except Exception as e:
            logger.error(f"Review generation failed: {e}")
            raise
    return wrapper

def monitor_model_loading(func):
    """Decorator to monitor model loading time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            MODEL_LOAD_TIME.set(time.time() - start_time)
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    return wrapper

class CacheManager:
    """Simple caching layer for review suggestions."""
    
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self.enabled = redis_available
    
    def get_cache_key(self, diff_hunk: str) -> str:
        """Generate cache key for diff hunk."""
        import hashlib
        return f"review:{hashlib.sha256(diff_hunk.encode()).hexdigest()}"
    
    async def get(self, diff_hunk: str) -> Optional[str]:
        """Get cached review suggestion."""
        if not self.enabled:
            return None
        
        try:
            key = self.get_cache_key(diff_hunk)
            return redis_client.get(key)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None
    
    async def set(self, diff_hunk: str, suggestion: str) -> None:
        """Cache review suggestion."""
        if not self.enabled:
            return
        
        try:
            key = self.get_cache_key(diff_hunk)
            redis_client.setex(key, self.ttl, suggestion)
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

# Global cache instance
cache_manager = CacheManager()

class HealthChecker:
    """Health check utilities."""
    
    @staticmethod
    async def check_model_health() -> Dict[str, Any]:
        """Check if the model is loaded and responsive."""
        try:
            from .app import get_pipeline
            pipeline = get_pipeline()
            
            # Test with a simple input
            test_input = "@@ -1,1 +1,2 @@\n def test():\n+    pass"
            result = pipeline(test_input, max_length=32, num_beams=2)
            
            return {
                "status": "healthy",
                "model_loaded": True,
                "test_successful": bool(result)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_loaded": False,
                "error": str(e)
            }
    
    @staticmethod
    async def check_github_integration() -> Dict[str, Any]:
        """Check GitHub App integration health."""
        try:
            from .github_app import GitHubAppClient
            client = GitHubAppClient()
            
            # Test JWT generation
            jwt_token = client.get_jwt_token()
            
            return {
                "status": "healthy",
                "github_configured": True,
                "jwt_generation": bool(jwt_token)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "github_configured": False,
                "error": str(e)
            }
    
    @staticmethod
    async def check_cache_health() -> Dict[str, Any]:
        """Check Redis cache health."""
        if not redis_available:
            return {
                "status": "disabled",
                "cache_available": False
            }
        
        try:
            redis_client.ping()
            return {
                "status": "healthy",
                "cache_available": True,
                "redis_connected": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "cache_available": False,
                "error": str(e)
            }

# Metrics endpoint
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Structured logging setup
def setup_logging():
    """Configure structured logging."""
    import json
    import sys
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            if hasattr(record, 'request_id'):
                log_entry['request_id'] = record.request_id
            
            if hasattr(record, 'user_id'):
                log_entry['user_id'] = record.user_id
            
            return json.dumps(log_entry)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler with JSON formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

# Performance monitoring context manager
@asynccontextmanager
async def monitor_performance(operation_name: str):
    """Context manager for monitoring operation performance."""
    start_time = time.time()
    logger.info(f"Starting operation: {operation_name}")
    
    try:
        yield
        duration = time.time() - start_time
        logger.info(f"Operation completed: {operation_name}", extra={
            "operation": operation_name,
            "duration": duration,
            "status": "success"
        })
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Operation failed: {operation_name}", extra={
            "operation": operation_name,
            "duration": duration,
            "status": "error",
            "error": str(e)
        })
        raise

# Request ID middleware for tracing
class RequestIDMiddleware:
    """Add unique request ID for tracing."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        import uuid
        request_id = str(uuid.uuid4())
        scope["request_id"] = request_id
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                message["headers"] = [
                    *message.get("headers", []),
                    [b"x-request-id", request_id.encode()]
                ]
            await send(message)
        
        await self.app(scope, receive, send_wrapper)