import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger
from src.serving.metrics import PREDICTION_LATENCY, PREDICTION_ERRORS


logger = get_logger(__name__, log_file= 'logs/serving.log')


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with:
      - request_id (UUID)
      - method, path, status_code
      - latency (ms)
    Also records latency to Prometheus.
    """
    
    
    async def dispatch(self,
                       request: Request, 
                       call_next):
        
        request_id = str(uuid.uuid4())[:8]  # short UUID for tracing
        start_time = time.perf_counter()
        
        
        # Attach request_id to request state for downstream use
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            latency = time.perf_counter() - start_time
            
            logger.info(
                f"[{request_id}] {request.method} {request.url.path} "
                f"→ {response.status_code} ({latency*1000:.1f}ms)"
            )
            
            
            endpoint = request.url.path
            PREDICTION_LATENCY.labels(
                model_version= 'v1.0',
                endpoint= endpoint
            ).observe(latency)
            
            response.headers["X-Request-ID"]      = request_id
            response.headers["X-Response-Time-ms"] = f"{latency*1000:.1f}"
            
            return response
        
        
        except Exception as exc:
            latency = time.perf_counter() - start_time
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} "
                f"→ ERROR ({latency*1000:.1f}ms) — {exc}"
            )
            PREDICTION_ERRORS.labels(error_type=type(exc).__name__).inc()
            raise