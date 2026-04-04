from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.api.middleware import RequestIdMiddleware
from api.api.endpoints import api_router
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


app = FastAPI()

app.add_middleware(RequestIdMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],    
    allow_headers=["*"],
)

app.include_router(api_router)


@app.middleware("http")
async def log_all_requests(request: Request, call_next):
    """Log every HTTP request so we can confirm Streamlit (or any client) reaches this process."""
    client = request.client.host if request.client else "?"
    logger.info("http_request method=%s path=%s client=%s", request.method, request.url.path, client)
    response = await call_next(request)
    logger.info(
        "http_response method=%s path=%s status=%s",
        request.method,
        request.url.path,
        response.status_code,
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok", "service": "api"}


@app.post("/")
def root(request:Request):
    return {"message":"API"}

