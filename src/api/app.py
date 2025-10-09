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

@app.post("/")
def root(request:Request):
    return {"message":"API"}

