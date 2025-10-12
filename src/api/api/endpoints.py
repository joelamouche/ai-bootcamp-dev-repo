from fastapi import APIRouter,Request
from src.api.api.models import RagRequest,RagResponse
import logging
from qdrant_client import QdrantClient
from api.rag.retrieval_generation import rag_pipeline

qdrant_client = QdrantClient(url="http://qdrant:6333")


logger=logging.getLogger(__name__)  

rag_router=APIRouter()

@rag_router.post("/")
def rag(
    request:Request,
    payload:RagRequest
    )->RagResponse:

    answer=rag_pipeline(payload.query,qdrant_client)

    return RagResponse(request_id=request.state.request_id,answer=answer["answer"])

api_router=APIRouter()
api_router.include_router(rag_router,prefix="/rag",tags=["rag"])