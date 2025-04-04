from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np

from src.embedding.embedder import BGEEmbedder
from src.vector_store.faiss_store import FAISSVectorStore
from src.utils import logger

# 将在main.py中注册此路由器
router = APIRouter(
    prefix="/api/v1",
    tags=["knowledge base"]
)

# 请求和响应模型
class DocumentQuery(BaseModel):
    query: str
    top_k: int = 5
    filter_metadata: Optional[Dict] = None

class DocumentResponse(BaseModel):
    text: str
    metadata: Dict
    score: float

class QueryResponse(BaseModel):
    results: List[DocumentResponse]
    query: str

# 依赖项
def get_embedder(embedder: BGEEmbedder = Depends()):
    return embedder

def get_vector_store(vector_store: FAISSVectorStore = Depends()):
    return vector_store

@router.post("/search", response_model=QueryResponse)
async def search_documents(
    request: DocumentQuery,
    embedder: BGEEmbedder = Depends(get_embedder),
    vector_store: FAISSVectorStore = Depends(get_vector_store)
):
    """
    搜索与查询语义相关的文档
    """
    try:
        # 嵌入查询文本
        query_embedding = embedder.embed_text(request.query)
        
        # 搜索向量存储
        _, raw_results = vector_store.search(query_embedding, k=request.top_k)
        
        # 应用元数据过滤器（如果提供）
        results = []
        for doc in raw_results:
            if request.filter_metadata:
                # 检查文档是否满足过滤条件
                match = True
                for key, value in request.filter_metadata.items():
                    if key not in doc.get("metadata", {}) or doc["metadata"][key] != value:
                        match = False
                        break
                
                if not match:
                    continue
            
            # 转换为响应格式
            results.append(DocumentResponse(
                text=doc["text"],
                metadata=doc.get("metadata", {}),
                score=doc.get("score", 0.0)
            ))
        
        return QueryResponse(
            results=results,
            query=request.query
        )
    except Exception as e:
        logger.error(f"Error in search_documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
