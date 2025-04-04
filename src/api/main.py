import os
import yaml
import json
import gc
import time
from typing import List, Dict, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from src.embedding.embedder import BGEEmbedder
from src.vector_store.faiss_store import FAISSVectorStore
from src.utils import logger

# 加载配置
def load_config():
    config_path = os.path.join(os.getcwd(), "config/config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config()

# 确定是否使用低内存模式
low_memory_mode = config.get('server', {}).get('low_memory_mode', False)
load_model_on_demand = False  # 始终加载模型到内存中，不按需加载
enable_model_unloading = False  # 不允许卸载模型

# 初始化应用，添加root_path参数以适应反向代理
app = FastAPI(
    title="RAG Knowledge Base API",
    description="API for querying the RAG knowledge base",
    version=config['app']['version'],
    root_path=config.get('api', {}).get('root_path', ''),  # 适应反向代理的路径前缀
)

# 初始化向量存储和嵌入模型
vector_store = FAISSVectorStore(config)
embedder = BGEEmbedder(config)

# 在启动时加载索引和模型
@app.on_event("startup")
async def startup_event():
    # 始终加载模型到内存，无论是否为低内存模式
    logger.info("Loading embedding model...")
    embedder.load_model()
    
    # 对于向量存储，仅验证文件存在性，不预加载数据到内存
    logger.info("Initializing vector store...")
    try:
        vector_store.initialize()  # 只初始化，不加载数据
    except FileNotFoundError:
        logger.warning("Vector store not found. Please build the index first.")
        # 创建一个空索引
        vector_store.create_index()
    
    if low_memory_mode:
        logger.info("Running in low memory mode - vector data will be loaded on demand")

# 在服务关闭时清理资源
@app.on_event("shutdown")
async def shutdown_event():
    # 不卸载模型，但可以卸载向量数据
    vector_store.unload()
    logger.info("Vector store unloaded")

# 请求和响应模型
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    results: List[Dict]
    query: str
    execution_time: float

@app.post("/api/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    查询知识库
    
    Args:
        request: 包含查询字符串和结果数量的请求
        
    Returns:
        匹配的文档列表
    """
    start_time = time.time()
    try:
        # 模型已经加载，直接使用
        # 嵌入查询
        query_embedding = embedder.embed_text(request.query)
        
        # 搜索向量存储（向量存储会自动处理按需加载和释放）
        _, results = vector_store.search(query_embedding, k=request.top_k)
        
        # 如果开启了向量数据按需卸载功能，可以在后台任务中卸载数据
        if low_memory_mode and config.get('memory', {}).get('unload_after_query', False):
            background_tasks.add_task(vector_store.unload)
        
        execution_time = time.time() - start_time
        
        # 返回结果
        return QueryResponse(
            results=results,
            query=request.query,
            execution_time=execution_time
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    mem_info = {}
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = {
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "model_loaded": embedder.model is not None,
            "vector_data_loaded": vector_store._index_loaded and vector_store._documents_loaded
        }
    except ImportError:
        mem_info = {
            "model_loaded": embedder.model is not None,
            "vector_data_loaded": vector_store._index_loaded and vector_store._documents_loaded
        }
    
    return {
        "status": "healthy", 
        "version": config['app']['version'],
        "memory_usage": mem_info,
        "low_memory_mode": low_memory_mode
    }
