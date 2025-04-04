import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import faiss
from src.utils import logger

class FAISSVectorStore:
    """使用FAISS实现向量存储，支持按需加载和释放内存"""
    
    def __init__(self, config: Dict):
        """
        初始化FAISS向量存储
        
        Args:
            config: 包含FAISS配置的字典
        """
        self.config = config
        self.index = None
        self.documents = []
        self.dimension = config['embedding']['dimension']
        self.index_type = config['faiss']['index_type']
        self.vector_store_path = os.path.join(os.getcwd(), config['data']['vector_store_path'])
        self.index_path = os.path.join(self.vector_store_path, "faiss.index")
        self.documents_path = os.path.join(self.vector_store_path, "documents.pkl")
        
        # 是否使用延迟加载模式
        self.lazy_loading = config.get('memory', {}).get('lazy_loading', False)
        # 是否在查询后卸载向量数据
        self.unload_after_query = config.get('memory', {}).get('unload_after_query', False)
        # 是否使用内存映射
        self.use_mmap = config.get('faiss', {}).get('use_mmap', True)
        
        # 延迟加载的标志
        self._index_loaded = False
        self._documents_loaded = False
        self._initialized = False
    
    def create_index(self):
        """创建新的FAISS索引"""
        if self.index_type == "IndexFlatIP":
            # 使用内积 (IP) 作为距离度量的扁平索引
            # 适用于已归一化的向量，计算余弦相似度
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexFlatL2":
            # 使用L2距离的扁平索引
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexIVFFlat":
            # 创建IVF索引以加快搜索速度
            # 先创建一个量化器
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = self.config['faiss']['nlist']
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
            # 在添加向量前，IVF索引需要训练
            # 但这将在添加实际数据时进行
        else:
            # 默认使用IndexFlatIP
            self.index = faiss.IndexFlatIP(self.dimension)
        
        logger.info(f"Created FAISS index of type {self.index_type} with dimension {self.dimension}")
        return self.index
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict]):
        """
        将文档向量添加到索引
        
        Args:
            embeddings: 嵌入向量的NumPy数组
            documents: 源文档的列表
        """
        if self.index is None:
            self.create_index()
            
        # 对于IVF索引，需要先训练
        if isinstance(self.index, faiss.IndexIVF) and not self.index.is_trained:
            if embeddings.shape[0] < self.config['faiss']['nlist']:
                logger.warning(f"Not enough vectors for training. Creating temporary random vectors.")
                # 创建临时随机向量进行训练
                train_vectors = np.random.random((self.config['faiss']['nlist'], self.dimension)).astype(np.float32)
                # 归一化向量
                faiss.normalize_L2(train_vectors)
            else:
                train_vectors = embeddings
                
            logger.info("Training IVF index...")
            self.index.train(train_vectors)
            
        # 确保向量是float32类型
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            
        # 归一化向量
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(embeddings)
            
        # 添加向量到索引
        start_id = len(self.documents)
        self.index.add(embeddings)
        
        # 存储文档
        self.documents.extend(documents)
        
        logger.info(f"Added {len(embeddings)} vectors to index. Total vectors: {len(self.documents)}")
    
    def initialize(self):
        """
        初始化向量存储，但不加载数据到内存
        验证索引和文档文件是否存在
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.documents_path):
            raise FileNotFoundError(f"Index or documents file not found at {self.vector_store_path}")
        
        logger.info("Vector store initialized. Files exist but not loaded into memory.")
        self._initialized = True
        return True
    
    def load_index(self) -> bool:
        """
        只加载FAISS索引，不加载文档数据
        
        Returns:
            是否成功加载
        """
        if self._index_loaded and self.index is not None:
            return True
            
        if not os.path.exists(self.index_path):
            logger.error(f"FAISS index file not found: {self.index_path}")
            return False
        
        try:
            logger.info(f"Loading FAISS index from {self.index_path}")
            if self.use_mmap and self.index_type == "IndexFlatIP":
                # 使用内存映射加载FlatIndex
                logger.info("Using memory-mapped loading for FAISS index")
                self.index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP)
            else:
                self.index = faiss.read_index(self.index_path)
                
            # 为IVF索引设置nprobe
            if isinstance(self.index, faiss.IndexIVF):
                self.index.nprobe = self.config['faiss'].get('nprobe', 10)
                
            self._index_loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
            return False
    
    def load_documents(self) -> bool:
        """
        只加载文档数据，不加载索引
        
        Returns:
            是否成功加载
        """
        if self._documents_loaded and self.documents:
            return True
            
        if not os.path.exists(self.documents_path):
            logger.error(f"Documents file not found: {self.documents_path}")
            return False
        
        try:
            logger.info(f"Loading documents from {self.documents_path}")
            with open(self.documents_path, 'rb') as f:
                self.documents = pickle.load(f)
                
            self._documents_loaded = True
            logger.info(f"Loaded {len(self.documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to load documents: {str(e)}")
            return False
    
    def load(self, use_mmap=True):
        """从磁盘加载索引和文档，可选择使用内存映射"""
        # 如果使用延迟加载，仅验证文件存在性
        if self.lazy_loading:
            return self.initialize()
        
        # 非延迟加载模式，立即加载索引和文档
        success_index = self.load_index()
        success_docs = self.load_documents()
        
        if not success_index or not success_docs:
            raise FileNotFoundError(f"Failed to load index or documents from {self.vector_store_path}")
            
        logger.info(f"Vector store loaded successfully with {len(self.documents)} documents")
        return True
    
    def unload_index(self):
        """卸载索引以释放内存"""
        if self.index is not None:
            del self.index
            self.index = None
            self._index_loaded = False
            logger.info("FAISS index unloaded from memory")
    
    def unload_documents(self):
        """卸载文档数据以释放内存"""
        if self.documents:
            self.documents = []
            self._documents_loaded = False
            logger.info("Documents unloaded from memory")
    
    def unload(self):
        """卸载索引和文档以释放内存"""
        self.unload_index()
        self.unload_documents()
        # 强制垃圾回收
        import gc
        gc.collect()
        logger.info("Vector store completely unloaded from memory")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Dict]]:
        """
        搜索最相似的向量，支持按需加载和释放功能
        
        Args:
            query_vector: 查询向量
            k: 返回的结果数量
            
        Returns:
            相似度分数和匹配的文档
        """
        # 按需加载索引和文档
        if not self._index_loaded:
            self.load_index()
        if not self._documents_loaded:
            self.load_documents()
        
        if self.index is None:
            raise ValueError("Index not initialized. Please add documents first or load an existing index.")
            
        # 确保查询向量是正确的形状和类型
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        # 对于IP索引类型，归一化查询向量
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(query_vector)
            
        # 执行搜索
        distances, indices = self.index.search(query_vector, k)
        
        # 收集匹配的文档
        matching_docs = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and 0 <= idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["score"] = float(distances[0][i])  # 添加相似度分数
                matching_docs.append(doc)
        
        # 如果配置为查询后卸载，则释放内存
        if self.unload_after_query:
            # 不要在这里立即卸载，让API层决定何时卸载
            pass
                
        return distances[0], matching_docs
    
    def save(self):
        """保存索引和文档到磁盘"""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # 保存FAISS索引
        logger.info(f"Saving FAISS index to {self.index_path}")
        faiss.write_index(self.index, self.index_path)
        
        # 保存文档
        logger.info(f"Saving {len(self.documents)} documents to {self.documents_path}")
        with open(self.documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
            
        logger.info("Vector store saved successfully")
