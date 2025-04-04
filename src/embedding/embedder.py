from typing import Dict, List, Union, Optional, Tuple
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.utils import logger

class BGEEmbedder:
    """使用BGE中文向量模型进行文本嵌入"""
    
    def __init__(self, config: Dict):
        """
        初始化嵌入器
        
        Args:
            config: 配置字典，包含模型路径和维度信息
        """
        self.config = config
        self.model_path = config['embedding']['model_path']
        self.dimension = config['embedding']['dimension']
        self.batch_size = config['embedding']['batch_size']
        self.normalize = config['embedding'].get('normalize_embeddings', True)
        self.use_gpu = config['embedding'].get('use_gpu', torch.cuda.is_available())
        self.model = None
        
        # 记录环境信息
        if self.use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
            logger.info(f"GPU加速已启用 - {gpu_info}")
        else:
            self.device = "cpu"
            logger.info("使用CPU模式运行")
        
    def load_model(self):
        """加载嵌入模型"""
        if self.model is not None:
            return
            
        try:
            logger.info(f"Loading BGE embedding model from {self.model_path}")
            
            # 使用sentence-transformers加载BGE模型
            self.model = SentenceTransformer(self.model_path, device=self.device)
            
            # 打印内存占用信息
            model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            logger.info(f"BGE model loaded successfully (approximate size: {model_size_mb:.2f} MB)")
        except Exception as e:
            logger.error(f"Error loading BGE embedding model: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        为单个文本生成嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        if self.model is None:
            self.load_model()
            
        try:
            # 生成文本嵌入
            embedding = self.model.encode(text, normalize_embeddings=self.normalize, show_progress_bar=False)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros(self.dimension)
    
    def embed_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        为一批文本生成嵌入向量
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度
            
        Returns:
            嵌入向量的NumPy数组
        """
        if self.model is None:
            self.load_model()
            
        try:
            # 批量生成嵌入
            embeddings = self.model.encode(
                texts, 
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=show_progress
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            # 返回零矩阵
            return np.zeros((len(texts), self.dimension))
    
    def embed_documents(self, documents: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """
        为文档列表生成嵌入
        
        Args:
            documents: 包含文本和元数据的文档字典列表
            
        Returns:
            嵌入向量的NumPy数组和对应的元数据列表
        """
        texts = [doc["text"] for doc in documents]
        logger.info(f"开始为 {len(texts)} 个文本块生成嵌入向量...")
        
        # 使用更大的批处理大小如果是在GPU上运行
        if self.use_gpu and torch.cuda.is_available():
            original_batch_size = self.batch_size
            self.batch_size = min(128, len(texts))  # 在GPU上使用更大的批处理大小
            logger.info(f"在GPU模式下使用批处理大小: {self.batch_size}")
        
        embeddings = self.embed_batch(texts)
        
        # 恢复原始批处理大小
        if self.use_gpu and torch.cuda.is_available() and hasattr(self, 'original_batch_size'):
            self.batch_size = original_batch_size
        
        # 确保每个文档都有一个有效的嵌入向量
        valid_embeddings = []
        valid_documents = []
        
        for i, (embedding, document) in enumerate(zip(embeddings, documents)):
            if not np.all(embedding == 0):  # 检查是否为全零向量
                valid_embeddings.append(embedding)
                valid_documents.append(document)
            else:
                logger.warning(f"Skipping document at index {i} with zero embedding")
        
        logger.info(f"成功生成 {len(valid_embeddings)} 个有效嵌入向量")
        return np.array(valid_embeddings), valid_documents
    
    def unload_model(self):
        """释放模型以节省内存"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("嵌入模型已卸载以释放内存")
