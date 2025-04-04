from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from src.utils import logger

class TextChunker:
    """将文本分割成固定大小的块"""
    
    def __init__(self, config: Dict):
        """
        初始化文本分块器
        
        Args:
            config: 包含分块参数的配置字典
        """
        self.chunk_size = config['preprocessing']['chunk_size']
        self.chunk_overlap = config['preprocessing']['chunk_overlap']
    
    def chunk_text(self, text: str, include_metadata: bool = True, 
                  metadata: Optional[Dict] = None) -> List[Dict]:
        """
        将长文本分块
        
        Args:
            text: 要分块的文本
            include_metadata: 是否在每个块中包含元数据
            metadata: 可选的元数据字典
            
        Returns:
            包含文本块和元数据的字典列表
        """
        if not text or not isinstance(text, str):
            return []
            
        text = text.strip()
        if not text:
            return []
            
        # 中文环境下，可以按字符直接分割
        if len(text) <= self.chunk_size:
            chunk_dict = {"text": text}
            if include_metadata and metadata:
                chunk_dict["metadata"] = metadata
            return [chunk_dict]
            
        chunks = []
        start = 0
        
        while start < len(text):
            # 计算当前块的结束位置
            end = min(start + self.chunk_size, len(text))
            
            # 创建当前块
            chunk_text = text[start:end]
            chunk_dict = {"text": chunk_text}
            
            if include_metadata and metadata:
                # 为每个块添加源文档信息和块位置信息
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_start"] = start
                chunk_metadata["chunk_end"] = end
                chunk_dict["metadata"] = chunk_metadata
                
            chunks.append(chunk_dict)
            
            # 移动到下一个块的起始位置，考虑重叠
            start += self.chunk_size - self.chunk_overlap
            
            # 确保起始位置不会超过文本长度
            if start >= len(text):
                break
                
        return chunks
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str, 
                         id_column: Optional[str] = None,
                         metadata_columns: Optional[List[str]] = None) -> List[Dict]:
        """
        处理DataFrame，将文本列分块
        
        Args:
            df: 包含文本的DataFrame
            text_column: 包含要分块文本的列名
            id_column: 可选的ID列名
            metadata_columns: 要包含在元数据中的列名列表
            
        Returns:
            包含所有文本块的列表
        """
        all_chunks = []
        
        for i, row in df.iterrows():
            if text_column not in row or not isinstance(row[text_column], str):
                continue
                
            text = row[text_column]
            
            # 准备元数据
            metadata = {}
            if id_column and id_column in row:
                metadata["id"] = row[id_column]
            else:
                metadata["id"] = i  # 使用行索引作为ID
                
            if metadata_columns:
                for col in metadata_columns:
                    if col in row:
                        metadata[col] = row[col]
            
            # 分块
            chunks = self.chunk_text(text, include_metadata=True, metadata=metadata)
            all_chunks.extend(chunks)
            
            # 每处理100行记录一次日志
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1} rows, generated {len(all_chunks)} chunks so far")
        
        logger.info(f"Text chunking complete. Generated {len(all_chunks)} chunks from {len(df)} documents")
        return all_chunks
