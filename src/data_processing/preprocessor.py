import re
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from src.utils import logger

class TextPreprocessor:
    """中文文本预处理类"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def clean_text(self, text: str) -> str:
        """
        清理文本：删除多余空格、特殊字符等
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        if not isinstance(text, str):
            return ""
            
        # 替换多个空格为单个空格
        text = re.sub(r'\s+', ' ', text)
        
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 去除URL
        text = re.sub(r'http\S+', '', text)
        
        # 删除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        return text.strip()
    
    def process_dataframe(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """
        处理DataFrame中的文本列
        
        Args:
            df: 输入DataFrame
            text_columns: 要处理的文本列名列表
            
        Returns:
            处理后的DataFrame
        """
        logger.info(f"Processing {len(df)} rows of data")
        
        # 创建副本避免修改原始数据
        processed_df = df.copy()
        
        for column in text_columns:
            if column in processed_df.columns:
                logger.info(f"Cleaning text in column: {column}")
                processed_df[column] = processed_df[column].apply(self.clean_text)
            else:
                logger.warning(f"Column {column} not found in DataFrame")
        
        # 删除空行
        processed_df = processed_df.dropna(subset=text_columns, how='all')
        processed_df = processed_df.reset_index(drop=True)
        
        logger.info(f"Text preprocessing complete. Resultant dataframe has {len(processed_df)} rows")
        return processed_df
    
    def combine_columns(self, df: pd.DataFrame, columns_to_combine: List[str], 
                        separator: str = " ", new_column_name: str = "combined_text") -> pd.DataFrame:
        """
        合并多个列到一个文本列
        
        Args:
            df: 输入DataFrame
            columns_to_combine: 要合并的列名列表
            separator: 列之间的分隔符
            new_column_name: 新列的名称
            
        Returns:
            添加了合并文本列的DataFrame
        """
        def combine_row(row):
            texts = []
            for col in columns_to_combine:
                if col in row and isinstance(row[col], str) and row[col].strip():
                    texts.append(row[col].strip())
            return separator.join(texts)
        
        result_df = df.copy()
        result_df[new_column_name] = df.apply(combine_row, axis=1)
        
        logger.info(f"Combined columns {columns_to_combine} into new column '{new_column_name}'")
        return result_df
