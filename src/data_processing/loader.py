import os
import pandas as pd
from typing import Dict, List, Optional, Union
from src.utils import logger

class DataLoader:
    """加载和处理Excel数据"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.raw_file = config['data']['raw_file']
        
    def load_excel_data(self, sheet_name: Optional[Union[str, int]] = 0) -> pd.DataFrame:
        """
        从Excel文件加载数据
        
        Args:
            sheet_name: 要加载的工作表名称或索引
            
        Returns:
            pandas DataFrame包含加载的数据
        """
        file_path = os.path.join(os.getcwd(), self.raw_file)
        logger.info(f"Loading data from {file_path}")
        
        try:
            # 使用chunks加载大型Excel文件以减少内存使用
            df = pd.read_excel(
                file_path, 
                sheet_name=sheet_name,
                engine='openpyxl'
            )
            logger.info(f"Successfully loaded {len(df)} rows from Excel file")
            return df
        
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            raise
    
    def get_column_names(self) -> List[str]:
        """获取Excel文件的列名"""
        try:
            df = pd.read_excel(
                os.path.join(os.getcwd(), self.raw_file),
                sheet_name=0,
                nrows=0,  # 只读取标题行
                engine='openpyxl'
            )
            return df.columns.tolist()
        except Exception as e:
            logger.error(f"Error getting column names: {str(e)}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> str:
        """
        保存处理后的数据
        
        Args:
            df: 要保存的DataFrame
            filename: 输出文件名
            
        Returns:
            保存的文件路径
        """
        processed_dir = os.path.join(os.getcwd(), self.config['data']['processed_dir'])
        os.makedirs(processed_dir, exist_ok=True)
        
        output_path = os.path.join(processed_dir, filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Processed data saved to {output_path}")
        
        return output_path
