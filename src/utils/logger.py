import logging
import os
import yaml
from logging.config import dictConfig
import datetime

def setup_logging(config_path='config/logging_config.yaml'):
    """设置日志配置"""
    # 确保日志目录存在
    os.makedirs('logs', exist_ok=True)
    
    if os.path.exists(config_path):
        with open(config_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        
        # 添加日期到日志文件名
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        if 'file' in config['handlers']:
            config['handlers']['file']['filename'] = f"logs/rag-knowledge-base-{today}.log"
        
        dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.warning(f"Logging config file not found at {config_path}. Using basic configuration.")
    
    return logging.getLogger()

# 创建默认日志记录器实例
logger = setup_logging()
