import os
import sys
import argparse
import yaml
import time
import pandas as pd
import torch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import DataLoader, TextPreprocessor, TextChunker
from src.embedding import BGEEmbedder
from src.vector_store import FAISSVectorStore
from src.utils import logger

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='使用GPU构建知识库索引')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--columns', type=str, nargs='+', help='要处理的Excel列名')
    parser.add_argument('--id-column', type=str, help='文档ID列名')
    parser.add_argument('--metadata-columns', type=str, nargs='+', help='要包含为元数据的列名')
    parser.add_argument('--gpu', action='store_true', help='强制启用GPU')
    parser.add_argument('--output-dir', type=str, help='索引输出目录')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 强制启用GPU
    if args.gpu:
        if torch.cuda.is_available():
            config['embedding']['use_gpu'] = True
            logger.info(f"GPU加速已启用 - 使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("指定了GPU但未检测到可用的CUDA设备，将使用CPU")
    
    # 设置输出目录
    if args.output_dir:
        config['data']['vector_store_path'] = args.output_dir
        logger.info(f"索引将保存到: {args.output_dir}")
    
    start_time = time.time()
    logger.info("开始构建知识库索引...")
    
    try:
        # 1. 加载Excel数据
        loader = DataLoader(config)
        df = loader.load_excel_data()
        
        # 如果未指定列名，获取所有文本列
        if not args.columns:
            logger.info("未指定处理列，将显示所有可用列...")
            columns = loader.get_column_names()
            logger.info(f"可用列: {columns}")
            return
            
        # 2. 预处理文本
        preprocessor = TextPreprocessor(config)
        processed_df = preprocessor.process_dataframe(df, args.columns)
        
        # 合并指定的列到单个文本列
        combined_df = preprocessor.combine_columns(processed_df, args.columns, new_column_name="combined_text")
        
        # 3. 文本分块
        chunker = TextChunker(config)
        chunks = chunker.process_dataframe(
            combined_df, 
            text_column="combined_text",
            id_column=args.id_column,
            metadata_columns=args.metadata_columns
        )
        
        logger.info(f"生成了 {len(chunks)} 个文本块")
        
        # 4. 生成嵌入向量
        embedder = BGEEmbedder(config)
        embeddings, documents = embedder.embed_documents(chunks)
        
        # 5. 构建向量索引
        vector_store = FAISSVectorStore(config)
        vector_store.create_index()
        vector_store.add_documents(embeddings, documents)
        
        # 6. 保存索引
        vector_store.save()
        
        elapsed_time = time.time() - start_time
        logger.info(f"索引构建完成！处理了 {len(df)} 行，生成了 {len(chunks)} 个文本块。耗时: {elapsed_time:.2f} 秒")
        
        # 显示索引文件位置
        index_path = os.path.join(os.getcwd(), config['data']['vector_store_path'])
        logger.info(f"索引文件保存在: {index_path}")
        logger.info("您可以将此目录中的文件复制到服务器上相同的路径位置")
        
    except Exception as e:
        logger.error(f"构建索引时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
