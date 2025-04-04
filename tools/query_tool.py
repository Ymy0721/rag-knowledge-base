import os
import sys
import argparse
import yaml
import gc
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embedding import BGEEmbedder
from src.vector_store import FAISSVectorStore
from src.utils import logger

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='查询知识库')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--query', type=str, help='查询文本')
    parser.add_argument('--top-k', type=int, default=5, help='返回的结果数量')
    parser.add_argument('--low-memory', action='store_true', help='启用低内存模式')
    args = parser.parse_args()
    
    # 如果没有提供查询，进入交互模式
    interactive = args.query is None
    
    # 加载配置
    config = load_config(args.config)
    
    # 如果指定了低内存模式，覆盖配置
    if args.low_memory:
        if 'server' not in config:
            config['server'] = {}
        config['server']['low_memory_mode'] = True
        config['memory'] = {
            'lazy_loading': True,
            'unload_after_query': True
        }
        logger.info("启用低内存模式")
    
    try:
        # 始终加载模型
        logger.info("加载嵌入模型...")
        embedder = BGEEmbedder(config)
        embedder.load_model()
        
        # 初始化向量存储但不加载数据
        logger.info("初始化向量存储...")
        vector_store = FAISSVectorStore(config)
        vector_store.initialize()
        
        def process_query(query, top_k):
            query_start = time.time()
            
            # 嵌入查询
            query_embedding = embedder.embed_text(query)
            
            # 搜索向量存储（按需加载数据）
            distances, results = vector_store.search(query_embedding, k=top_k)
            
            # 如果开启了向量数据卸载功能，查询后释放数据内存
            if config.get('memory', {}).get('unload_after_query', False):
                vector_store.unload()
            
            query_time = time.time() - query_start
            
            # 打印结果
            print(f"\n查询: \"{query}\"  (耗时: {query_time:.2f}秒)\n")
            print(f"找到 {len(results)} 个结果:")
            
            for i, doc in enumerate(results):
                print(f"\n{i+1}. 相似度: {doc['score']:.4f}")
                print(f"文本: {doc['text'][:200]}...")
                if 'metadata' in doc:
                    print("元数据:")
                    for key, value in doc['metadata'].items():
                        print(f"  {key}: {value}")
            
        if interactive:
            print("知识库查询工具 (输入 'exit' 或 'quit' 退出)")
            while True:
                query = input("\n请输入查询: ")
                if query.lower() in ['exit', 'quit']:
                    break
                if not query.strip():
                    continue
                    
                try:
                    process_query(query, args.top_k)
                except Exception as e:
                    print(f"查询处理出错: {str(e)}")
        else:
            process_query(args.query, args.top_k)
            
    except Exception as e:
        logger.error(f"查询工具出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
