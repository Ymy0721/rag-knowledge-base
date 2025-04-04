import requests
import json
import argparse
import sys

class RAGClient:
    """RAG知识库API客户端"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        
    def query(self, query_text, top_k=5):
        """
        查询知识库
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            查询结果
        """
        url = f"{self.base_url}/api/query"
        payload = {
            "query": query_text,
            "top_k": top_k
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # 如果请求失败，抛出异常
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"查询请求失败: {str(e)}")
            return None

def main():
    parser = argparse.ArgumentParser(description="RAG知识库API客户端")
    parser.add_argument("--url", default="http://localhost:8000", help="API基础URL")
    parser.add_argument("--query", required=True, help="查询文本")
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数量")
    parser.add_argument("--format", choices=['pretty', 'json'], default='pretty', help="输出格式")
    
    args = parser.parse_args()
    
    client = RAGClient(args.url)
    results = client.query(args.query, args.top_k)
    
    if not results:
        sys.exit(1)
        
    if args.format == 'json':
        print(json.dumps(results))
    else:
        print(f"\n查询: \"{results['query']}\"")
        print(f"执行时间: {results['execution_time']:.2f}秒")
        print(f"找到 {len(results['results'])} 个结果:\n")
        
        for i, result in enumerate(results['results']):
            print(f"{i+1}. 相似度: {result['score']:.4f}")
            print(f"   文本: {result['text'][:200]}...")
            if 'metadata' in result:
                print("   元数据:")
                for key, value in result['metadata'].items():
                    print(f"     {key}: {value}")
            print()

if __name__ == "__main__":
    main()
