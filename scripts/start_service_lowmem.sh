#!/bin/bash
# 启动低内存模式的RAG知识库服务

# 出错时终止脚本
set -e

# 激活虚拟环境
source venv/bin/activate

# 设置工作目录为项目根目录
cd "$(dirname "$0")/.."

# 检查索引和文档数据库是否已构建
if [ ! -f "data/vector_db/faiss_index/faiss.index" ] || [ ! -f "data/vector_db/faiss_index/documents.db" ]; then
    echo "警告: 向量索引 (faiss.index) 或文档数据库 (documents.db) 尚未构建，请先运行索引构建工具或复制预构建的文件"
    exit 1
fi

# 获取模型路径
MODEL_PATH=$(grep -A 3 "embedding:" config/config.yaml | grep "model_path" | awk '{print $2}' | tr -d '"')

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "警告: 嵌入模型路径不存在: $MODEL_PATH"
    echo "将尝试在运行时自动下载模型"
fi

# 更新配置为低内存模式
cat > config/server_config.yaml << EOF
# 服务器端低内存配置 (覆盖 config.yaml 中的部分设置)
embedding:
  use_gpu: false
  batch_size: 8

server:
  low_memory_mode: true

faiss:
  use_mmap: false             # <--- 确认: 禁用 mmap
  nprobe: 10

memory:
  lazy_loading: true          # <--- 确认: 启用惰性加载
  unload_after_query: true    # <--- 确认: 查询后卸载

api:
  workers: 1                  # 强制单进程
EOF

# 获取配置中的API端口
PORT=$(grep -A 2 "api:" config/config.yaml | grep "port" | awk '{print $2}')
PORT=${PORT:-8000}

# 获取可用内存信息
MEM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{print $2}')
MEM_FREE=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
MEM_TOTAL_MB=$((MEM_TOTAL / 1024))
MEM_FREE_MB=$((MEM_FREE / 1024))

echo "系统内存: 总计 ${MEM_TOTAL_MB}MB, 可用 ${MEM_FREE_MB}MB"
echo "启动低内存模式的API服务，端口: $PORT，仅使用1个工作进程"
echo "配置: FAISS mmap=false, Lazy Loading=true (SQLite), Unload After Query=true"

# 启动服务 (确保 main.py 能正确加载 server_config.yaml 来覆盖默认值)
# 假设 load_config() 函数已实现合并逻辑
exec uvicorn src.api.main:app --host localhost --port $PORT --workers 1 --log-level info
