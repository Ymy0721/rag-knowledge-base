#!/bin/bash
# 启动RAG知识库服务

# 出错时终止脚本
set -e

# 激活虚拟环境
source venv/bin/activate

# 设置工作目录为项目根目录
cd "$(dirname "$0")/.."

# 检查索引是否已构建
if [ ! -f "data/vector_db/faiss_index/faiss.index" ]; then
    echo "警告: 向量索引尚未构建，请先运行索引构建工具"
    exit 1
fi

# 获取模型路径
MODEL_PATH=$(grep -A 3 "embedding:" config/config.yaml | grep "model_path" | awk '{print $2}' | tr -d '"')

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 嵌入模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 获取配置中的API端口
PORT=$(grep -A 2 "api:" config/config.yaml | grep "port" | awk '{print $2}')
WORKERS=$(grep -A 3 "api:" config/config.yaml | grep "workers" | awk '{print $2}')

# 默认值
PORT=${PORT:-8000}
WORKERS=${WORKERS:-4}

# 启动服务
echo "启动API服务，端口: $PORT，工作进程数: $WORKERS"
exec uvicorn src.api.main:app --host localhost --port $PORT --workers $WORKERS
