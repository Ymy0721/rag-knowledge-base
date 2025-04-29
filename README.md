# RAG知识库

基于检索增强生成（RAG）的知识库系统，使用FAISS作为向量数据库（支持PQ压缩），SQLite存储文档内容，并使用BGE中文嵌入模型进行文本向量化。

## 功能特点

- 从Excel表格加载并处理中文知识库数据
- 使用BGE中文嵌入模型(bge-small-zh-v1.5)进行文本向量化
- 基于FAISS实现高性能向量检索 (支持 `IndexIVFPQ` 压缩索引)
- **使用SQLite存储文档内容，实现高效的按需检索**
- 提供REST API接口供大模型调用
- 针对CPU环境优化，内存占用低
- **支持文档数据惰性加载和查询后卸载，进一步减少常驻内存**
- ~~支持FAISS索引内存映射 (mmap) 加载~~ (当前配置禁用mmap以优先考虑PQ压缩)

## 目录结构

```
rag-knowledge-base/
├── config/                   # 配置文件目录
├── data/                     # 数据目录
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后的数据
│   └── vector_db/            # FAISS索引和SQLite文档数据库存储
│       ├── faiss.index       # FAISS 索引文件
│       └── documents.db      # SQLite 文档数据库文件
├── models/                   # 模型目录
├── src/                      # 源代码
├── tools/                    # 工具脚本
├── scripts/                  # 部署和维护脚本
├── tests/                    # 测试代码
├── requirements.txt          # 项目依赖
└── README.md                 # 项目说明
```

## 快速开始

### Windows开发环境配置

1. 克隆此项目
2. 创建并激活虚拟环境:
```
python -m venv venv
venv\Scripts\activate
```
3. 安装依赖:
```
pip install -r requirements.txt
```
4. 下载BGE中文嵌入模型:
```
# 从HuggingFace下载bge-small-zh-v1.5模型
# 或使用sentence-transformers自动下载
# 模型将存储在models/bge-small-zh-v1.5目录
```
5. 将知识库Excel文件放入data/raw目录

### 构建索引

```bash
# 这会生成 faiss.index 和 documents.db 文件
python tools/build_index.py --columns 列名1 列名2 --id-column ID列名 --metadata-columns 元数据列1 元数据列2
```

### 测试查询

```
python tools/query_tool.py --query "您的查询文本"
```

### 启动API服务

```
uvicorn src.api.main:app --host localhost --port 8000
```

## Ubuntu部署

1. 将项目复制到Ubuntu服务器
2. 运行环境配置脚本:
```
sudo bash scripts/setup_ubuntu.sh
```
3. 构建索引:
```
source venv/bin/activate
python3 tools/build_index.py --columns 列名1 列名2 --id-column ID列名
```
4. 启动服务:
```
bash scripts/start_service.sh
```

## API接口

### 知识库查询

**端点**: `/api/query`

**方法**: POST

**请求体**:
```json
{
  "query": "查询文本",
  "top_k": 5
}
```

**响应**:
```json
{
  "results": [
    {
      "text": "匹配的文本内容...",
      "metadata": {
        "id": "doc_id",
        "其他元数据字段": "值"
      },
      "score": 0.95
    }
  ],
  "query": "查询文本"
}
```

**服务配置**

1. 添加rag-knowledge-base服务
```bash
sudo vim /etc/systemd/system/rag-knowledge-base.service
```
2. 按照如下配置进行修改
```bash
[Unit]
Description=RAG Knowledge Base Service
After=network.target

[Service]
User=root
Group=root
WorkingDirectory=/app/rag-knowledge-base
ExecStart=/app/rag-knowledge-base/venv/bin/python -m uvicorn src.api.main:app --host localhost --port 8000 --workers 1
Restart=always
RestartSec=5
StartLimitInterval=0
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```
3. 设置开机自启动
```bash
sudo systemctl start rag-knowledge-base.service
sudo systemctl enable rag-knowledge-base.service
sudo systemctl status rag-knowledge-base.service
```

**nginx配置**

1. 创建配置文件
```bash
sudo vim /etc/nginx/sites-available/rag-knowledge-base
```
2. 创建符号链接到sites-enabled目录
```bash
sudo ln -s /etc/nginx/sites-available/rag-knowledge-base /etc/nginx/sites-enabled/
```
3. 按如下配置进行修改
```nginx
server {
    listen 80;
    server_name your-server-domain-or-ip;
    
    location /rag/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
        proxy_buffering off;
        proxy_read_timeout 300s;
    }
    # 可能的其他配置...
}
```
4. 检查语法错误并重启nginx
```bash
sudo nginx -t
sudo systemctl reload nginx
```

**使用curl测试API**
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"查询文本","top_k":3}'
```

## 性能优化

- 使用BGE中文模型提供更好的中文文本理解
- FAISS使用 `IndexIVFPQ` 压缩索引以减少内存占用和磁盘空间
- **SQLite存储**: 使用SQLite数据库存储文档，避免一次性加载所有文档到内存，实现高效的按ID检索。
- 批处理以优化资源使用
- 调整分块大小以平衡精度和性能
- 向量归一化以提高相似度计算准确性 (如果使用IP度量)
- **惰性加载 (Lazy Loading)**: 通过 `memory.lazy_loading: true` 配置，服务启动时不加载索引或连接数据库，仅在首次查询时加载索引并按需连接数据库。
- **查询后卸载 (Unload After Query)**: 配合惰性加载，通过 `memory.unload_after_query: true` 配置，在每次查询结束后释放加载的资源（如果配置了卸载索引，则卸载索引；数据库连接通常按需建立和关闭，此设置影响不大）。
- ~~内存映射 (Memory Mapping)~~: 当前配置 (`faiss.use_mmap: false`) 禁用mmap，优先使用 `IndexIVFPQ` 进行内存优化。
