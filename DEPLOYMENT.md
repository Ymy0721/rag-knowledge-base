# 部署指南：Windows构建 + 服务器查询

本文档提供了如何在Windows上使用GPU构建向量数据库，然后在低内存服务器上部署查询服务的详细步骤。

## 1. Windows环境准备

### 环境设置
1. 创建并激活虚拟环境:
```
python -m venv venv
venv\Scripts\activate
```

2. 安装依赖:
```
pip install -r requirements.txt
```

3. 安装PyTorch GPU版本(如果需要):
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```

4. 验证GPU可用:
```
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}, 设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无"}')"
```

### 下载模型
1. 使用sentence-transformers自动下载:
```
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-small-zh-v1.5')"
```

2. 模型将被下载到models/bge-small-zh-v1.5目录

## 2. 构建向量数据库

1. 将Excel文件放入data/raw目录

2. 运行GPU构建脚本:
```
python tools/build_index_gpu.py --gpu --columns 发明名称 摘要 --metadata-columns 发明名称 摘要申请号 公开（公告）号 申请日 公开（公告）日 IPC分类号 申请（专利权）人 发明人 申请人邮编 代理人 代理机构 文献类型 申请人所在国（省）
```

3. 脚本将生成两个关键文件:
   - `data/vector_db/faiss_index/faiss.index`: FAISS索引文件
   - `data/vector_db/faiss_index/documents.pkl`: 文档元数据文件

## 3. 将索引文件传输到服务器

有多种方式可以传输文件:

### 使用SCP(Linux/Mac)
```bash
scp -r data/vector_db/faiss_index user@server_ip:/path/to/rag-knowledge-base/data/vector_db/
```

### 使用WinSCP(Windows)
1. 下载并安装WinSCP
2. 连接到服务器
3. 复制`data/vector_db/faiss_index`目录到服务器相应位置

### 使用Python脚本(适用于大文件)
在项目根目录下创建`tools/compress_index.py`:

```python
import os
import shutil
import zipfile

# 压缩索引文件
vector_db_path = 'data/vector_db/faiss_index'
zip_file = 'vector_db_index.zip'

with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(vector_db_path):
        for file in files:
            zipf.write(os.path.join(root, file))

print(f"索引已压缩到 {zip_file}")
```

然后复制生成的zip文件到服务器并解压。

## 4. 服务器部署

1. 在服务器上克隆项目(如果尚未克隆):
```bash
git clone <repository_url> rag-knowledge-base
cd rag-knowledge-base
```

2. 设置环境:
```bash
sudo bash scripts/setup_ubuntu.sh
```

3. 启动低内存模式服务:
```bash
bash scripts/start_service_lowmem.sh
```

## 5. 内存占用优化

如果服务器内存仍然不足，可以尝试以下优化:

1. 使用IVF索引类型减少内存占用，编辑`config/config.yaml`:
```yaml
faiss:
  index_type: "IndexIVFFlat"
  nlist: 100
  nprobe: 10
```

2. 减小文本块大小，降低维度:
```yaml
preprocessing:
  chunk_size: 128
  chunk_overlap: 20
```

3. 考虑在Nginx后面以单进程模式运行服务:
```bash
uvicorn src.api.main:app --host localhost --port 8000 --workers 1
```

## 6. 监控与故障排除

### 监控内存使用
```bash
watch -n 1 "ps -o pid,user,%mem,rss,command -p \$(pgrep -f uvicorn)"
```

### 常见问题

1. **模型加载失败**: 
   - 确保模型目录存在或者自动下载配置正确
   - 检查磁盘空间是否足够

2. **索引加载失败**:
   - 确保两个索引文件(faiss.index和documents.pkl)都已正确传输
   - 检查文件路径是否与配置文件一致

3. **服务崩溃**:
   - 检查日志文件(logs/rag-knowledge-base-*.log)
   - 可能是内存不足，尝试进一步减少批处理大小和工作进程数

## 7. 性能调优

1. 对于高QPS场景，考虑将索引载入共享内存:
```python
# 在vector_store.py中添加共享内存支持
import mmap
```

2. 增加GPU服务器时更新配置:
```yaml
embedding:
  use_gpu: true
server:
  low_memory_mode: false
  load_model_on_demand: false
```
