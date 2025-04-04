#!/bin/bash
# Ubuntu服务器环境配置脚本

# 出错时终止脚本
set -e

echo "开始配置Ubuntu环境..."

# 更新包列表
echo "更新包列表..."
apt-get update

# 安装Python相关工具
echo "安装Python和相关工具..."
apt-get install -y python3 python3-pip python3-venv build-essential

# 创建Python虚拟环境
echo "创建Python虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 安装项目依赖
echo "安装项目依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 创建必要的目录
echo "创建项目目录..."
mkdir -p data/raw data/processed data/vector_db logs models

echo "环境配置完成！"
echo "请将原始Excel文件放在 data/raw/ 目录中"
echo "请将嵌入模型文件放在 models/ 目录中"
