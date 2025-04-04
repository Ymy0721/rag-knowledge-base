#!/bin/bash
# 将RAG知识库设置为系统服务

# 出错时终止脚本
set -e

# 检查是否以root权限运行
if [ "$(id -u)" != "0" ]; then
   echo "此脚本必须以root权限运行" 1>&2
   exit 1
fi

# 设置项目路径
APP_PATH="/app/rag-knowledge-base"
SERVICE_NAME="rag-knowledge-base"

# 检查项目目录是否存在
if [ ! -d "$APP_PATH" ]; then
    echo "错误: 项目目录 $APP_PATH 不存在"
    exit 1
fi

# 创建systemd服务文件
cat > /etc/systemd/system/${SERVICE_NAME}.service << EOF
[Unit]
Description=RAG Knowledge Base Service
After=network.target

[Service]
User=root
Group=root
WorkingDirectory=${APP_PATH}
ExecStart=${APP_PATH}/venv/bin/python -m uvicorn src.api.main:app --host localhost --port 8000 --workers 1
Restart=always
RestartSec=5
StartLimitInterval=0
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "服务文件已创建: /etc/systemd/system/${SERVICE_NAME}.service"

# 重新加载systemd配置
systemctl daemon-reload

# 启用服务开机自启
systemctl enable ${SERVICE_NAME}.service
echo "服务已设置为开机自启"

# 启动服务
systemctl start ${SERVICE_NAME}.service
echo "服务已启动"

# 显示服务状态
systemctl status ${SERVICE_NAME}.service

echo "----------------------------------------"
echo "RAG知识库服务已成功配置为系统服务!"
echo "服务名称: ${SERVICE_NAME}"
echo "项目路径: ${APP_PATH}"
echo ""
echo "管理命令:"
echo "  启动服务:   sudo systemctl start ${SERVICE_NAME}"
echo "  停止服务:   sudo systemctl stop ${SERVICE_NAME}"
echo "  重启服务:   sudo systemctl restart ${SERVICE_NAME}"
echo "  查看日志:   sudo journalctl -u ${SERVICE_NAME} -f"
echo "----------------------------------------"
