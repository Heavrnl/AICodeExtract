# 基于 Python 官方镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 仅复制 requirements.txt 并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 暴露应用程序的端口
EXPOSE 5000

# 启动 Flask 应用
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
