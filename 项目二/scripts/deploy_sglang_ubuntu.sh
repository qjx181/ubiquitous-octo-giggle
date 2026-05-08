#!/bin/bash
# =============================================================================
# SGLang 部署脚本（Ubuntu）
# =============================================================================
# 作用：在 Ubuntu 服务器上自动部署 SGLang 服务
# 使用方法：
#   1. 将本脚本上传到 Ubuntu 服务器
#   2. chmod +x deploy_sglang_ubuntu.sh
#   3. ./deploy_sglang_ubuntu.sh
# =============================================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "SGLang 部署脚本"
echo "=========================================="

# ------------------------------------------------------------------------------
# 配置参数（可根据需要修改）
# ------------------------------------------------------------------------------
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"  # 模型名称
PORT=30000                              # 服务端口
HOST="0.0.0.0"                          # 监听地址（0.0.0.0 允许远程访问）
WORK_DIR="$HOME/sglang_server"          # 工作目录
VENV_DIR="$WORK_DIR/venv"               # 虚拟环境目录

# ------------------------------------------------------------------------------
# 检查系统环境
# ------------------------------------------------------------------------------
echo ""
echo "[1/6] 检查系统环境..."

# 检查是否 NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误：未检测到 nvidia-smi，请确保已安装 NVIDIA 驱动"
    exit 1
fi

echo "GPU 信息："
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 检查 CUDA
if ! command -v nvcc &> /dev/null; then
    echo "警告：未检测到 CUDA，SGLang 可能无法正常使用 GPU"
    echo "建议安装 CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
fi

# ------------------------------------------------------------------------------
# 创建工作环境
# ------------------------------------------------------------------------------
echo ""
echo "[2/6] 创建工作环境..."

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "工作目录: $WORK_DIR"

# ------------------------------------------------------------------------------
# 创建 Python 虚拟环境
# ------------------------------------------------------------------------------
echo ""
echo "[3/6] 创建 Python 虚拟环境..."

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "虚拟环境创建完成"
else
    echo "虚拟环境已存在，跳过创建"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# 升级 pip
pip install --upgrade pip

# ------------------------------------------------------------------------------
# 安装依赖
# ------------------------------------------------------------------------------
echo ""
echo "[4/6] 安装 SGLang..."

# 安装 PyTorch（CUDA 版本）
pip install torch torchvision torchaudio

# 安装 SGLang
pip install sglang

echo "SGLang 安装完成"

# ------------------------------------------------------------------------------
# 下载模型
# ------------------------------------------------------------------------------
echo ""
echo "[5/6] 准备模型: $MODEL_NAME..."

# 模型会在首次启动时自动下载
# 也可以预先下载：
# huggingface-cli download $MODEL_NAME

echo "模型将在首次启动时自动下载"

# ------------------------------------------------------------------------------
# 创建启动脚本
# ------------------------------------------------------------------------------
echo ""
echo "[6/6] 创建启动脚本..."

cat > "$WORK_DIR/start.sh" << 'EOF'
#!/bin/bash
# SGLang 启动脚本

WORK_DIR="$HOME/sglang_server"
VENV_DIR="$WORK_DIR/venv"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
PORT=30000
HOST="0.0.0.0"

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# 启动 SGLang 服务
echo "启动 SGLang 服务..."
echo "模型: $MODEL_NAME"
echo "地址: http://$HOST:$PORT"
echo ""

python -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT"
EOF

chmod +x "$WORK_DIR/start.sh"

# 创建 systemd 服务文件（可选，用于开机自启）
cat > "$WORK_DIR/sglang.service" << EOF
[Unit]
Description=SGLang LLM Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR
ExecStart=$WORK_DIR/start.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "=========================================="
echo "部署完成！"
echo "=========================================="
echo ""
echo "启动方式 1 - 手动启动:"
echo "  $WORK_DIR/start.sh"
echo ""
echo "启动方式 2 - 后台运行:"
echo "  cd $WORK_DIR && nohup ./start.sh > sglang.log 2>&1 &"
echo ""
echo "启动方式 3 - 系统服务（需 sudo）:"
echo "  sudo cp $WORK_DIR/sglang.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable sglang"
echo "  sudo systemctl start sglang"
echo ""
echo "查看日志:"
echo "  tail -f $WORK_DIR/sglang.log"
echo ""
echo "测试服务:"
echo "  curl http://localhost:$PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}'"
echo ""
echo "Windows 后端配置:"
echo "  set SGLANG_HOST=http://<Ubuntu-IP>:$PORT"
echo ""
