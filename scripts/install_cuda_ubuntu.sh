#!/bin/bash
# =============================================================================
# CUDA 安装脚本（Ubuntu）
# =============================================================================
# 作用：在 Ubuntu 上自动安装 NVIDIA 驱动和 CUDA Toolkit
# 支持版本：Ubuntu 20.04 / 22.04 / 24.04
# =============================================================================

set -e

echo "=========================================="
echo "CUDA 安装脚本"
echo "=========================================="

# ------------------------------------------------------------------------------
# 检查系统信息
# ------------------------------------------------------------------------------
echo ""
echo "[1/8] 检查系统信息..."

UBUNTU_VERSION=$(lsb_release -rs)
echo "Ubuntu 版本: $UBUNTU_VERSION"

# 检查 GPU
if ! lspci | grep -i nvidia > /dev/null; then
    echo "错误：未检测到 NVIDIA GPU"
    echo "请确认服务器有 NVIDIA 显卡"
    exit 1
fi

echo "检测到 NVIDIA GPU"

# ------------------------------------------------------------------------------
# 更新系统
# ------------------------------------------------------------------------------
echo ""
echo "[2/8] 更新系统包..."

sudo apt update
sudo apt upgrade -y

# ------------------------------------------------------------------------------
# 安装依赖
# ------------------------------------------------------------------------------
echo ""
echo "[3/8] 安装依赖..."

sudo apt install -y \
    build-essential \
    dkms \
    linux-headers-$(uname -r) \
    wget \
    curl \
    gnupg \
    software-properties-common

# ------------------------------------------------------------------------------
# 添加 NVIDIA 驱动 PPA
# ------------------------------------------------------------------------------
echo ""
echo "[4/8] 添加 NVIDIA 驱动源..."

# 添加 graphics-drivers PPA
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt update

# ------------------------------------------------------------------------------
# 安装 NVIDIA 驱动
# ------------------------------------------------------------------------------
echo ""
echo "[5/8] 安装 NVIDIA 驱动..."

# 检测推荐的驱动版本
RECOMMENDED_DRIVER=$(ubuntu-drivers devices | grep "recommended" | awk '{print $3}')

if [ -n "$RECOMMENDED_DRIVER" ]; then
    echo "推荐驱动: $RECOMMENDED_DRIVER"
    sudo apt install -y "$RECOMMENDED_DRIVER"
else
    echo "安装默认驱动 nvidia-driver-535..."
    sudo apt install -y nvidia-driver-535
fi

# ------------------------------------------------------------------------------
# 安装 CUDA Toolkit
# ------------------------------------------------------------------------------
echo ""
echo "[6/8] 安装 CUDA Toolkit..."

# 根据 Ubuntu 版本选择 CUDA 版本
if [[ "$UBUNTU_VERSION" == "22.04" ]] || [[ "$UBUNTU_VERSION" == "24.04" ]]; then
    CUDA_VERSION="12.4"
    CUDA_PACKAGE="cuda-toolkit-12-4"
elif [[ "$UBUNTU_VERSION" == "20.04" ]]; then
    CUDA_VERSION="12.4"
    CUDA_PACKAGE="cuda-toolkit-12-4"
else
    echo "未测试的 Ubuntu 版本，使用默认 CUDA 12.4"
    CUDA_VERSION="12.4"
    CUDA_PACKAGE="cuda-toolkit-12-4"
fi

echo "安装 CUDA $CUDA_VERSION..."

# 下载 CUDA 安装包
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(echo $UBUNTU_VERSION | tr -d .)/x86_64/cuda-keyring_1.1-1_all.deb

sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y $CUDA_PACKAGE

# ------------------------------------------------------------------------------
# 配置环境变量
# ------------------------------------------------------------------------------
echo ""
echo "[7/8] 配置环境变量..."

# 添加到 .bashrc
if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA 环境变量" >> ~/.bashrc
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo "环境变量已添加到 ~/.bashrc"
fi

# 立即生效
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# ------------------------------------------------------------------------------
# 验证安装
# ------------------------------------------------------------------------------
echo ""
echo "[8/8] 验证安装..."

echo ""
echo "NVIDIA 驱动版本:"
nvidia-smi

echo ""
echo "CUDA 版本:"
nvcc --version

echo ""
echo "=========================================="
echo "CUDA 安装完成！"
echo "=========================================="
echo ""
echo "请重启系统以完成驱动加载:"
echo "  sudo reboot"
echo ""
echo "重启后验证:"
echo "  nvidia-smi"
echo "  nvcc --version"
echo ""
