"""pytest 共享配置：自动添加项目根到 sys.path"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent  # 在线部分/
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# 禁用外部依赖连接（环境变量标记）
import os
os.environ["AUTH_BACKEND"] = "sqlite"
os.environ["MILVUS_HOST"] = "127.0.0.1"

import logging
logging.disable(logging.CRITICAL)
