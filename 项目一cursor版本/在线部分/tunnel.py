import atexit
import logging
import os
import time

from pyngrok import ngrok
from pyngrok.exception import PyngrokNgrokError

logger = logging.getLogger(__name__)


def cleanup():
    """退出时尝试关闭隧道，失败不影响主程序。"""
    try:
        ngrok.kill()
        logger.info("tunnel closed")
    except Exception:
        logger.exception("cleanup tunnel failed")


atexit.register(cleanup)


def start_tunnel(port: int = 8000):
    """启动 ngrok 隧道。失败时返回 None，不阻塞主服务启动。"""
    try:
        ngrok.kill()
    except Exception:
        pass

    token = os.getenv("NGROK_AUTHTOKEN", "").strip()
    if token:
        try:
            ngrok.set_auth_token(token)
        except Exception:
            logger.exception("set ngrok auth token failed")

    try:
        tunnel = ngrok.connect(port, "http")
        logger.info("tunnel started: %s", tunnel.public_url)
        print("=" * 50)
        print("✅ 内网穿透启动成功！")
        print("🌐 公网地址：", tunnel.public_url)
        print("=" * 50)
        return tunnel
    except PyngrokNgrokError as exc:
        logger.exception("ngrok tunnel failed")
        print("⚠️ 内网穿透启动失败，已自动忽略，继续启动本地服务。")
        print(f"原因：{exc}")
        return None
    except Exception as exc:
        logger.exception("unexpected tunnel error")
        print("⚠️ 内网穿透启动失败，已自动忽略，继续启动本地服务。")
        print(f"原因：{exc}")
        return None


def main():
    retry = 5
    while True:
        try:
            tunnel = start_tunnel()
            if tunnel is not None:
                print("运行中... 按 Ctrl+C 停止")
            else:
                print("运行中... 本地服务可用，按 Ctrl+C 停止")
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n退出程序")
            break
        except Exception as e:
            logger.exception("tunnel loop error")
            print(f"❌ 异常：{e}")
            print(f"{retry} 秒后重试...")
            time.sleep(retry)
            retry = min(retry * 2, 60)


if __name__ == "__main__":
    main()
