import time

import json

import requests

from config import LLM_API_BASE, LLM_MODEL_NAME, LLM_WAIT_SECONDS


def call_llm(messages):
    time.sleep(LLM_WAIT_SECONDS)
    url = f"{LLM_API_BASE}/api/chat"
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": messages,
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {})
        content = message.get("content", "")
        if not content:
            raise ValueError("Ollama 返回内容为空")
        return content.strip()
    except (requests.RequestException, ValueError, KeyError) as e:
        raise RuntimeError(f"Ollama 接口错误：{str(e)}") from e


def stream_llm(messages):
    time.sleep(LLM_WAIT_SECONDS)
    url = f"{LLM_API_BASE}/api/chat"
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": messages,
        "stream": True,
    }
    try:
        with requests.post(url, json=payload, timeout=300, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=False):
                if not line:
                    continue
                if isinstance(line, bytes):
                    if line.startswith(b"data: "):
                        line = line[6:]
                    chunk = line.decode("utf-8", errors="ignore").strip()
                else:
                    chunk = str(line).strip()
                    if chunk.startswith("data: "):
                        chunk = chunk[6:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                except Exception:
                    continue
                message = data.get("message", {})
                content = message.get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama 流式接口错误：{str(e)}") from e
