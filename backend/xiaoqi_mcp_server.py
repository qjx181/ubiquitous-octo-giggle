"""
xiaoqi_mcp_server.py — 小漆 MCP Server

让 Hermes（二哥）能调用小漆的能力：
  - 代码分析、审查、建议
  - 项目结构洞察
  - 问题诊断与修复建议
  - 架构分析与设计决策

用法（Stdio 模式，给 Hermes 用）:
    python xiaoqi_mcp_server.py

连接到 Hermes Agent（在 config.yaml 中添加）:
    mcp_servers:
      xiaoqi:
        command: "D:\\an\\envs\\project2\\python.exe"
        args: ["C:\\path\\to\\xiaoqi_mcp_server.py"]
        timeout: 300

依赖:
    pip install mcp
"""

import asyncio
import json
import os
import sys
import logging
import subprocess
import textwrap
from contextlib import contextmanager
from typing import Any, Optional
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
logger = logging.getLogger("xiaoqi-mcp")


# =============================================================================
# _QuietStdout — 拯救 MCP 管道的神器
# =============================================================================
# MCP Stdio 模式用 stdout 传递 JSON-RPC 消息。
# 任何 stray print / tqdm 进度条都会污染管道，让 MCP 客户端解析失败。
# 这个 context manager 临时把 stdout 重定向到 stderr，确保管道纯净。
# =============================================================================

@contextmanager
def _QuietStdout():
    """临时将 sys.stdout 重定向到 sys.stderr，保护 MCP 管道"""
    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = original_stdout


# =============================================================================
# 工具实现
# =============================================================================

def _safe_read_project_file(rel_path: str) -> Optional[str]:
    """安全地读取项目文件（限制在项目目录内）"""
    # 确定项目根目录（xiaoqi_mcp_server.py 所在目录的父目录）
    script_dir = Path(__file__).resolve().parent  # backend/
    project_root = script_dir.parent               # 项目根目录

    # 安全检查：防止路径穿越
    target = (project_root / rel_path).resolve()
    if not str(target).startswith(str(project_root)):
        return None, "路径越界，不允许读取项目目录外的文件"

    if not target.exists():
        return None, f"文件不存在: {rel_path}"

    if target.is_dir():
        return None, f"目标是一个目录，请指定文件路径"

    try:
        content = target.read_text(encoding="utf-8")
        return content, None
    except Exception as e:
        return None, f"读取失败: {e}"


def _get_git_status() -> dict:
    """获取 git 工作状态"""
    project_root = Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "status", "--short"],
            capture_output=True, text=True, timeout=10,
        )
        branch = subprocess.run(
            ["git", "-C", str(project_root), "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
        )
        return {
            "branch": branch.stdout.strip(),
            "changes": result.stdout.strip() or "无未提交变更",
            "change_count": len([l for l in result.stdout.split("\n") if l.strip()]),
        }
    except Exception as e:
        return {"error": str(e)}


def _get_project_tree(max_depth: int = 3) -> str:
    """生成项目目录树（限制深度，排除忽略目录）"""
    project_root = Path(__file__).resolve().parent.parent
    ignore_dirs = {".git", "__pycache__", ".idea", "data", "logs", "__pycache__"}
    ignore_exts = {".pyc", ".pdf", ".pptx"}

    lines = []
    root_name = project_root.name

    def _walk(dir_path: Path, depth: int, prefix: str = ""):
        if depth > max_depth:
            return
        try:
            entries = sorted(dir_path.iterdir(), key=lambda x: (x.is_file(), x.name))
        except PermissionError:
            return

        entries = [e for e in entries
                   if e.name not in ignore_dirs
                   and e.suffix not in ignore_exts]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")

            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                _walk(entry, depth + 1, next_prefix)
            else:
                size = entry.stat().st_size
                size_str = f"{size/1024:.1f}K" if size > 1024 else f"{size}B"
                lines.append(f"{prefix}{connector}{entry.name} ({size_str})")

    _walk(project_root, 1)
    return f"{root_name}/\n" + "\n".join(lines)


def _find_issues() -> list[dict]:
    """扫描项目中的常见问题标记"""
    project_root = Path(__file__).resolve().parent.parent
    issues = []
    search_patterns = [
        ("TODO", "待办事项"),
        ("FIXME", "需修复"),
        ("HACK", "临时方案"),
        ("XXX", "需注意"),
        ("BUG", "潜在缺陷"),
    ]

    for pattern, label in search_patterns:
        try:
            result = subprocess.run(
                ["grep", "-rn", pattern, "--include=*.py", "--include=*.md",
                 "--include=*.sh", "--include=*.bat", "--include=*.json",
                 "-l", str(project_root)],
                capture_output=True, text=True, timeout=10,
            )
            files = [f for f in result.stdout.strip().split("\n") if f]
            if files:
                # 只保留相对路径
                rel_files = [os.path.relpath(f, project_root) for f in files[:10]]
                issues.append({
                    "type": label,
                    "pattern": pattern,
                    "files": rel_files,
                    "count": len(files),
                })
        except Exception:
            pass

    return issues


def _find_large_files(min_kb: int = 500) -> list[dict]:
    """查找项目中较大的文件"""
    project_root = Path(__file__).resolve().parent.parent
    large = []
    for f in project_root.rglob("*"):
        if f.is_file() and f.suffix not in {".pyc", ".pdf", ".pptx", ".zip"}:
            try:
                rel = os.path.relpath(f, project_root)
                if rel.startswith(".git") or rel.startswith("data") or rel.startswith("__pycache__"):
                    continue
                size_kb = f.stat().st_size / 1024
                if size_kb >= min_kb:
                    large.append({"file": rel, "size_kb": round(size_kb, 1)})
            except Exception:
                pass
    large.sort(key=lambda x: x["size_kb"], reverse=True)
    return large[:15]


# =============================================================================
# DeepSeek 调用（让小漆能"思考"）
# =============================================================================

import httpx

async def _call_deepseek(prompt: str, system_prompt: str = "") -> str:
    """调用 DeepSeek API 处理需要推理的请求"""
    env_path = Path(__file__).resolve().parent / ".env"
    api_key = os.environ.get("DEEPSEEK_KEY")
    api_host = os.environ.get("DEEPSEEK_HOST", "https://api.deepseek.com/v1")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    # 尝试从 .env 文件读取
    if not api_key and env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("DEEPSEEK_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                elif line.startswith("DEEPSEEK_HOST="):
                    api_host = line.split("=", 1)[1].strip()
                elif line.startswith("DEEPSEEK_MODEL="):
                    model = line.split("=", 1)[1].strip()

    if not api_key:
        return "❌ DeepSeek API 密钥未配置，无法完成推理。请检查 backend/.env 中的 DEEPSEEK_KEY。"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{api_host}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 4096,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"DeepSeek API 调用失败: {e}")
        return f"❌ 调用 DeepSeek 失败: {e}"


# =============================================================================
# MCP 主入口
# =============================================================================

def main_stdio():
    """Stdio 模式"""
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    logger.info("🚀 启动小漆 MCP Server (Stdio 模式)")

    # 初始化时避免 tqdm 污染 stdout
    # 这里没有需要静默的初始化步骤，但保留机制以备后续
    logger.info("🖌️ 小漆已上线，等二哥来调!")

    server = Server("xiaoqi")

    # -- 工具列表 -----------------------------------------------------------

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="think",
                description="让小漆思考和回答问题。适合需要推理、分析、决策的场景。"
                            "比如：分析架构、审查设计、回答技术问题。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "需要小漆思考的问题或任务描述",
                        },
                        "context": {
                            "type": "string",
                            "description": "额外上下文信息（可选），如代码片段、日志等",
                        },
                    },
                    "required": ["question"],
                },
            ),
            Tool(
                name="analyze_code",
                description="深入分析代码文件。返回结构、逻辑、潜在问题、改进建议。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "相对于项目根目录的文件路径，如 backend/app/core/retrieval.py",
                        },
                        "focus": {
                            "type": "string",
                            "description": "分析重点（可选）：architecture / performance / bugs / security / readability",
                            "enum": ["architecture", "performance", "bugs", "security", "readability"],
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="review_changes",
                description="审查 Git 中未提交的代码变更，提供审查意见和改进建议。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "focus": {
                            "type": "string",
                            "description": "审查重点（可选）",
                            "enum": ["general", "security", "performance", "style"],
                            "default": "general",
                        },
                    },
                },
            ),
            Tool(
                name="project_map",
                description="展示项目目录结构和概览信息。快速了解项目布局。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "detail": {
                            "type": "string",
                            "description": "详细程度: brief(仅目录) / full(含文件大小和未提交变更)",
                            "enum": ["brief", "full"],
                            "default": "brief",
                        },
                    },
                },
            ),
            Tool(
                name="find_issues",
                description="扫描项目中的 TODO、FIXME、BUG、HACK 等标记，以及大文件。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "min_file_size_kb": {
                            "type": "number",
                            "description": "超过此大小(KB)的文件会列入大文件清单",
                            "default": 500,
                        },
                    },
                },
            ),
        ]

    # -- 工具执行 -----------------------------------------------------------

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        project_root = Path(__file__).resolve().parent.parent

        if name == "think":
            question = arguments["question"]
            context = arguments.get("context", "")
            prompt = f"## 问题\n{question}\n"
            if context:
                prompt += f"\n## 上下文\n{context}\n"
            prompt += "\n请给出深入、结构化的分析。"

            system_prompt = (
                "你是小漆，一个能独立思考、主动解决问题的AI伙伴。"
                "你的回答要深入、有见解，不要敷衍。"
                "对于技术问题，分析利弊、给出具体建议。"
                "对于设计问题，提出多个方案并比较。"
            )
            result = await _call_deepseek(prompt, system_prompt)
            return [TextContent(type="text", text=result)]

        elif name == "analyze_code":
            file_path = arguments["file_path"]
            focus = arguments.get("focus", "general")

            content, err = _safe_read_project_file(file_path)
            if err:
                return [TextContent(type="text", text=f"❌ {err}")]

            focus_prompts = {
                "architecture": "重点分析代码架构：模块职责、耦合度、扩展性、设计模式使用。",
                "performance": "重点分析性能：算法复杂度、IO操作、内存使用、可优化的热点。",
                "bugs": "重点找Bug：逻辑错误、边界条件、类型问题、竞态条件。",
                "security": "重点分析安全风险：注入、鉴权、敏感信息泄露、输入校验。",
                "readability": "重点分析可读性：命名、注释、复杂度、代码组织。",
                "general": "全面分析：结构、逻辑、可读性、潜在问题、改进建议。",
            }
            focus_prompt = focus_prompts.get(focus, focus_prompts["general"])

            prompt = (
                f"## 文件\n{file_path}\n"
                f"## {focus} 分析\n{focus_prompt}\n"
                f"## 代码\n```python\n{content}\n```\n"
                "请给出结构化的代码审查意见。"
            )
            system_prompt = (
                "你是一位资深代码审查专家。分析要深入具体，"
                "指出问题的同时给出改进方案和代码示例。"
            )
            result = await _call_deepseek(prompt, system_prompt)
            return [TextContent(type="text", text=result)]

        elif name == "review_changes":
            focus = arguments.get("focus", "general")
            status = _get_git_status()

            if "error" in status:
                return [TextContent(type="text", text=f"❌ Git 错误: {status['error']}")]

            if status["change_count"] == 0:
                return [TextContent(type="text", text="✅ 没有未提交的变更。")]

            # 获取 diff
            try:
                diff_result = subprocess.run(
                    ["git", "-C", str(project_root), "diff"],
                    capture_output=True, text=True, timeout=15,
                )
                diff = diff_result.stdout[:8000]  # 截断避免 token 过多
            except Exception as e:
                diff = f"(获取 diff 失败: {e})"

            prompt = (
                f"## 分支: {status['branch']}\n"
                f"## 变更文件数: {status['change_count']}\n"
                f"## 变更概览\n{status['changes']}\n\n"
                f"## Diff\n```diff\n{diff}\n```\n\n"
                f"请审查这些变更，重点是: {focus}"
            )

            system_prompt = "你是一位严谨的代码审查专家，审查意见要具体、有建设性。"
            result = await _call_deepseek(prompt, system_prompt)
            return [TextContent(type="text", text=result)]

        elif name == "project_map":
            detail = arguments.get("detail", "brief")
            tree = _get_project_tree(max_depth=4 if detail == "full" else 3)
            output = f"## 📁 项目结构\n\n```\n{tree}\n```\n"

            if detail == "full":
                status = _get_git_status()
                output += f"\n## 🔀 Git 状态\n\n- 分支: {status['branch']}\n"
                output += f"- 未提交变更: {status['change_count']} 个文件\n"

                issues = _find_issues()
                if issues:
                    output += "\n## ⚠️ 代码标记\n\n"
                    for issue in issues:
                        output += f"- **{issue['type']}** ({issue['count']} 处): "
                        output += ", ".join(issue["files"][:5])
                        if issue["count"] > 5:
                            output += f" 等{issue['count']}个文件"
                        output += "\n"

            return [TextContent(type="text", text=output)]

        elif name == "find_issues":
            min_size = arguments.get("min_file_size_kb", 500)

            # 代码标记
            markers = _find_issues()
            large_files = _find_large_files(min_kb=min_size)

            output = "## 🔍 项目扫描结果\n\n"

            if markers:
                output += "### 📌 代码标记\n\n"
                for m in markers:
                    output += f"- **{m['type']}** ({m['pattern']}): "
                    output += f"{m['count']} 处出现在 "
                    output += ", ".join(m["files"][:5])
                    output += "\n"
            else:
                output += "### ✅ 代码标记\n\n干净！没有待办或修复标记。\n\n"

            if large_files:
                output += "\n### 📦 大文件 (>{}KB)\n\n".format(min_size)
                for f in large_files:
                    output += f"- {f['file']}: {f['size_kb']}KB\n"
            else:
                output += "\n### ✅ 大文件\n\n没有超过 {}KB 的文件。\n".format(min_size)

            return [TextContent(type="text", text=output)]

        else:
            raise ValueError(f"未知工具: {name}")

    # -- 启动 ---------------------------------------------------------------

    async def _run():
        with _QuietStdout():
            async with stdio_server() as (read, write):
                await server.run(
                    read, write,
                    server.create_initialization_options(),
                )

    asyncio.run(_run())


if __name__ == "__main__":
    main_stdio()
