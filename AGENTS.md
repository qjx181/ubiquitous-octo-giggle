# 项目二(github) - RAG在线服务 - Hermes Agent 配置

项目二的GitHub镜像版本，增加企业级部署配置

## 可用 Agent 角色

当在此项目下工作时，以下 Hermes Skills 可用：

- `agency-ai-engineer`
- `agency-backend-architect`
- `agency-code-reviewer`
- `agency-technical-writer`
- `agency-evidence-collector`

## 使用方式

在 Hermes 对话中，调用 `skill_view("agent-name")` 加载对应角色。
也可以直接在提问时提及角色名称，例如：
- "用 Code Reviewer 模式审查这段代码"
- "以 AI Engineer 的视角分析这个架构"
- "启动 Evidence Collector 模式，找出所有问题"

## 自动化任务

以下 cron 任务已为此项目配置：
- 每周代码审查（用 Code Reviewer 角色）
- 每周架构审查（用 Backend Architect 角色）