# RAG 评估指南

## 📋 概述

本项目提供 **多种评估方案**，从简单到高级：

1. **方案 A：手动实现（推荐）** - `eval_v2.py` / `run_ragas_eval.py`
   - ✅ 零依赖，零冲突
   - ✅ 独立 LLM 评估，避免自评偏差
   - ✅ 幻觉检测（反例题支持）
   - ✅ 详细推理过程输出

2. **方案 B：RAGAS 官方库（可选）** - `ragas_integration.py`
   - 使用官方库，但需要解决依赖冲突

---

## 🎯 快速开始（推荐方案 A）

### 直接运行增强版评估

```bash
cd backend
python eval/eval_v2.py
```

这个脚本**不需要安装任何额外依赖**，直接用项目已有的 LLM 做评估！

---

## 🧪 如果需要使用 RAGAS 官方库

### 方案 1：安全隔离安装（推荐）

使用 `install_ragas_safe.py` 把 RAGAS 安装到独立目录，不影响主项目：

```bash
cd backend/eval

# 1. 安装 RAGAS 到隔离目录
python install_ragas_safe.py --install

# 2. 运行 RAGAS 评估
python install_ragas_safe.py --run ragas_integration.py
```

### 方案 2：虚拟环境隔离

```bash
cd backend/eval

# 创建虚拟环境
python -m venv eval_env

# 激活（Windows）
eval_env\Scripts\activate
# 激活（Linux/Mac）
source eval_env/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行评估
python ragas_integration.py

# 退出环境
deactivate
```

---

## 📊 评估指标

### 四个核心指标

| 指标 | 范围 | 说明 |
|------|------|------|
| **Faithfulness（忠实度）** | 0.0 - 1.0 | 回答是否基于检索文档，无幻觉 |
| **Answer Relevancy（回答相关性）** | 0.0 - 1.0 | 回答是否解决用户问题 |
| **Context Precision（上下文精确率）** | 0.0 - 1.0 | 检索到的文档是否相关 |
| **Context Recall（上下文召回率）** | 0.0 - 1.0 | 是否检索到所有相关文档 |

### 项目独有指标（eval_v2.py）

- **Hallucination Resistance（抗幻觉能力）**：测试系统对无答案问题的拒答能力

---

## 📝 测试数据集

### 数据格式

```json
[
  {
    "question": "用户问题",
    "ground_truth": "标准答案",
    "negative": false  // 可选，标记为反例题（无答案问题）
  }
]
```

### 已有数据集

- `test_dataset.json` - 基础测试集（10条）
- `test_dataset_v2.json` - 增强版（含反例题）

---

## 🔍 评估结果解读

| 综合得分 | 评价 |
|---------|------|
| >= 0.8 | 🎉 优秀 |
| 0.6 - 0.8 | ✅ 良好 |
| 0.4 - 0.6 | ⚠️ 需优化 |
| < 0.4 | ❌ 需大幅改进 |

---

## 💡 优化建议

### 如果忠实度低
- 检查 LLM prompt，强调"基于提供的文档回答"
- 降低 temperature 参数
- 使用更强的指令遵循模型

### 如果答案相关性低
- 优化 LLM prompt，明确要求回答用户问题
- 检查检索结果是否相关

### 如果上下文精确率低
- 优化检索策略（向量检索、BM25、混合检索）
- 调整 top_k 参数
- 使用 Reranker 重排序

### 如果上下文召回率低
- 增加 top_k 参数
- 优化 embedding 模型
- 使用查询扩展技术

---

## ❓ 常见问题

### Q: 如何避免 LangChain 依赖冲突？

A: 使用 `install_ragas_safe.py` 隔离安装，或直接用 `eval_v2.py`（推荐）。

### Q: 评估过程中某个查询失败？

A: 脚本会跳过失败查询，继续评估其他问题。查看日志了解失败原因。

### Q: 评估结果不稳定？

A: LLM 评估有随机性，可以多次运行取平均值。

### Q: 如何选择评估模型？

A: `eval_v2.py` 支持通过环境变量 `EVAL_PROVIDER` 选择：
- `deepseek`（默认）
- `openai`
- `mimo`

---

## 📚 参考资料

- [RAGAS 官方文档](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- [RAG 评估最佳实践](https://docs.ragas.io/en/latest/concepts/evaluation/)


