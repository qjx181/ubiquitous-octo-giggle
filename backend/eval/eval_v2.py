"""
RAG 评估脚本 V2 — 独立 LLM 裁判 + 严格 Prompt + 反例检测
============================================================
改进点：
  1. 评估用独立 LLM（DeepSeek），不和 RAG 系统共用同一模型
  2. Faithfulness：拆声明逐条核查，输出推理过程
  3. Context Precision：不截断上下文，按片段逐一判断
  4. 新增 Hallucination Rate：反例题检测（应当拒答但系统回答了）
  5. 每题输出 LLM 的推理过程，便于人工复核
  6. 支持通过环境变量 EVAL_PROVIDER 选择评估模型
"""

import os
import sys
import json
import asyncio
import re
import logging
import argparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# 加载 .env 文件（RAG 系统和评估 LLM 共用）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag_eval_v2")


# ==================== 独立评估 LLM 客户端 ====================

class EvalLLMClient:
    """独立的评估用 LLM 客户端，不走 LLMRouter，避免自评偏差"""

    def __init__(self):
        import httpx
        self._httpx = httpx

        # 评估模型优先级：环境变量 EVAL_PROVIDER > DeepSeek > MiMo
        provider = os.environ.get("EVAL_PROVIDER", "deepseek").lower()

        if provider == "deepseek" or provider == "":
            self.host = os.environ.get("DEEPSEEK_HOST", "https://api.deepseek.com/v1")
            self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
            self.api_key = os.environ.get("DEEPSEEK_KEY", "")
            self.chat_url = f"{self.host}/chat/completions"
        elif provider == "mimo":
            self.host = os.environ.get("MIMO_HOST", "https://api.xiaomimimo.com/v1")
            self.model = os.environ.get("MIMO_MODEL", "Mimo-v2.5-pro")
            self.api_key = os.environ.get("MIMO_KEY", "")
            self.chat_url = f"{self.host}/chat/completions"
        elif provider == "openai":
            self.host = os.environ.get("OPENAI_HOST", "https://api.openai.com/v1")
            self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            self.api_key = os.environ.get("OPENAI_KEY", "")
            self.chat_url = f"{self.host}/chat/completions"
        else:
            raise ValueError(f"不支持的评估模型提供商: {provider}")

        self.provider = provider
        logger.info(f"评估 LLM 初始化: provider={provider}, model={self.model}")

    async def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """调用评估 LLM，temperature 设低保证打分稳定"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(3):
            try:
                async with self._httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(self.chat_url, json=payload, headers=headers)
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"评估 LLM 调用失败 (尝试 {attempt+1}/3): {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"评估 LLM 调用失败: {e}")
        return ""


# ==================== 评估 Prompt（严格版） ====================

FAITHFULNESS_PROMPT = """你是一个严格的事实核查员。你的任务是判断"回答"中的每一个事实性声明是否有"上下文"支撑。

【上下文】
{context}

【回答】
{answer}

【评估步骤】
1. 把回答拆成独立的事实性声明（忽略语气词和过渡句）
2. 逐条检查每个声明是否有上下文直接支撑
3. 如果声明在上下文中找不到依据，标记为"无支撑"
4. 如果声明与上下文矛盾，标记为"矛盾"

请按以下 JSON 格式输出（不要输出其他内容）：
{{
  "claims": [
    {{"claim": "声明1", "supported": true, "evidence": "支撑该声明的上下文原文片段"}},
    {{"claim": "声明2", "supported": false, "evidence": "无"}},
    ...
  ],
  "score": 0.0到1.0之间的分数（有支撑的声明数 / 总声明数）,
  "reasoning": "一句话总结评估理由"
}}

只输出 JSON，不要输出其他内容。"""


ANSWER_RELEVANCY_PROMPT = """你是一个严格的评估员。请判断"回答"是否真正回答了"问题"。

【问题】
{question}

【回答】
{answer}

【评估标准】
- 1.0：回答直接、完整地回答了问题
- 0.7-0.9：回答部分回答了问题，有遗漏但核心信息正确
- 0.4-0.6：回答提到了相关话题，但没有直接回答问题
- 0.1-0.3：回答与问题关系不大
- 0.0：回答与问题完全无关

特别注意：
- 如果问题问"主要业务是什么"，回答只提到了子公司信息，分数应 <= 0.3
- 如果回答说"无法从提供的信息中回答"，但其实上下文中有相关信息，分数应 <= 0.2
- 如果回答信息不完整但方向正确，给 0.5-0.7

请按以下 JSON 格式输出：
{{
  "score": 0.0到1.0之间的分数,
  "reasoning": "一句话说明评分理由"
}}

只输出 JSON，不要输出其他内容。"""


CONTEXT_PRECISION_PROMPT = """你是一个严格的检索评估员。请逐条判断检索到的上下文片段是否与问题相关。

【问题】
{question}

【检索到的上下文片段（按相关性排序）】
{context_list}

【评估标准】
- 对每个片段判断：是否包含回答该问题所需的关键信息
- 如果片段只包含问题中提到的关键词但信息不对（比如问A公司的业务，检索到了B公司的业务），算"不相关"
- 如果片段包含部分相关信息，算"相关"

请按以下 JSON 格式输出：
{{
  "per_fragment": [
    {{"index": 1, "relevant": true, "reason": "包含公司主营业务描述"}},
    {{"index": 2, "relevant": false, "reason": "是子公司信息，不是主营业务"}},
    ...
  ],
  "score": 0.0到1.0之间的分数（相关片段数 / 总片段数）,
  "reasoning": "一句话总结"
}}

只输出 JSON，不要输出其他内容。"""


CONTEXT_RECALL_PROMPT = """你是一个严格的检索评估员。请判断"标准答案"中的关键信息是否被"检索上下文"覆盖。

【标准答案】
{ground_truth}

【检索到的上下文】
{context}

【评估步骤】
1. 从标准答案中提取关键信息点（事实性内容，忽略表述方式）
2. 逐条检查每个关键信息点是否在上下文中有对应内容
3. 不要求措辞完全一致，语义相同即可

请按以下 JSON 格式输出：
{{
  "key_points": [
    {{"point": "关键信息点1", "covered": true, "evidence": "上下文中的对应内容"}},
    {{"point": "关键信息点2", "covered": false, "evidence": "无"}},
    ...
  ],
  "score": 0.0到1.0之间的分数（被覆盖的关键信息点数 / 总关键信息点数）,
  "reasoning": "一句话总结"
}}

只输出 JSON，不要输出其他内容。"""


HALLUCINATION_PROMPT = """你是一个严格的评估员。以下问题是一个"陷阱题"——正确答案应该是"无法确定"或"拒绝回答"（因为上下文中没有相关信息）。

请判断 RAG 系统是否正确地拒绝了回答，还是产生了幻觉（编造了不存在的信息）。

【问题】
{question}

【RAG 系统的回答】
{answer}

【检索到的上下文】
{context}

【评估标准】
- 1.0：系统明确表示无法从给定信息中回答（正确拒绝）
- 0.5：系统说"可能"或"不确定"，但还是给了一些信息（部分幻觉）
- 0.0：系统自信地给出了答案，但该答案在上下文中没有依据（完全幻觉）

请按以下 JSON 格式输出：
{{
  "refused": true/false,
  "score": 0.0到1.0之间的分数,
  "reasoning": "一句话说明"
}}

只输出 JSON，不要输出其他内容。"""


# ==================== 分数提取 ====================

def extract_json_from_llm(raw: str) -> dict:
    """从 LLM 输出中提取 JSON，容忍 markdown 代码块等格式"""
    # 尝试直接解析
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 去掉 markdown 代码块
    for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

    # 最后尝试从第一个 { 到最后一个 } 提取
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end+1])
        except json.JSONDecodeError:
            pass

    logger.warning(f"无法从 LLM 输出中提取 JSON: {raw[:200]}")
    return {}


def extract_score(raw: str) -> float:
    """从 LLM 输出中提取 0-1 分数，兼容纯数字和 JSON 格式"""
    parsed = extract_json_from_llm(raw)
    if parsed and "score" in parsed:
        return max(0.0, min(1.0, float(parsed["score"])))

    # 兜底：正则提取数字
    match = re.search(r'(0?\.\d+|1\.0|1|0)', raw)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))

    logger.warning(f"无法解析评分: {raw[:200]}")
    return 0.0


# ==================== RAG 结果收集 ====================

async def collect_rag_results(dataset: List[dict]) -> List[dict]:
    """遍历测试数据集，收集 RAG 系统的实际回答和检索结果"""
    from app.core.retrieval import get_retrieval_service
    service = get_retrieval_service()

    results = []
    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        is_negative = item.get("negative", False)
        logger.info(f"[{i+1}/{len(dataset)}] {'[反例] ' if is_negative else ''}查询: {question}")

        try:
            result = await service.query(question=question, top_k=5)
            answer = result.get("answer", "")
            contexts = [r.get("text", "") for r in result.get("retrieval_results", [])]
            if not contexts:
                contexts = [s.get("snippet", s.get("text", "")) for s in result.get("sources", [])]
        except Exception as e:
            logger.warning(f"查询失败: {e}")
            answer = ""
            contexts = []

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "negative": is_negative,
        })

    return results


# ==================== 评估主流程 ====================

async def evaluate_single(eval_client: EvalLLMClient, r: dict) -> dict:
    """评估单条结果，返回四维分数 + 推理过程"""
    q = r["question"]
    a = r["answer"]
    ctx = r["contexts"]
    gt = r["ground_truth"]
    is_negative = r["negative"]
    ctx_str = "\n\n".join([f"[片段{j+1}] {c}" for j, c in enumerate(ctx)]) if ctx else "无相关文档"

    if not a:
        return {
            "faithfulness": 0.0, "answer_relevancy": 0.0,
            "context_precision": 0.0, "context_recall": 0.0,
            "hallucination_score": 1.0 if is_negative else None,
            "reasoning": {"faithfulness": "无回答", "answer_relevancy": "无回答",
                          "context_precision": "无回答", "context_recall": "无回答"},
        }

    # 反例题：走专门的幻觉检测
    if is_negative:
        hall_raw = await eval_client.generate(
            HALLUCINATION_PROMPT.format(question=q, answer=a, context=ctx_str)
        )
        hall_parsed = extract_json_from_llm(hall_raw)
        hall_score = hall_parsed.get("score", extract_score(hall_raw))

        # 对反例题，faithfulness 和 answer_relevancy 用特殊逻辑
        # 如果系统正确拒答，F=1.0, AR=1.0；否则 F=0, AR=0
        refused = hall_parsed.get("refused", hall_score > 0.7)

        return {
            "faithfulness": 1.0 if refused else 0.0,
            "answer_relevancy": 1.0 if refused else 0.0,
            "context_precision": 0.0,  # 反例题不应检索到相关文档
            "context_recall": 1.0 if refused else 0.0,
            "hallucination_score": hall_score,
            "reasoning": {
                "hallucination": hall_parsed.get("reasoning", ""),
                "faithfulness": "正确拒答" if refused else "产生了幻觉",
                "answer_relevancy": "正确拒答" if refused else "不应该回答",
                "context_precision": "反例题",
                "context_recall": "反例题",
            },
        }

    # 正常题：四个指标并行评估
    ctx_list = "\n\n".join([f"[片段{j+1}] {c}" for j, c in enumerate(ctx)]) if ctx else "无"

    prompts = {
        "faithfulness": FAITHFULNESS_PROMPT.format(context=ctx_str, answer=a),
        "answer_relevancy": ANSWER_RELEVANCY_PROMPT.format(question=q, answer=a),
        "context_precision": CONTEXT_PRECISION_PROMPT.format(question=q, context_list=ctx_list),
        "context_recall": CONTEXT_RECALL_PROMPT.format(ground_truth=gt, context=ctx_str) if gt else None,
    }

    # 并行调用评估 LLM
    tasks = {}
    for metric, prompt in prompts.items():
        if prompt is not None:
            tasks[metric] = asyncio.create_task(eval_client.generate(prompt))

    raw_results = {}
    for metric, task in tasks.items():
        raw_results[metric] = await task

    # 解析分数和推理过程
    scores = {}
    reasoning = {}
    for metric, raw in raw_results.items():
        parsed = extract_json_from_llm(raw)
        scores[metric] = parsed.get("score", extract_score(raw))
        reasoning[metric] = parsed.get("reasoning", "")

    # context_recall 如果没有 ground_truth 就跳过
    if "context_recall" not in scores:
        scores["context_recall"] = 0.0
        reasoning["context_recall"] = "无标准答案"

    return {
        "faithfulness": scores.get("faithfulness", 0.0),
        "answer_relevancy": scores.get("answer_relevancy", 0.0),
        "context_precision": scores.get("context_precision", 0.0),
        "context_recall": scores.get("context_recall", 0.0),
        "hallucination_score": None,
        "reasoning": reasoning,
    }


async def run_evaluation(results: List[dict]) -> List[dict]:
    """对所有结果逐条评估"""
    eval_client = EvalLLMClient()

    eval_results = []
    for i, r in enumerate(results):
        logger.info(f"[{i+1}/{len(results)}] 评估: {r['question'][:50]}...")
        try:
            eval_result = await evaluate_single(eval_client, r)
        except Exception as e:
            logger.error(f"评估失败: {e}")
            eval_result = {
                "faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0,
                "hallucination_score": None, "reasoning": {"error": str(e)},
            }

        eval_results.append(eval_result)
        f = eval_result["faithfulness"]
        ar = eval_result["answer_relevancy"]
        cp = eval_result["context_precision"]
        cr = eval_result["context_recall"]
        hall = eval_result.get("hallucination_score")
        hall_str = f"  hallucination={hall:.2f}" if hall is not None else ""
        logger.info(f"  F={f:.2f}  AR={ar:.2f}  CP={cp:.2f}  CR={cr:.2f}{hall_str}")

    return eval_results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="RAG 评估脚本 V2")
    parser.add_argument("--group", choices=["train", "eval", "all"], default="all",
                        help="只评估指定组: train(前30题), eval(后27题+反例), all(全部)")
    args = parser.parse_args()

    eval_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(eval_dir, "test_dataset_v2.json")
    output_path = os.path.join(eval_dir, "eval_results_v2.json")

    # 加载测试数据集
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 按组过滤
    if args.group == "train":
        dataset = [d for d in dataset if d.get("group") == "train"]
        output_path = os.path.join(eval_dir, "eval_results_train.json")
    elif args.group == "eval":
        dataset = [d for d in dataset if d.get("group") == "eval"]
        output_path = os.path.join(eval_dir, "eval_results_eval.json")

    logger.info(f"测试数据集: {dataset_path} (过滤: {args.group}, {len(dataset)} 条)")
    logger.info("=" * 60)

    # 第一步：收集 RAG 结果
    logger.info("第一步：收集 RAG 系统的回答和检索结果")
    rag_results = asyncio.run(collect_rag_results(dataset))
    logger.info(f"共收集 {len(rag_results)} 条结果")

    # 第二步：用独立 LLM 评估
    logger.info("=" * 60)
    logger.info("第二步：用独立 LLM 评估每个指标")
    eval_results = asyncio.run(run_evaluation(rag_results))

    # 第三步：汇总
    logger.info("=" * 60)
    logger.info("评估结果汇总:")

    # 分开统计正常题和反例题
    normal_indices = [i for i, r in enumerate(dataset) if not r.get("negative")]
    negative_indices = [i for i, r in enumerate(dataset) if r.get("negative")]

    def avg_metric(results_list, indices, metric):
        vals = [results_list[i][metric] for i in indices if i < len(results_list)]
        return sum(vals) / len(vals) if vals else 0.0

    avg_scores = {}
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        avg_scores[metric] = round(avg_metric(eval_results, normal_indices, metric), 4)
        logger.info(f"  正常题 {metric}: {avg_scores[metric]:.4f}")

    if negative_indices:
        hall_vals = [eval_results[i]["hallucination_score"] for i in negative_indices
                     if i < len(eval_results) and eval_results[i].get("hallucination_score") is not None]
        if hall_vals:
            avg_scores["hallucination_resistance"] = round(sum(hall_vals) / len(hall_vals), 4)
            logger.info(f"  反例题 hallucination_resistance: {avg_scores['hallucination_resistance']:.4f}")

    # 保存详细结果
    output = {
        "scores": avg_scores,
        "evaluator_model": os.environ.get("EVAL_PROVIDER", "deepseek"),
        "details": [
            {
                "question": rag_results[i]["question"],
                "answer": rag_results[i]["answer"][:500],
                "negative": rag_results[i].get("negative", False),
                "contexts_count": len(rag_results[i]["contexts"]),
                "faithfulness": eval_results[i]["faithfulness"],
                "answer_relevancy": eval_results[i]["answer_relevancy"],
                "context_precision": eval_results[i]["context_precision"],
                "context_recall": eval_results[i]["context_recall"],
                "hallucination_score": eval_results[i].get("hallucination_score"),
                "reasoning": eval_results[i].get("reasoning", {}),
            }
            for i in range(len(rag_results))
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"详细结果已保存到: {output_path}")

    # 打印摘要表格
    print("\n" + "=" * 70)
    print(f"{'题目':<20} {'F':>6} {'AR':>6} {'CP':>6} {'CR':>6}")
    print("-" * 70)
    for i, r in enumerate(dataset):
        q = r["question"][:18]
        e = eval_results[i]
        neg = " [NEG]" if r.get("negative") else ""
        print(f"{q:<18}{neg} {e['faithfulness']:>5.2f} {e['answer_relevancy']:>5.2f} "
              f"{e['context_precision']:>5.2f} {e['context_recall']:>5.2f}")
    print("-" * 70)
    print(f"{'正常题均值':<20} {avg_scores['faithfulness']:>5.2f} {avg_scores['answer_relevancy']:>5.2f} "
          f"{avg_scores['context_precision']:>5.2f} {avg_scores['context_recall']:>5.2f}")
    if "hallucination_resistance" in avg_scores:
        print(f"{'反例题抗幻觉':<20} {avg_scores['hallucination_resistance']:>5.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
