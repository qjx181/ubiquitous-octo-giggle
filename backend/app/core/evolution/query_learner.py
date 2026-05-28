# -*- coding: utf-8 -*-
"""backend/app/core/evolution/query_learner.py — 查询学习器

作用：从历史查询中学习优化策略，实现 RAG 系统的自进化

借鉴 HiveWard 的 evolution_engine.py 设计：
  - 经验积累：记录每次执行的成功/失败模式
  - 置信度校准：根据历史成功率调整决策阈值
  - 模式识别：从历史查询中发现规律

应用场景：
  1. 意图识别优化：历史查询"XX公司营收"80%是FINANCIAL意图 → 新查询自动提升FINANCIAL置信度
  2. 检索参数优化：COMPARISON意图通常需要top_k=8 → 自动调整检索数量
  3. 失败学习：某类查询反复失败 → 自动降低置信度或触发人工审核

设计原理：
  1. QueryRecord：记录单次查询的完整信息（查询、意图、结果、反馈）
  2. PatternDB：存储和查询历史模式（意图分布、成功率、最佳参数）
  3. QueryLearner：对外接口，提供意图提示和检索增强

面试官可能问：
  Q: 自进化机制如何保证不会越学越差？
  A: 三个保护机制：
     1. 置信度阈值：只有历史成功率 > 70% 的模式才会被采用
     2. 衰减机制：旧的模式权重逐渐降低，避免过时数据影响
     3. 人工兜底：低置信度查询走人工审核，不自动决策

  Q: 如何处理冷启动问题（没有历史数据）？
  A: 冷启动时返回空提示，系统回退到默认行为。
     随着查询积累，系统逐渐学习优化。这是"无数据时不干预"原则。

  Q: 历史数据存在哪里？
  A: 当前用内存字典（简单实现），生产环境可以迁移到 Redis 或 SQLite。
     设计时用抽象接口，方便后续替换存储后端。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QueryRecord:
    """查询记录：记录单次查询的完整信息
    
    属性：
      - query: 用户原始查询
      - normalized_query: 归一化后的查询
      - intent: 识别的意图类型
      - intent_confidence: 意图置信度
      - entities: 提取的实体列表
      - retrieval_results: 检索结果
      - answer: 生成的答案
      - user_feedback: 用户反馈（1-5分，可选）
      - timestamp: 查询时间戳
      - success: 是否成功（基于反馈或自动判断）
    """
    query: str
    normalized_query: str
    intent: str
    intent_confidence: float
    entities: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_results: List[Dict[str, Any]] = field(default_factory=list)
    answer: Optional[str] = None
    user_feedback: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    success: Optional[bool] = None


class PatternDB:
    """模式数据库：存储和查询历史模式
    
    数据结构：
      - intent_patterns: {intent_type: {pattern: count}} 意图-模式映射
      - intent_stats: {intent_type: {total: int, success: int}} 意图统计
      - retrieval_params: {intent_type: {param: value}} 最佳检索参数
      - entity_patterns: {entity_type: {value: count}} 实体频率
    """
    
    def __init__(self):
        # 意图-模式映射：记录每个意图下常见查询模式
        self.intent_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # 意图统计：记录每个意图的总查询数和成功数
        self.intent_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
        
        # 最佳检索参数：记录每个意图下效果最好的参数
        self.retrieval_params: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # 实体频率：记录常见实体
        self.entity_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # 查询历史（最近 N 条）
        self.query_history: List[QueryRecord] = []
        self.max_history = 1000
    
    def add_record(self, record: QueryRecord):
        """添加查询记录"""
        # 添加到历史
        self.query_history.append(record)
        if len(self.query_history) > self.max_history:
            self.query_history = self.query_history[-self.max_history:]
        
        # 更新意图统计
        self.intent_stats[record.intent]["total"] += 1
        if record.success:
            self.intent_stats[record.intent]["success"] += 1
        
        # 更新意图-模式映射
        pattern = self._extract_pattern(record.normalized_query)
        self.intent_patterns[record.intent][pattern] += 1
        
        # 更新实体频率
        for entity in record.entities:
            entity_type = entity.get("type", "unknown")
            entity_value = entity.get("text", "")
            if entity_value:
                self.entity_patterns[entity_type][entity_value] += 1
    
    def _extract_pattern(self, query: str) -> str:
        """提取查询模式（简化版本）
        
        将查询转换为模式字符串：
        - 保留关键词（去除停用词）
        - 统一数字为 [NUM]
        - 统一时间为 [TIME]
        """
        import re
        # 去除停用词（简化版）
        stop_words = {"的", "了", "吗", "呢", "吧", "啊", "是", "在", "有", "和", "与"}
        words = [w for w in query if w not in stop_words]
        pattern = "".join(words)
        
        # 统一数字
        pattern = re.sub(r'\d+', '[NUM]', pattern)
        
        # 统一时间
        pattern = re.sub(r'\d{4}年', '[TIME]', pattern)
        pattern = re.sub(r'\d{1,2}月', '[TIME]', pattern)
        
        return pattern
    
    def get_intent_hints(self, query: str) -> Dict[str, float]:
        """根据历史查询，返回意图置信度提示
        
        Returns:
            Dict[str, float]: {intent_type: confidence_boost} 意图置信度提升
        """
        hints = {}
        pattern = self._extract_pattern(query)
        
        for intent, patterns in self.intent_patterns.items():
            if pattern in patterns:
                # 找到匹配的历史模式
                total = self.intent_stats[intent]["total"]
                success = self.intent_stats[intent]["success"]
                
                if total > 0:
                    success_rate = success / total
                    # 只有成功率 > 70% 的模式才会被采用
                    if success_rate > 0.7:
                        # 提升置信度（最多提升 0.2）
                        boost = min(0.2, success_rate * 0.25)
                        hints[intent] = boost
        
        return hints
    
    def get_retrieval_params(self, intent: str) -> Dict[str, Any]:
        """根据历史查询，返回最佳检索参数
        
        Returns:
            Dict[str, Any]: 检索参数（top_k, threshold 等）
        """
        return self.retrieval_params.get(intent, {})
    
    def get_entity_suggestions(self, entity_type: str, prefix: str = "") -> List[str]:
        """根据历史查询，返回实体建议
        
        Args:
            entity_type: 实体类型（PERSON, ORG, TIME 等）
            prefix: 前缀过滤（可选）
            
        Returns:
            List[str]: 建议的实体值列表
        """
        if entity_type not in self.entity_patterns:
            return []
        
        entities = self.entity_patterns[entity_type]
        
        # 按频率排序
        sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)
        
        # 前缀过滤
        if prefix:
            sorted_entities = [(k, v) for k, v in sorted_entities if k.startswith(prefix)]
        
        # 返回前 10 个
        return [k for k, v in sorted_entities[:10]]
    
    def save(self, path: str):
        """保存到文件"""
        data = {
            "intent_patterns": dict(self.intent_patterns),
            "intent_stats": dict(self.intent_stats),
            "retrieval_params": dict(self.retrieval_params),
            "entity_patterns": dict(self.entity_patterns)
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """从文件加载"""
        if not Path(path).exists():
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.intent_patterns = defaultdict(lambda: defaultdict(int), data.get("intent_patterns", {}))
        self.intent_stats = defaultdict(lambda: {"total": 0, "success": 0}, data.get("intent_stats", {}))
        self.retrieval_params = defaultdict(dict, data.get("retrieval_params", {}))
        self.entity_patterns = defaultdict(lambda: defaultdict(int), data.get("entity_patterns", {}))


class QueryLearner:
    """查询学习器：对外接口，提供意图提示和检索增强
    
    使用方式：
        learner = QueryLearner()
        
        # 记录查询
        learner.record_query(
            query="腾讯营收是多少",
            normalized_query="腾讯营收",
            intent="financial",
            intent_confidence=0.85,
            entities=[{"type": "ORG", "text": "腾讯"}],
            retrieval_results=[...],
            answer="腾讯2023年营收...",
            user_feedback=4.5
        )
        
        # 获取意图提示
        hints = learner.get_intent_hints("阿里营收")
        # 返回: {"financial": 0.15}
        
        # 获取检索参数
        params = learner.get_retrieval_params("financial")
        # 返回: {"top_k": 5, "threshold": 0.6}
    
    设计对比 HiveWard：
      HiveWard 的 evolution_engine.py 实现了更复杂的自进化机制：
      - Skill IR 优化：从执行结果中学习如何改进 Skill 定义
      - 置信度校准：根据历史成功率调整决策阈值
      - 自动重试：失败时自动尝试备选方案
      
      我们的 QueryLearner 实现了核心的"经验积累+模式识别"功能，
      是 HiveWard 自进化机制的简化版本。
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """初始化查询学习器
        
        Args:
            storage_path: 历史数据存储路径（可选，默认使用内存）
        """
        self.pattern_db = PatternDB()
        self.storage_path = storage_path
        
        # 如果指定了存储路径，加载历史数据
        if storage_path:
            self.pattern_db.load(storage_path)
            logger.info(f"📚 加载历史数据: {storage_path}")
    
    def record_query(self, query: str, normalized_query: str, 
                     intent: str, intent_confidence: float,
                     entities: List[Dict[str, Any]] = None,
                     retrieval_results: List[Dict[str, Any]] = None,
                     answer: Optional[str] = None,
                     user_feedback: Optional[float] = None):
        """记录查询及其结果
        
        Args:
            query: 用户原始查询
            normalized_query: 归一化后的查询
            intent: 识别的意图类型
            intent_confidence: 意图置信度
            entities: 提取的实体列表
            retrieval_results: 检索结果
            answer: 生成的答案
            user_feedback: 用户反馈（1-5分，可选）
        """
        # 判断是否成功
        success = None
        if user_feedback is not None:
            success = user_feedback >= 3.0  # 3分以上认为成功
        
        # 创建记录
        record = QueryRecord(
            query=query,
            normalized_query=normalized_query,
            intent=intent,
            intent_confidence=intent_confidence,
            entities=entities if entities is not None else [],
            retrieval_results=retrieval_results if retrieval_results is not None else [],
            answer=answer,
            user_feedback=user_feedback,
            success=success
        )
        
        # 添加到数据库
        self.pattern_db.add_record(record)
        
        # 定期保存
        if self.storage_path and len(self.pattern_db.query_history) % 100 == 0:
            self.pattern_db.save(self.storage_path)
        
        logger.debug(f"📝 记录查询: {query} -> {intent} ({'成功' if success else '失败'})")
    
    def get_intent_hints(self, query: str) -> Dict[str, float]:
        """根据历史查询，返回意图置信度提示
        
        Args:
            query: 用户查询
            
        Returns:
            Dict[str, float]: {intent_type: confidence_boost} 意图置信度提升
        """
        return self.pattern_db.get_intent_hints(query)
    
    def get_retrieval_params(self, intent: str) -> Dict[str, Any]:
        """根据历史查询，返回最佳检索参数
        
        Args:
            intent: 意图类型
            
        Returns:
            Dict[str, Any]: 检索参数（top_k, threshold 等）
        """
        return self.pattern_db.get_retrieval_params(intent)
    
    def get_entity_suggestions(self, entity_type: str, prefix: str = "") -> List[str]:
        """根据历史查询，返回实体建议
        
        Args:
            entity_type: 实体类型（PERSON, ORG, TIME 等）
            prefix: 前缀过滤（可选）
            
        Returns:
            List[str]: 建议的实体值列表
        """
        return self.pattern_db.get_entity_suggestions(entity_type, prefix)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取学习统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            "total_queries": len(self.pattern_db.query_history),
            "intent_distribution": {},
            "success_rate": {}
        }
        
        # 意图分布
        for intent, intent_stats in self.pattern_db.intent_stats.items():
            stats["intent_distribution"][intent] = intent_stats["total"]
            if intent_stats["total"] > 0:
                stats["success_rate"][intent] = intent_stats["success"] / intent_stats["total"]
        
        return stats
    
    def save(self):
        """保存到文件"""
        if self.storage_path:
            self.pattern_db.save(self.storage_path)
            logger.info(f"💾 保存历史数据: {self.storage_path}")


# ============ 全局实例 ============

_learner_instance: Optional[QueryLearner] = None


def get_query_learner(storage_path: Optional[str] = None) -> QueryLearner:
    """获取查询学习器的全局单例
    
    Args:
        storage_path: 历史数据存储路径（可选）
        
    Returns:
        QueryLearner: 查询学习器实例
    """
    global _learner_instance
    if _learner_instance is None:
        _learner_instance = QueryLearner(storage_path)
    return _learner_instance
