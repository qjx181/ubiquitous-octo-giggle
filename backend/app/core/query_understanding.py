# -*- coding: utf-8 -*-
"""
Query理解模块
功能：意图识别、实体提取、问题分解、消歧处理

模块职责：
1. 意图识别 - 识别用户问题是关于财务、风险、业务等哪个方面
2. 实体提取 - 从问题中提取时间、金钱、人名等实体
3. 问题分解 - 将复杂问题分解为多个简单子问题
4. 消歧处理 - 判断是否需要向用户澄清问题
"""

# 导入正则表达式模块，用于模式匹配
import re
# 导入类型注解，用于类型提示
from typing import Dict, List, Optional, Tuple
# 导入数据类装饰器，用于创建数据结构
from dataclasses import dataclass
# 导入枚举类，用于定义常量
from enum import Enum
# 导入jieba分词库，用于中文分词和词性标注
import jieba
import jieba.posseg as pseg


class IntentType(Enum):
    """
    意图类型枚举
    
    定义系统能识别的用户意图类型：
    - FINANCIAL: 财务数据类问题（如：收入、利润、资产等）
    - RISK: 风险因素类问题（如：经营风险、市场风险等）
    - BUSINESS: 业务介绍类问题（如：主营业务、产品服务等）
    - LEGAL: 法律合规类问题（如：诉讼、处罚等）
    - MANAGEMENT: 管理层信息类问题（如：董事长、总经理等）
    - COMPARISON: 对比分析类问题（如：对比两家公司的财务数据）
    - GREETING: 问候/闲聊类（如：你好、谢谢等）
    - UNCLEAR: 意图不明确，无法分类
    - OUT_OF_SCOPE: 超出系统知识范围
    """
    FINANCIAL = "financial"          # 财务数据类
    RISK = "risk"                   # 风险因素类
    BUSINESS = "business"            # 业务介绍类
    LEGAL = "legal"                 # 法律合规类
    MANAGEMENT = "management"       # 管理层信息类
    COMPARISON = "comparison"       # 对比分析类
    GREETING = "greeting"           # 问候/闲聊
    UNCLEAR = "unclear"             # 意图不明确
    OUT_OF_SCOPE = "out_of_scope"   # 超出范围


@dataclass
class Entity:
    """
    实体数据类
    
    用于存储从用户问题中提取的实体信息
    """
    text: str   # 实体文本内容
    type: str   # 实体类型（TIME/MONEY/PERCENT/PERSON等）
    start: int  # 实体在原文中的起始位置
    end: int    # 实体在原文中的结束位置


@dataclass
class QueryUnderstandingResult:
    """
    Query理解结果数据类
    
    存储完整的Query理解结果，供后续模块使用
    """
    original_query: str            # 用户原始问题
    intent: IntentType              # 识别出的意图类型
    intent_confidence: float      # 意图识别置信度（0-1）
    entities: List[Entity]         # 提取的实体列表
    normalized_query: str          # 归一化后的问题
    sub_queries: List[str]        # 分解后的子问题列表
    needs_clarification: bool     # 是否需要澄清
    clarification_prompt: Optional[str]  # 澄清提示语


class QueryUnderstandingService:
    """
    Query理解服务类
    
    核心服务类，提供完整的Query理解功能
    """
    
    def __init__(self):
        """
        初始化Query理解服务
        
        在初始化时加载意图识别规则和实体提取规则
        """
        self._init_intent_patterns()   # 初始化意图识别正则规则
        self._init_entity_patterns()   # 初始化实体提取正则规则
        
    def _init_intent_patterns(self):
        """
        初始化意图识别规则
        
        为每种意图类型定义多个正则表达式模式，
        用于匹配用户问题中的关键词
        """
        # 意图识别规则字典，key为意图类型，value为正则表达式列表
        self.intent_patterns = {
            # 财务类：匹配收入、利润、资产等财务相关词汇
            IntentType.FINANCIAL: [
                r"(收入|营收|利润|净利润|毛利率|成本|费用|资产|负债|现金流|财报|业绩)",  # 匹配财务核心术语
                r"(多少|金额|数值|比例|占比|增长率|同比|环比)",  # 匹配财务数据查询量词
                r"(20\d{2}年?|近?期|上?年度)",  # 匹配年份/期间限定词（如'2023年'、'近期'）
            ],
            # 风险类：匹配风险相关词汇
            IntentType.RISK: [
                r"(风险|隐患|问题|挑战|不确定性|波动|下滑|亏损)",  # 匹配通用风险描述词
                r"(经营风险|市场风险|财务风险|法律风险|政策风险)",  # 匹配具体风险类型名称
                r"(有什么|存在|面临|可能|潜在)",  # 匹配风险存在性/可能性描述
            ],
            # 业务类：匹配业务相关词汇
            IntentType.BUSINESS: [
                r"(业务|主营业务|产品|服务|客户|市场|行业|商业模式)",  # 匹配业务核心术语
                r"(做什么|经营|提供|面向|定位)",  # 匹配业务行为/定位动词
            ],
            # 法律类：匹配法律相关词汇
            IntentType.LEGAL: [
                r"(诉讼|仲裁|法律|合规|违法|违规|处罚|监管)",  # 匹配法律合规相关术语
                r"(未决|进行|涉及|存在)",  # 匹配法律状态描述词
            ],
            # 管理层类：匹配管理层相关词汇
            IntentType.MANAGEMENT: [
                r"(董事长|总经理|高管|管理层|创始人|实际控制人|股东|董事|监事)",  # 匹配管理层职位/角色
                r"(是谁|姓名|名字|背景|经历|持股)",  # 匹配管理层信息查询意图
            ],
            # 对比类：匹配对比分析相关词汇
            IntentType.COMPARISON: [
                r"(对比|比较|vs|versus|差距|差异|变化|趋势)",  # 匹配对比操作动词
                r"(和|与|跟|相比|同期|历年)",  # 匹配对比连接词和时间维度
            ],
            # 问候类：匹配问候和结束语
            IntentType.GREETING: [
                r"^(你好|您好|嗨|hello|hi|在吗|在嘛)\s*[!.。]?$",  # 匹配问候语（整行匹配，允许尾随标点）
                r"^(谢谢|感谢|再见|拜拜)\s*[!.。]?$",  # 匹配结束语（整行匹配，允许尾随标点）
            ],
        }
        
    def _init_entity_patterns(self):
        """
        初始化实体提取规则
        
        为每种实体类型定义正则表达式模式
        """
        # 实体提取规则字典
        self.entity_patterns = {
            # 时间实体：匹配年份、日期等
            "TIME": r"(20\d{2}年?|20\d{2}-\d{2}|近?期|上?年度|本?年度)",
            # 金额实体：匹配金钱数额
            "MONEY": r"(\d+[\.,]?\d*\s*[万亿]?元|\d+[\.,]?\d*\s*万?元)",
            # 百分比实体：匹配百分比
            "PERCENT": r"(\d+[\.,]?\d*\s*%|百分之[\d一二三四五六七八九十]+)",
            # 公司实体：匹配公司名称
            "COMPANY": r"([^\s]{2,10}(?:公司|集团|股份|企业|科技))",
            # 人物实体：匹配职位+姓名
            "PERSON": r"([^\s]{2,4}(?:董事长|总经理|CEO|总裁|创始人|实控人|股东))",
        }
        
    def understand(self, query: str, conversation_history: List[Dict] = None) -> QueryUnderstandingResult:
        """
        理解用户Query的主入口方法
        
        执行完整的Query理解流程：
        1. 意图识别
        2. 实体提取
        3. 查询归一化
        4. 问题分解
        5. 消歧处理
        
        Args:
            query: 用户原始问题
            conversation_history: 对话历史，用于指代消解
            
        Returns:
            QueryUnderstandingResult: 包含完整理解结果的数据类
        """
        # 步骤1：识别用户意图和置信度
        intent, confidence = self._recognize_intent(query)
        
        # 步骤2：提取问题中的实体（时间、金钱、人名等）
        entities = self._extract_entities(query)
        
        # 步骤3：归一化问题（如将"去年"转换为"2023年"）
        normalized_query = self._normalize_query(query, entities)
        
        # 步骤4：分解复杂问题为多个子问题
        sub_queries = self._decompose_query(normalized_query, intent)
        
        # 步骤5：检查是否需要向用户澄清
        needs_clarification, clarification_prompt = self._check_clarification(
            query, intent, entities, conversation_history
        )
        
        # 返回完整的理解结果
        return QueryUnderstandingResult(
            original_query=query,              # 原始问题
            intent=intent,                     # 识别出的意图
            intent_confidence=confidence,      # 置信度
            entities=entities,                # 提取的实体
            normalized_query=normalized_query, # 归一化后的问题
            sub_queries=sub_queries,          # 子问题列表
            needs_clarification=needs_clarification,  # 是否需要澄清
            clarification_prompt=clarification_prompt  # 澄清提示
        )
        
    def _recognize_intent(self, query: str) -> Tuple[IntentType, float]:
        """
        识别用户意图
        
        使用规则匹配方法：
        1. 对每种意图类型，计算匹配得分
        2. 选择得分最高的意图
        3. 根据得分计算置信度
        
        Args:
            query: 用户问题
            
        Returns:
            Tuple[IntentType, float]: 意图类型和置信度
        """
        # 初始化所有意图的得分为0.0
        scores = {intent: 0.0 for intent in IntentType}
        
        # 遍历所有意图规则进行匹配
        for intent, patterns in self.intent_patterns.items():
            # 遍历该意图的所有正则模式
            for pattern in patterns:
                # 若当前正则匹配到查询文本，该意图得分累加1.0
                if re.search(pattern, query, re.IGNORECASE):
                    # 注意：同一意图下匹配多个模式会累加得分，非互斥
                    scores[intent] += 1.0
                    
        # 特殊处理：问候语直接返回，置信度0.95
        if scores[IntentType.GREETING] > 0:
            return IntentType.GREETING, 0.95
            
        # 选择得分最高的意图
        if max(scores.values()) > 0:
            best_intent = max(scores, key=scores.get)
            # 置信度基准0.5，每配中一个模式加0.1，上限0.9以防过度自信
            confidence = min(0.9, 0.5 + scores[best_intent] * 0.1)
            return best_intent, confidence
            
        # 无匹配，返回不明确意图，置信度0.3
        return IntentType.UNCLEAR, 0.3
        
    def _extract_entities(self, query: str) -> List[Entity]:
        """
        从问题中提取实体
        
        使用两种方法：
        1. 正则表达式匹配（TIME/MONEY/PERCENT等）
        2. jieba词性标注（人名nr、机构nt等）
        
        Args:
            query: 用户问题
            
        Returns:
            List[Entity]: 提取的实体列表
        """
        entities = []
        
        # 方法1：正则表达式匹配结构化的实体（时间、金额、百分比等）
        for entity_type, pattern in self.entity_patterns.items():
            # 遍历正则所有非重叠匹配位置
            for match in re.finditer(pattern, query):
                entities.append(Entity(
                    text=match.group(),      # 提取匹配到的原始文本片段
                    type=entity_type,         # 标注实体类型（TIME/MONEY/PERCENT等）
                    start=match.start(),     # 记录实体在原文中的字符起始索引
                    end=match.end()          # 记录实体在原文中的字符结束索引（不包含）
                ))

        # 方法2：jieba词性标注补充非结构化实体（人名、机构名）
        words = pseg.cut(query)  # jieba分词并逐一标注词性，返回(word, flag)迭代器
        for word, flag in words:
            # nr开头 = 人名（nr=人名，nr1=复姓，nr2=单姓）
            if flag.startswith('nr'):
                entities.append(Entity(
                    text=word,
                    type="PERSON_NAME",       # 标记为人名类型
                    start=query.find(word),  # 通过字符串查找定位起始索引（首次出现位置）
                    end=query.find(word) + len(word)
                ))
            # nt开头 = 机构名（nt=机构团体名）
            elif flag.startswith('nt'):
                entities.append(Entity(
                    text=word,
                    type="ORGANIZATION",      # 标记为机构类型
                    start=query.find(word),
                    end=query.find(word) + len(word)
                ))
                
        # 去重：使用set记录已见过的(text, type)组合
        seen = set()
        unique_entities = []
        # 按起始位置排序
        for e in sorted(entities, key=lambda x: x.start):
            key = (e.text, e.type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)
                
        return unique_entities
        
    def _normalize_query(self, query: str, entities: List[Entity]) -> str:
        """
        归一化查询
        
        将口语化表达转换为标准表达：
        - "去年" -> "2023年"
        - "今年" -> "2024年"
        - 等等
        
        Args:
            query: 原始问题
            entities: 已提取的实体列表（用于上下文）
            
        Returns:
            str: 归一化后的问题
        """
        normalized = query
        
        # 时间表达映射表：将口语化时间指称映射到标准年份/表达
        time_mapping = {
            "去年": "2023年",
            "今年": "2024年",
            "前年": "2022年",
            "近期": "最近",
            "上期": "上一期",
        }

        # 对查询进行硬替换，将口语时间词替换为标准表达
        for old, new in time_mapping.items():
            # 普通字符串替换，非正则，精确匹配全词
            normalized = normalized.replace(old, new)
            
        return normalized
        
    def _decompose_query(self, query: str, intent: IntentType) -> List[str]:
        """
        分解复杂问题
        
        将复杂问题分解为多个简单的子问题，
        以便分别检索和回答
        
        Args:
            query: 归一化后的问题
            intent: 识别出的意图
            
        Returns:
            List[str]: 子问题列表
        """
        sub_queries = [query]  # 默认返回原问题
        
        # 对比类问题分解
        if intent == IntentType.COMPARISON:
            # 提取公司名
            companies = re.findall(r"([^\s]{2,10}(?:公司|集团))", query)
            # 提取年份
            years = re.findall(r"(20\d{2})年?", query)
            
            # 如果有多个公司，分别生成子问题
            if len(companies) >= 2:
                sub_queries = []
                for company in companies:
                    metric = self._extract_metric(query)  # 提取要对比的指标
                    sub_queries.append(f"{company}的{metric}")
                    
        # 多条件问题分解（用"和"、"与"连接）
        elif "和" in query or "与" in query:
            # 以"和"/"与"为分隔符，将问题拆成多个独立条件片段
            parts = re.split(r"[和与]", query)
            if len(parts) > 1:
                # 剔除长度<=3的残片（如语气词、单个字），只保留有实际语义的子句
                sub_queries = [p.strip() for p in parts if len(p.strip()) > 3]
                
        return sub_queries if sub_queries else [query]
        
    def _extract_metric(self, query: str) -> str:
        """
        从问题中提取要对比的指标
        
        Args:
            query: 用户问题
            
        Returns:
            str: 指标名称
        """
        metrics = ["营收", "收入", "利润", "净利润", "毛利率", "成本", "资产"]
        for m in metrics:
            if m in query:
                return m
        return "财务数据"
        
    def _check_clarification(
        self, 
        query: str, 
        intent: IntentType, 
        entities: List[Entity],
        conversation_history: List[Dict]
    ) -> Tuple[bool, Optional[str]]:
        """
        检查是否需要向用户澄清
        
        触发澄清的条件：
        1. 意图不明确
        2. 问题超出知识范围
        3. 财务问题缺少时间信息
        4. 存在指代需要消解
        
        Args:
            query: 用户问题
            intent: 识别的意图
            entities: 提取的实体
            conversation_history: 对话历史
            
        Returns:
            Tuple[bool, Optional[str]]: (是否需要澄清, 澄清提示语)
        """
        # 条件1：意图不明确
        if intent == IntentType.UNCLEAR:
            return True, "抱歉，我没有理解您的问题。您可以问关于公司财务、业务、风险、管理层等方面的问题。"
            
        # 条件2：超出范围
        if intent == IntentType.OUT_OF_SCOPE:
            return True, "抱歉，这个问题超出了我的知识范围。我只能回答关于招股说明书内容的问题。"
            
        # 条件3：财务问题缺少时间实体
        if intent == IntentType.FINANCIAL:
            has_time = any(e.type == "TIME" for e in entities)
            if not has_time:
                return True, "您想了解哪个时间段的财务数据？比如2023年或2024年。"
                
        # 条件4：指代消解（"该公司"、"它"等）
        if conversation_history:
            pronouns = ["该公司", "这家公司", "该企业", "它"]
            for pronoun in pronouns:
                if pronoun in query:
                    # 从历史中查找公司名
                    for hist in reversed(conversation_history):
                        if "公司" in hist.get("query", ""):
                            company = self._extract_company_from_history(hist["query"])
                            if company:
                                return True, f"您指的是{company}吗？"
                                
        return False, None
        
    def _extract_company_from_history(self, query: str) -> Optional[str]:
        """
        从历史查询中提取公司名
        
        Args:
            query: 历史查询
            
        Returns:
            Optional[str]: 公司名或None
        """
        match = re.search(r"([^\s]{2,10}(?:公司|集团|股份))", query)
        return match.group(1) if match else None


# 全局服务实例（单例模式）
_query_understanding_service = None

def get_query_understanding_service() -> QueryUnderstandingService:
    """
    获取Query理解服务实例（单例）
    
    使用单例模式确保全局只有一个服务实例，
    避免重复初始化和资源浪费
    
    Returns:
        QueryUnderstandingService: 服务实例
    """
    global _query_understanding_service
    # 如果实例不存在，创建新实例
    if _query_understanding_service is None:
        _query_understanding_service = QueryUnderstandingService()
    return _query_understanding_service
