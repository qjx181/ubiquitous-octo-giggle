# -*- coding: utf-8 -*-
"""
Query理解模块
功能：意图识别、实体提取、问题分解、消歧处理 + 公司实体识别

模块职责：
1. 意图识别 - 识别用户问题是关于财务、风险、业务等哪个方面
2. 实体提取 - 从问题中提取时间、金钱、人名等实体
3. 公司实体识别 - 从问题中识别公司名称，自动推断知识库过滤条件
4. 问题分解 - 将复杂问题分解为多个简单子问题
5. 消歧处理 - 判断是否需要向用户澄清问题
"""

# 【作用】导入正则表达式模块，用于模式匹配
# 【原理】re.search在字符串中搜索匹配，re.findall提取所有匹配，re.match从开头匹配
import re
# 【作用】导入类型注解，用于类型提示
from typing import Dict, List, Optional, Tuple
# 【作用】导入数据类装饰器，用于创建数据结构
# 【原理】@dataclass自动生成__init__、__repr__、__eq__等方法，减少样板代码
from dataclasses import dataclass
# 【作用】导入枚举类，用于定义常量
# 【原理】Enum是Python内置的枚举类型，比普通类更安全、更可读
from enum import Enum
# 【作用】导入jieba分词库，用于中文分词和词性标注
# 【原理】jieba是Python最流行的中文分词库，支持精确模式和搜索引擎模式
import jieba
# 【作用】导入jieba的词性标注模块
# 【原理】pseg.cut(text)返回(word, flag)迭代器，flag是词性标记如"nr"=人名、"nt"=机构名
import jieba.posseg as pseg

# 【作用】导入知识库名称解析器，用于多公司场景下的自动 kb_name 推断
# 【设计意图】这是"架构级解决跨公司数据混淆"的核心——在查询理解层自动识别
#            公司实体并填入 kb_name，不依赖用户手动指定。
from .kb_mapping import resolve_kb_name, list_knowledge_bases


class IntentType(Enum):
    """
    作用：意图类型枚举

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
    # 【作用】定义每个枚举成员，值是对应的字符串标识
    # 【原理】枚举成员是类级别的常量，通过 IntentType.FINANCIAL 访问
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
    作用：实体数据类

    用于存储从用户问题中提取的实体信息
    """
    # 【作用】实体文本内容（如"2023年"、"1.5亿元"）
    text: str
    # 【作用】实体类型（TIME/MONEY/PERCENT/PERSON等）
    type: str
    # 【作用】实体在原文中的起始位置（字符索引）
    start: int
    # 【作用】实体在原文中的结束位置（字符索引，不包含）
    end: int


@dataclass
class QueryUnderstandingResult:
    """
    作用：Query理解结果数据类

    存储完整的Query理解结果，供后续模块使用

    字段说明：
      - original_query: 用户原始输入（不做任何修改）
      - intent: 意图类型枚举（FINANCIAL/RISK/BUSINESS/MANAGEMENT/COMPARISON/UNCLEAR等）
      - intent_confidence: 意图识别置信度（0.0~1.0）
      - entities: 从问题中提取的实体（时间、金额、人名、机构等）
      - normalized_query: 归一化后的查询（"去年"→"2023年"）
      - sub_queries: 问题分解结果，复杂问题拆成多个子问题
      - needs_clarification: 是否需追问用户补充信息
      - clarification_prompt: 追问提示语（如"您想了解哪个时间段的财务数据？"）
      - detected_kb_name: 自动检测到的知识库名称（如"招股说明书1"），
        用于多公司场景下自动加 kb_name 过滤，None 表示未检测到

    面试官可能问：
      Q: 为什么需要QueryUnderstandingResult这个数据类？
      A: 把理解结果封装成一个对象，调用方（retrieval.py的_run_retrieval_pipeline）
         只需要检查needs_clarification字段、取normalized_query和sub_queries，
         不需要关心内部的规则匹配细节。这是"数据与逻辑分离"。

      Q: needs_clarification为True时，后续检索还进行吗？
      A: 不进行。retrieval.py发现True就直接返回澄清文本，不走embedding和检索。
         因为如果连"问什么"都不确定，检索没有意义。这是"早停"策略。

      Q: sub_queries有什么用？为什么不直接用normalized_query？
      A: 复杂问题（如"对比A公司和B公司的营收"）可以拆成
         ["A公司的营收", "B公司的营收"]两个子查询分别检索，
         然后合并结果。比用完整问题检索更精准。但不建议拆太多，
         通常是3个以内，否则检索延迟会线性增长。

      Q: detected_kb_name 和外部传入的 kb_name 哪个优先级高？
      A: 外部传入的 kb_name（用户/API显式指定）优先级最高。
         detected_kb_name 只在外部未指定时作为兜底使用。
         这样既保留了显式过滤的能力，又为模糊查询提供了自动推断。
    """
    # 【作用】用户输入的原始问题文本
    original_query: str
    # 【作用】识别出的意图类型（如财务、风险等）
    intent: IntentType
    # 【作用】意图识别置信度（0-1之间的浮点数）
    intent_confidence: float
    # 【作用】从问题中提取的实体列表
    entities: List[Entity]
    # 【作用】归一化后的问题（如"去年"→"2023年"）
    normalized_query: str
    # 【作用】分解后的子问题列表（用于复杂问题多路检索）
    sub_queries: List[str]
    # 【作用】是否需要向用户澄清（如缺少时间信息时）
    needs_clarification: bool
    # 【作用】如果需要澄清，澄清提示语的内容
    clarification_prompt: Optional[str]
    # 【作用】自动检测到的知识库名称（多公司场景，None=未检测到）
    detected_kb_name: Optional[str] = None


class QueryUnderstandingService:
    """
    作用：Query理解服务类

    核心服务类，提供完整的Query理解功能

    能力范围：
      1. 意图识别 - 6类意图（财务/风险/业务/管理/对比/不明）+ 超出范围检测
      2. 实体提取 - 正则提取结构化实体 + jieba提取非结构化实体
      3. 公司实体识别 - 从问题中识别公司名称，自动推断知识库过滤条件
      4. 查询归一化 - 口语→标准表达转换
      5. 问题分解 - 复杂问题拆成多个子查询
      6. 消歧澄清 - 关键信息缺失时追问用户

    设计原理：
      - 纯规则驱动（正则+关键词），不依赖模型
      - 在__init__中初始化所有规则，后续understand()直接使用
      - 意图识别用打分制，非简单匹配

    面试官可能问：
      Q: 为什么Query理解不用模型而是纯规则？
      A: 招股说明书的问题是领域特定的（财务/风险/业务），规则足够覆盖。
         模型方案需要标注数据、训练、部署，复杂度高很多。
         规则方案的好处是：可解释（能说清楚为什么分到某个意图）、
         零延迟（不需要加载模型）、容易调整（加个规则就行）。
         如果后期需要更好的泛化能力，可以加一个小分类模型做意图识别。

      Q: 意图识别是怎么打分的？
      A: _detect_intent中，每个意图有多个正则模式。遍历所有模式，
         hits计数命中次数，best_intent = 得分最高的。
         置信度= min(0.9, 0.5 + hits * 0.1)。基准0.5是"至少有点信心"，
         每命中一个模式+0.1，上限0.9（不设1.0因为规则总有不准的时候）。
         全不命中返回UNCLEAR+0.3。

      Q: 实体提取为什么要用两种方法？
      A: 时间（"2023年"）、金额（"1000万"）、百分比（"20%"）等结构化实体
         格式固定，正则最快最准。人名、机构名格式多变（如"阿里巴巴集团控股有限公司"
         和"阿里"），jieba词性标注更适合。两者互补。

      Q: 消歧（澄清）条件有哪些？
      A: (1) 意图不明确 → "没理解您的问题"
         (2) 超出范围 → "超出了知识范围"
         (3) 财务问题缺少时间 → "您想了解哪个时间段？"（这是用户最喜欢的功能）
         (4) 公司名模糊 → "请问您指的是哪家公司？"
    """

    def __init__(self):
        """
        作用：初始化Query理解服务

        在初始化时加载意图识别规则和实体提取规则
        """
        # 【作用】初始化意图识别的正则规则
        # 【逻辑】在__init__中准备好所有规则，后续understand()直接使用
        self._init_intent_patterns()
        # 【作用】初始化实体提取的正则规则
        self._init_entity_patterns()

    def _init_intent_patterns(self):
        """
        作用：初始化意图识别规则

        为每种意图类型定义多个正则表达式模式，
        用于匹配用户问题中的关键词
        """
        # 【作用】意图识别规则字典，key为意图类型，value为正则表达式列表
        # 【原理】每个意图类型对应多个正则模式，匹配数量越多，意图越明确
        self.intent_patterns = {
            # 【作用】财务类：匹配收入、利润、资产等财务相关词汇
            IntentType.FINANCIAL: [
                r"(收入|营收|盈利|利润|净利润|毛利率|净利率|成本|费用|资产|负债|现金流|财报|业绩)",
                r"(多少|金额|数值|比例|占比|增长率|同比|环比|ROE|净资产|市盈率)",
                r"(20\d{2}年?|近?期|上?年度|募集资金|IPO|上市)",
                r"(营业收入|营业利润|毛利|净利|总资产|总负债|应收账款)",
            ],
            # 【作用】风险类：匹配风险相关词汇
            IntentType.RISK: [
                r"(风险|隐患|问题|挑战|不确定性|波动|下滑|亏损|退市)",
                r"(经营风险|市场风险|财务风险|法律风险|政策风险)",
                r"(有什么|存在|面临|可能|潜在|依赖|集中度|单一客户|应收账款|坏账|商誉)",
            ],
            # 【作用】业务类：匹配业务相关词汇
            IntentType.BUSINESS: [
                r"(业务|主营业务|产品|服务|客户|市场|行业|商业模式|领域|赛道|供应商)",
                r"(做什么|经营|提供|面向|定位|核心优势|竞争优势|竞争力|技术|研发|专利)",
                r"(主营|客户群体|市场份额|应用场景|解决方案|上下游|产业链|产品线|收入构成|营收构成)",
                r"(主营是什么|主要做什么|靠什么赚钱|盈利模式|产品介绍|业务介绍)",
                r"(发行|股本|总股本|发行后|公开发行|战略配售)",  # IPO/发行相关
                r"(注册资本|成立时间|成立日期|经营范围|注册地|法定代表人|公司类型)",  # 公司基本信息
                r"(保荐机构|承销商|审计机构|律师事务所|评估机构)",  # 中介机构
                r"(子公司|分公司|控股|参股|关联方|合资)",  # 子公司/股权关系
                r"(股权结构|股权|持股比例|控股股东|实际控制人|股东构成)",  # 股权相关
            ],
            # 【作用】法律类：匹配法律相关词汇
            IntentType.LEGAL: [
                r"(诉讼|仲裁|法律|合规|违法|违规|处罚|监管|行政处罚)",
                r"(知识产权|专利纠纷|侵权|未决|进行|涉及|存在|纠纷)",
            ],
            # 【作用】管理层类：匹配管理层相关词汇
            IntentType.MANAGEMENT: [
                r"(董事长|总经理|高管|管理层|创始人|实际控制人|股东|董事|监事)",
                r"(核心团队|高管薪酬|持股比例|团队背景|管理层介绍|谁创办)",
                r"(是谁|姓名|名字|背景|经历|持股)",
            ],
            # 【作用】对比类：匹配对比分析相关词汇
            IntentType.COMPARISON: [
                r"(对比|比较|vs|versus|差距|差异|变化|趋势)",
                r"(和|与|跟|相比|同期|历年)",
            ],
            # 【作用】问候类：匹配问候和结束语
            IntentType.GREETING: [
                r"^(你好|您好|嗨|hello|hi|在吗|在嘛)\s*[!.。]?$",
                r"^(谢谢|感谢|再见|拜拜)\s*[!.。]?$",
            ],
        }

    def _init_entity_patterns(self):
        """
        作用：初始化实体提取规则

        为每种实体类型定义正则表达式模式
        """
        # 【作用】实体提取规则字典
        # 【原理】每种实体类型对应一个正则表达式，用于从问题中抓取特定格式的信息
        self.entity_patterns = {
            # 【作用】时间实体：匹配年份、日期等
            "TIME": r"(20\d{2}年?|(?:最近|近)[一二三四五六七八九十\d]*年|近?期|上?年度|本?年度)",
            # 【作用】金额实体：匹配金钱数额
            "MONEY": r"(\d+[.,]?\d*\s*[万亿]?元|\d+[.,]?\d*\s*万?元)",
            # 【作用】百分比实体：匹配百分比
            "PERCENT": r"(\d+[.,]?\d*\s*%|百分之[\d一二三四五六七八九十]+)",
            # 【作用】公司实体：匹配公司名称
            "COMPANY": r"([^\s]{2,10}(?:公司|集团|股份|企业|科技))",
            # 【作用】人物实体：匹配职位+姓名
            "PERSON": r"([^\s]{2,4}(?:董事长|总经理|CEO|总裁|创始人|实控人|股东))",
        }

    def understand(self, query: str, conversation_history: Optional[List[Dict]] = None) -> QueryUnderstandingResult:
        """
        作用：理解用户Query的主入口方法

        执行完整的Query理解流程：
        0. 公司实体识别（v2.0新增，在意图识别之前运行，用于多公司知识库过滤）
        1. 意图识别
        2. 实体提取
        3. 查询归一化
        4. 问题分解
        5. 消歧处理
        6. kb_name消歧（v2.0新增：模糊问题时如果检测到可能的多公司歧义，提示用户指定）

        Args:
            query: 用户原始问题
            conversation_history: 对话历史，用于指代消解

        Returns:
            QueryUnderstandingResult: 包含完整理解结果的数据类
        """
        # 【作用】步骤0：公司实体识别（v2.0）
        # 【原理】在意图识别之前运行，因为公司名检测是独立于意图的
        # 【设计意图】无论用户问财务、业务还是风险，公司实体识别都在最前面执行
        detected_kb_name = resolve_kb_name(query)

        # 存一份原始问题用于 kb_name 澄清判断
        # 如果公司名模糊但问题内容指向特定行业/财务数据，触发 kb_name 消歧
        original_query = query

        # 【作用】步骤1：识别用户意图和置信度
        intent, confidence = self._recognize_intent(query)

        # 【作用】步骤2：提取问题中的实体（时间、金钱、人名等）
        entities = self._extract_entities(query)

        # 【作用】步骤3：归一化问题（如将"去年"转换为"2023年"）
        normalized_query = self._normalize_query(query, entities)

        # 【作用】步骤4：分解复杂问题为多个子问题
        sub_queries = self._decompose_query(normalized_query, intent)

        # 【作用】步骤5：检查是否需要向用户澄清
        needs_clarification, clarification_prompt = self._check_clarification(
            query, intent, entities, conversation_history
        )

        # 【作用】步骤6：如果不需要其他澄清但公司名模糊，触发 kb_name 消歧（v2.0）
        # 【设计意图】解决"公司营收多少"这种既没说年份也没说公司的模糊问题
        #           场景1：财务问题无公司名无时间 → 先问时间（原有逻辑已有）
        #           场景2：业务/风险类问题无公司名 → 检查是否有多公司歧义
        if not needs_clarification and detected_kb_name is None:
            kb_needs = self._check_kb_clarification(query, intent, conversation_history)
            if kb_needs:
                needs_clarification, clarification_prompt = kb_needs

        # 【作用】返回完整的理解结果，打包成QueryUnderstandingResult
        return QueryUnderstandingResult(
            original_query=original_query,
            intent=intent,
            intent_confidence=confidence,
            entities=entities,
            normalized_query=normalized_query,
            sub_queries=sub_queries,
            needs_clarification=needs_clarification,
            clarification_prompt=clarification_prompt,
            detected_kb_name=detected_kb_name,
        )

    def _check_kb_clarification(
        self,
        query: str,
        intent: IntentType,
        conversation_history: Optional[List[Dict]],
    ) -> Optional[Tuple[bool, str]]:
        """
        作用：检查是否因公司名模糊需要向用户澄清（v2.0 新增）

        触发条件：
          1. 问题涉及公司核心信息（业务/财务/风险）但没指名公司
          2. 对话历史中也没有明确指代
          3. 不是对比类问题（对比类天然跨公司）

        Args:
            query: 用户问题
            intent: 识别的意图
            conversation_history: 对话历史

        Returns:
            Optional[Tuple[bool, str]]: (True, 澄清提示语) 或 None
        """
        # 对比类问题不需要 kb_name 澄清（天然跨公司）
        if intent == IntentType.COMPARISON:
            return None

        # 问候/闲聊不触发
        if intent in (IntentType.GREETING, IntentType.UNCLEAR):
            return None

        # 检查问题本身是否包含"公司""发行"等模糊指代
        # 且不含具体公司名（detected_kb_name 已为 None 才进入此函数）
        company_indicators = [
            r"(公司|企业|发行人|本[公企])",
            r"(营收|收入|利润|资产|负债|风险|业务|产品|技术|专利|股本|发行)",
        ]
        has_indicator = any(re.search(p, query) for p in company_indicators)
        if not has_indicator:
            return None

        # 检查对话历史中是否有公司名信息
        if conversation_history:
            for hist in reversed(conversation_history):
                if isinstance(hist, dict):
                    hist_q = hist.get("query", "") or hist.get("question", "")
                    if hist_q:
                        hist_kb = resolve_kb_name(hist_q)
                        if hist_kb is not None:
                            # 历史中有明确公司，可以推断
                            return None

        # 收集可用知识库列表，生成澄清提示
        kbs = list_knowledge_bases()
        kb_list = "、".join(
            f'"{info["description"]}"' for _, info in kbs.items()
        )

        return True, (
            f"请问您指的是哪家公司？当前可查询以下知识库：{kb_list}。"
            f"您可以直接说公司全称或简称。"
        )

    def _recognize_intent(self, query: str) -> Tuple[IntentType, float]:
        """
        作用：识别用户意图

        使用规则匹配方法：
        1. 对每种意图类型，计算匹配得分
        2. 选择得分最高的意图
        3. 根据得分计算置信度

        Args:
            query: 用户问题

        Returns:
            Tuple[IntentType, float]: 意图类型和置信度
        """
        # 【作用】初始化所有意图的得分为0.0
        scores = {intent: 0.0 for intent in IntentType}

        # 【作用】遍历所有意图规则进行匹配
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                # 【作用】用正则搜索问题文本
                if re.search(pattern, query, re.IGNORECASE):
                    # 【逻辑】匹配到一个模式，该意图得分+1
                    scores[intent] += 1.0

        # 【作用】特殊处理：问候语直接返回，置信度0.95
        if scores[IntentType.GREETING] > 0:
            return IntentType.GREETING, 0.95

        # 【作用】选择得分最高的意图
        if max(scores.values()) > 0:
            best_intent = max(scores, key=lambda x: scores[x] or 0)
            # 【作用】计算置信度：基准0.5，每匹配一个模式+0.1，上限0.9
            confidence = min(0.9, 0.5 + scores[best_intent] * 0.1)
            return best_intent, confidence

        # 【作用】无匹配，返回不明确意图，置信度0.3
        return IntentType.UNCLEAR, 0.3

    def _extract_entities(self, query: str) -> List[Entity]:
        """
        作用：从问题中提取实体

        使用两种方法：
        1. 正则表达式匹配（TIME/MONEY/PERCENT等）
        2. jieba词性标注（人名nr、机构nt等）

        Args:
            query: 用户问题

        Returns:
            List[Entity]: 提取的实体列表
        """
        entities = []

        # 【作用】方法1：正则表达式匹配结构化的实体（时间、金额、百分比等）
        for entity_type, pattern in self.entity_patterns.items():
            # 【作用】遍历正则所有非重叠匹配位置
            for match in re.finditer(pattern, query):
                entities.append(Entity(
                    text=match.group(),      # 提取匹配到的原始文本片段
                    type=entity_type,         # 标注实体类型
                    start=match.start(),     # 记录在原文中的起始位置
                    end=match.end()          # 记录在原文中的结束位置
                ))

        # 【作用】方法2：jieba词性标注补充非结构化实体（人名、机构名）
        # 【原理】pseg.cut返回(word, flag)迭代器，flag是词性标记
        words = pseg.cut(query)
        for word, flag in words:
            # 【作用】nr开头 = 人名（nr=人名，nr1=复姓，nr2=单姓）
            if flag.startswith('nr'):
                entities.append(Entity(
                    text=word,
                    type="PERSON_NAME",       # 标记为人名类型
                    start=query.find(word),
                    end=query.find(word) + len(word)
                ))
            # 【作用】nt开头 = 机构名（nt=机构团体名）
            elif flag.startswith('nt'):
                entities.append(Entity(
                    text=word,
                    type="ORGANIZATION",      # 标记为机构类型
                    start=query.find(word),
                    end=query.find(word) + len(word)
                ))

        # 【作用】实体去重和排序
        # 【原理】用set记录已见过的(text, type)组合，避免重复
        seen = set()
        unique_entities = []
        # 【作用】按起始位置排序，让实体在结果中按原文顺序排列
        for e in sorted(entities, key=lambda x: x.start):
            key = (e.text, e.type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)

        return unique_entities

    def _normalize_query(self, query: str, entities: List[Entity]) -> str:
        """
        作用：归一化查询

        将口语化表达转换为标准表达：
        - "去年" -> "2025年"（动态计算）
        - "今年" -> "2026年"（动态计算）
        - 等等

        Args:
            query: 原始问题
            entities: 已提取的实体列表（用于上下文）

        Returns:
            str: 归一化后的问题
        """
        from datetime import datetime
        normalized = query

        # 【作用】动态计算年份，不再硬编码
        # 【原理】根据当前年份自动推算"去年"、"今年"、"前年"对应的实际年份
        # 【设计意图】避免每年手动更新代码，消除时间漂移bug
        current_year = datetime.now().year
        time_mapping = {
            "去年": f"{current_year - 1}年",
            "今年": f"{current_year}年",
            "前年": f"{current_year - 2}年",
            "近期": "最近",
            "上期": "上一期",
        }

        # 【作用】对查询进行硬替换，将口语时间词替换为标准表达
        for old, new in time_mapping.items():
            normalized = normalized.replace(old, new)

        return normalized

    def _decompose_query(self, query: str, intent: IntentType) -> List[str]:
        """
        作用：分解复杂问题 + 查询扩展

        1. 将复杂问题分解为多个简单的子问题
        2. 对模糊查询进行扩展，生成更具体的检索查询

        Args:
            query: 归一化后的问题
            intent: 识别出的意图

        Returns:
            List[str]: 子问题列表
        """
        # 【作用】默认返回原问题（不做分解）
        sub_queries = [query]

        # 【作用】对比类问题分解：如"对比A公司和B公司的营收"
        if intent == IntentType.COMPARISON:
            # 【作用】提取问题中的公司名
            companies = re.findall(r"([^\s]{2,10}(?:公司|集团))", query)
            # 【作用】提取问题中的年份
            years = re.findall(r"(20\d{2})年?", query)

            # 【作用】如果有多个公司，分别生成子问题
            if len(companies) >= 2:
                sub_queries = []
                for company in companies:
                    metric = self._extract_metric(query)
                    sub_queries.append(f"{company}的{metric}")

        # 【作用】多条件问题分解：用"和"、"与"连接的问题，拆成多个独立条件
        elif "和" in query or "与" in query:
            parts = re.split(r"[和与]", query)
            if len(parts) > 1:
                # 【作用】剔除长度<=3的残片（如语气词、单个字）
                sub_queries = [p.strip() for p in parts if len(p.strip()) > 3]

        # 【作用】查询扩展：对模糊的"X是什么"模式生成更具体的检索查询
        # 【原理】向量检索对宽泛查询（如"公司的技术标准是什么"）效果差，
        #         因为embedding模型难以匹配到具体段落。
        #         通过扩展为多个更具体的查询，提高召回率。
        # 【设计意图】这是解决"检索正确但回答错误"的关键优化
        expanded = self._expand_vague_query(query)
        if len(expanded) > 1:
            sub_queries = expanded

        # 【作用】组织架构类问题：检测"下设/部门构成/组成部分/由哪些部门"等关键词
        #          生成含具体部门名称的子查询，提高LM部门结构类chunk的检索命中
        # 【原理】向量检索对"销售部由哪些部门构成"这种层级关系问题效果不稳定，
        #          因为BGE-M3可能把语义相近的"销售团队划分为3部分"当成更匹配的chunk。
        #          通过加入具体部门名称作为关键词，BM25+向量双路检索互补。
        # 【设计意图】解决"组织架构"类问题的检索精度
        structure_keywords = ["下设", "部门构成", "组成部分", "由哪些部门", "部门设置", "组织架构", "销售部", "市场部"]
        if any(kw in query for kw in structure_keywords):
            # 添加部门结构关键词到子查询中
            dept_queries = [query]
            if "下设" not in query:
                dept_queries.append(f"{query} 下设")
            if "部门" in query and "组织结构" not in query:
                dept_queries.append(query.replace("部门", "组织结构"))
            sub_queries = list(dict.fromkeys(dept_queries))  # 去重保持顺序

        # 【作用】公司信息查询改写：添加"母公司"或"发行人"以区分母公司和子公司
        # 【原理】招股说明书中同时包含母公司和子公司信息，向量检索可能返回子公司信息
        #         通过在查询中添加"母公司"或"发行人"，可以提高母公司信息的检索优先级
        # 【设计意图】解决Q1（主要业务）、Q6（注册资本）、Q9（经营范围）等返回子公司信息的问题
        company_keywords = ["注册资本", "成立时间", "成立日期", "经营范围", "主要业务", "主营业务"]
        if any(kw in query for kw in company_keywords):
            # 如果查询中没有"母公司"或"发行人"，则添加
            if "母公司" not in query and "发行人" not in query:
                # 生成一个添加了"母公司"的扩展查询
                expanded_query = query.replace("公司的", "母公司（发行人）的")
                if expanded_query == query:
                    expanded_query = f"母公司（发行人）{query}"
                sub_queries = [expanded_query, query]  # 优先使用扩展查询

        # 【作用】确保至少返回一个子问题
        return sub_queries if sub_queries else [query]

    def _expand_vague_query(self, query: str) -> List[str]:
        """
        作用：对模糊查询进行扩展

        检测模式如"公司的X是什么"，生成多个更具体的检索查询。
        例如：
          "公司的技术标准是什么" → [
            "公司的技术标准是什么",
            "公司参与制定的技术标准",
            "公司参与制定的行业技术标准规范"
          ]

        Args:
            query: 归一化后的问题

        Returns:
            List[str]: 扩展后的查询列表（包含原始查询）
        """
        # 【作用】提取核心名词短语
        # 【原理】匹配"的X是什么"或"的X有哪些"模式，提取X
        match = re.search(r"的([^是什有]{2,15})(?:是什么|有哪些|是怎样的|怎么样|如何)", query)
        if not match:
            return [query]

        keyword = match.group(1).strip()
        if len(keyword) < 2:
            return [query]

        # 【作用】基于关键词生成扩展查询
        # 【原理】将宽泛的"X是什么"扩展为"参与制定的X"、"行业X"等更具体的表述
        # 【逻辑】招股说明书中，技术标准、资质荣誉等通常以"参与制定"、"获得"等动词出现
        expanded = [query]

        # 【作用】动词扩展：根据关键词类型添加相关动词
        verb_expansions = {
            "技术标准": ["参与制定的", "参与制定的行业"],
            "资质": ["获得的", "取得的"],
            "荣誉": ["获得的", "荣获的"],
            "奖项": ["获得的", "荣获的"],
            "认证": ["获得的", "通过的"],
            "专利": ["拥有的", "申请的"],
            "核心技术": ["掌握的", "自主研发的"],
            "竞争优势": ["具备的", "主要的"],
        }

        for key, verbs in verb_expansions.items():
            if key in keyword:
                for verb in verbs:
                    expanded_query = query.replace(keyword, f"{verb}{keyword}")
                    if expanded_query not in expanded:
                        expanded.append(expanded_query)
                break

        return expanded

    def _extract_metric(self, query: str) -> str:
        """
        作用：从问题中提取要对比的指标

        Args:
            query: 用户问题

        Returns:
            str: 指标名称
        """
        # 【作用】常见财务指标列表，按优先级排列
        metrics = ["营收", "收入", "利润", "净利润", "毛利率", "成本", "资产"]
        for m in metrics:
            if m in query:
                return m
        # 【作用】如果没匹配到具体指标，返回默认值
        return "财务数据"

    def _check_clarification(
        self,
        query: str,
        intent: IntentType,
        entities: List[Entity],
        conversation_history: Optional[List[Dict]]
    ) -> Tuple[bool, Optional[str]]:
        """
        作用：检查是否需要向用户澄清

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
        # 【作用】条件1：意图不明确
        if intent == IntentType.UNCLEAR:
            return True, "抱歉，我没有理解您的问题。您可以问关于公司财务、业务、风险、管理层等方面的问题。"

        # 【作用】条件2：超出范围
        if intent == IntentType.OUT_OF_SCOPE:
            return True, "抱歉，这个问题超出了我的知识范围。我只能回答关于招股说明书内容的问题。"

        # 【作用】条件3：财务问题缺少时间实体
        if intent == IntentType.FINANCIAL:
            has_time = any(e.type == "TIME" for e in entities)
            if not has_time:
                # 【作用】豁免：以下财务场景本身不需要时间
                # 1) IPO/发行相关：募集资金、发行价、上市等
                # 2) 当前状态：最新、当前、截至、目前等
                # 3) 结构性问题：资产结构、股权结构等
                # 4) 募集用途：补充流动资金、投资项目等
                no_time_needed_patterns = [
                    r"(募集资金|发行|IPO|上市|新股|招股|发行价|定价)",
                    r"(最新|当前|最近一期|截至|目前|现在)",
                    r"(结构|占比|比例|构成|明细|分布)",
                    r"(补充流动|投资|扩产|偿还|项目)",
                    r"(注册资本|成立|注册时间|注册日期)",  # 一次性信息，不需要时间
                    r"(变化趋势|趋势|走势|变化情况|变动|波动|历史|历年)",  # 趋势分析不需要具体时间点
                ]
                needs_time = True
                for pat in no_time_needed_patterns:
                    if re.search(pat, query):
                        needs_time = False
                        break
                if needs_time:
                    # 【作用】典型的消歧场景：用户问"公司营收是多少？"但没说年份
                    return True, "您想了解哪个时间段的财务数据？比如2023年或2024年。"

        # 【作用】条件4：指代消解（"该公司"、"它"等）
        if conversation_history:
            pronouns = ["该公司", "这家公司", "该企业", "它"]
            for pronoun in pronouns:
                if pronoun in query:
                    # 【作用】从最近的历史对话中查找公司名
                    for hist in reversed(conversation_history):
                        if "公司" in hist.get("query", ""):
                            company = self._extract_company_from_history(hist["query"])
                            if company:
                                return True, f"您指的是{company}吗？"

        # 【作用】不需要澄清
        return False, None

    def _extract_company_from_history(self, query: str) -> Optional[str]:
        """
        作用：从历史查询中提取公司名

        Args:
            query: 历史查询

        Returns:
            Optional[str]: 公司名或None
        """
        # 【作用】用正则匹配类似"某某公司""某某集团""某某股份"的模式
        match = re.search(r"([^\s]{2,10}(?:公司|集团|股份))", query)
        return match.group(1) if match else None


# 【作用】全局服务实例（单例模式）
# 【原理】_query_understanding_service是模块级变量，确保全局只有一个实例
_query_understanding_service = None


def get_query_understanding_service() -> QueryUnderstandingService:
    """
    作用：获取Query理解服务实例（单例）

    使用单例模式确保全局只有一个服务实例，
    避免重复初始化和资源浪费

    Returns:
        QueryUnderstandingService: 服务实例
    """
    global _query_understanding_service
    if _query_understanding_service is None:
        _query_understanding_service = QueryUnderstandingService()
    return _query_understanding_service
