"""
检索优化模块：基于意图的章节过滤

解决问题：
  - 检索到错误章节的文档（如问供应商却检索到财务数据）
  - 检索结果为空（如注册资本、仓储物流等具体信息找不到）

原理：
  - 根据Query理解的意图类型，自动添加章节过滤条件
  - 将检索范围限定在最相关的章节，提高精度
"""

import re
from typing import Optional, List, Dict
from .query_understanding import IntentType

# 【作用】意图→章节映射表（使用实际Milvus中的章节名称）
# 【原理】招股说明书的章节结构是固定的，每个意图对应最相关的章节
# 【设计意图】将检索范围从"全库搜索"缩小到"相关章节搜索"，大幅提高精度
INTENT_TO_SECTIONS = {
    IntentType.FINANCIAL: [
        "第八节 财务会计信息与管理层分析",
        "重大事项提示",
    ],
    IntentType.RISK: [
        "第四节 风险因素",
        "重大事项提示",
    ],
    IntentType.BUSINESS: [
        "第五节 发行人基本情况",
        "第六节 业务与技术",
        "重大事项提示",
    ],
    IntentType.LEGAL: [
        "第五节 发行人基本情况",
        "第七节 公司治理与独立性",
        "重大事项提示",
    ],
    IntentType.MANAGEMENT: [
        "第七节 公司治理与独立性",
        "第五节 发行人基本情况",
    ],
    IntentType.COMPARISON: [
        "第八节 财务会计信息与管理层分析",
    ],
}

# 【作用】关键词→章节的补充映射
# 【原理】某些具体问题需要更精确的章节定位
# 【设计意图】处理意图识别无法覆盖的具体问题
KEYWORD_TO_SECTIONS = {
    # 注册资本/成立时间：第二节概览+第五节基本情况+未知章节，覆盖母公司+子公司表格
    "注册资本": ["第二节 概览", "第五节 发行人基本情况", "未知"],
    "成立时间": ["第二节 概览", "第五节 发行人基本情况", "未知"],
    "成立日期": ["第二节 概览", "第五节 发行人基本情况", "未知"],
    "主营业务": ["第二节 概览", "第六节 业务与技术"],  # 概览通常有主营业务摘要
    "主要业务": ["第二节 概览", "第六节 业务与技术"],
    "仓储": ["第六节 业务与技术"],
    "物流": ["第六节 业务与技术"],
    "供应商": ["第六节 业务与技术"],
    "客户": ["第六节 业务与技术"],
    # 经营范围：加入第二节概览，概览中通常有经营范围的精炼描述
    "经营范围": ["第二节 概览", "第五节 发行人基本情况"],
    "产品": ["第六节 业务与技术"],
    "技术标准": ["第六节 业务与技术"],
    "专利": ["第六节 业务与技术"],
    "竞争优势": ["第六节 业务与技术"],
    "发行价": ["第三节 本次发行概况"],
    "发行股数": ["第三节 本次发行概况"],
    "发行": ["第三节 本次发行概况", "第二节 概览"],
    "总股本": ["第三节 本次发行概况", "第五节 发行人基本情况"],
    "募集资金": ["第九节 募集资金运用与未来发展规划"],
    "募投项目": ["第九节 募集资金运用与未来发展规划"],
    "保荐机构": ["第三节 本次发行概况"],
    "承销商": ["第三节 本次发行概况"],
    "独立董事": ["第七节 公司治理与独立性"],
    "董事长": ["第七节 公司治理与独立性"],
    "总经理": ["第七节 公司治理与独立性"],
    "实际控制人": ["第五节 发行人基本情况"],
    "股东": ["第五节 发行人基本情况"],
    "股权结构": ["第五节 发行人基本情况"],
    "风险": ["第四节 风险因素"],
    "净利润": ["第八节 财务会计信息与管理层分析"],
    "营收": ["第八节 财务会计信息与管理层分析"],
    "毛利率": ["第八节 财务会计信息与管理层分析"],
    # 控股股东/持股：涉及股东信息和发行概况
    "控股股东": ["第五节 发行人基本情况", "第二节 概览"],
    "持股比例": ["第五节 发行人基本情况", "第三节 本次发行概况"],
}


def get_section_filter(query: str, intent: IntentType) -> Optional[List[str]]:
    """
    作用：根据查询和意图，返回最合适的章节过滤条件列表
    
    参数：
        query: 用户问题
        intent: 识别出的意图类型
    
    返回：
        Optional[List[str]]: 章节过滤条件列表，None表示不使用过滤
    
    设计原理：
        1. 先检查关键词映射（更精确）
        2. 再使用意图映射（更通用）
        3. 返回None表示不使用过滤（全库搜索）
        4. 支持多章节检索（如注册资本可能在"第五节"或"未知"章节）
    """
    # 【作用】先检查关键词映射
    # 【原理】关键词匹配比意图更精确，优先使用
    for keyword, sections in KEYWORD_TO_SECTIONS.items():
        if keyword in query:
            # 【作用】返回所有匹配的章节
            # 【逻辑】某些信息可能分布在多个章节（如注册资本在"第五节"和"未知"）
            return sections
    
    # 【作用】再使用意图映射
    # 【原理】意图识别覆盖更广，但精度稍低
    if intent in INTENT_TO_SECTIONS:
        sections = INTENT_TO_SECTIONS[intent]
        if sections:
            return sections
    
    # 【作用】没有匹配的过滤条件，返回None（全库搜索）
    return None


def get_all_relevant_sections(query: str, intent: IntentType) -> List[str]:
    """
    作用：获取所有相关章节列表（用于多章节检索）
    
    参数：
        query: 用户问题
        intent: 识别出的意图类型
    
    返回：
        List[str]: 相关章节列表
    
    设计原理：
        - 某些问题可能涉及多个章节（如供应商信息可能在业务章节和财务章节）
        - 返回所有相关章节，让检索层做合并
    """
    sections = []
    
    # 【作用】收集关键词匹配的章节
    for keyword, keyword_sections in KEYWORD_TO_SECTIONS.items():
        if keyword in query:
            for s in keyword_sections:
                if s not in sections:
                    sections.append(s)
    
    # 【作用】如果没有关键词匹配，使用意图映射
    if not sections and intent in INTENT_TO_SECTIONS:
        sections = INTENT_TO_SECTIONS[intent][:]
    
    return sections


def should_use_section_filter(query: str) -> bool:
    """
    作用：判断是否应该使用章节过滤
    
    某些问题不适合使用章节过滤：
    - 开放性问题（如"介绍一下这家公司"）
    - 综合性问题（如"公司的主要情况"）
    - 问候语
    
    返回：
        bool: True表示使用章节过滤，False表示全库搜索
    """
    # 【作用】不需要过滤的模式
    no_filter_patterns = [
        r"^(你好|您好|嗨|hello|hi)",
        r"介绍一下",
        r"概况",
        r"综合",
        r"整体",
        r"全面",
    ]
    
    for pattern in no_filter_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False
    
    return True
