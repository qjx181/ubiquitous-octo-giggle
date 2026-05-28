# -*- coding: utf-8 -*-
"""backend/app/core/evolution/__init__.py

自进化模块：从历史查询中学习优化策略

核心组件：
  - QueryLearner: 查询学习器，提供意图提示和检索增强
  - PatternDB: 模式数据库，存储和查询历史模式
"""

from .query_learner import QueryLearner, PatternDB, get_query_learner

__all__ = ["QueryLearner", "PatternDB", "get_query_learner"]
