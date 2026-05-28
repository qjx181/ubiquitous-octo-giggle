# -*- coding: utf-8 -*-
"""backend/app/core/pipeline/state_machine.py — Pipeline 状态机

作用：管理 Pipeline 运行生命周期，支持断点续传和回滚保护

借鉴 HiveWard 的 state_machine.py 设计：
  - 状态枚举：定义 Pipeline 运行的所有可能状态
  - 转换规则：定义合法的状态转换路径
  - 快照机制：在关键状态保存上下文快照，支持回滚
  - 事件通知：状态变化时触发回调

应用场景：
  1. 断点续传：Pipeline 在 RETRIEVAL 阶段超时 → 恢复到 RETRIEVAL 阶段重试
  2. 回滚保护：LLM 生成失败 → 回滚到 RERANKING 阶段，避免重复检索
  3. 状态监控：实时查看 Pipeline 执行进度
  4. 超时控制：某个阶段执行过久 → 自动中断并回滚

设计原理：
  1. PipelineState：枚举所有可能的状态
  2. StateTransition：定义状态转换规则（from_state → to_state）
  3. PipelineStateMachine：管理状态转换和快照

面试官可能问：
  Q: 状态机在 RAG 系统中的作用是什么？
  A: 三个核心作用：
     1. 断点续传：网络超时或服务重启后，可以从上次失败的地方继续，而不是从头开始
     2. 回滚保护：LLM 生成失败时，可以回滚到检索阶段，避免重复调用 Embedding
     3. 状态监控：前端可以实时显示 Pipeline 执行进度，提升用户体验

  Q: 如何保证回滚后数据一致性？
  A: 每个状态转换时保存 PipelineContext 的快照（深拷贝）。
     回滚时恢复到快照状态，不会出现部分更新的问题。
     但要注意：外部副作用（如 API 调用、数据库写入）无法回滚，
     所以只在没有外部副作用的阶段保存快照。

  Q: 状态机和 try-except 的区别是什么？
  A: try-except 只能捕获异常，状态机可以：
     1. 定义合法的状态转换路径（防止非法跳转）
     2. 保存历史状态（支持回滚）
     3. 触发事件通知（监控、日志）
     4. 支持超时控制（自动中断）
"""

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
from enum import Enum

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Pipeline 状态枚举
    
    状态转换图：
    PENDING → QUERY_UNDERSTANDING → EMBEDDING → RETRIEVAL → RERANKING → LLM_GENERATION → COMPLETED
                                                                                      ↘ FAILED
    
    特殊状态：
    - PAUSED：暂停等待人工审核（预留）
    - ROLLBACK：正在回滚到某个历史状态
    """
    PENDING = "pending"                          # 初始状态
    QUERY_UNDERSTANDING = "query_understanding"  # Query 理解阶段
    EMBEDDING = "embedding"                      # Embedding 编码阶段
    RETRIEVAL = "retrieval"                      # 向量检索阶段
    RERANKING = "reranking"                      # 重排序阶段
    LLM_GENERATION = "llm_generation"            # LLM 生成阶段
    COMPLETED = "completed"                      # 完成
    FAILED = "failed"                            # 失败
    PAUSED = "paused"                            # 暂停（预留）
    ROLLBACK = "rollback"                        # 回滚中


# 合法的状态转换路径
VALID_TRANSITIONS = {
    PipelineState.PENDING: [PipelineState.QUERY_UNDERSTANDING],
    PipelineState.QUERY_UNDERSTANDING: [PipelineState.EMBEDDING, PipelineState.FAILED],
    PipelineState.EMBEDDING: [PipelineState.RETRIEVAL, PipelineState.FAILED],
    PipelineState.RETRIEVAL: [PipelineState.RERANKING, PipelineState.FAILED],
    PipelineState.RERANKING: [PipelineState.LLM_GENERATION, PipelineState.FAILED],
    PipelineState.LLM_GENERATION: [PipelineState.COMPLETED, PipelineState.FAILED],
    PipelineState.COMPLETED: [],  # 终态
    PipelineState.FAILED: [PipelineState.PENDING],  # 失败后可以重试
    PipelineState.PAUSED: [PipelineState.QUERY_UNDERSTANDING, PipelineState.EMBEDDING,
                           PipelineState.RETRIEVAL, PipelineState.RERANKING,
                           PipelineState.LLM_GENERATION],  # 暂停后可以恢复到任意阶段
    PipelineState.ROLLBACK: [PipelineState.QUERY_UNDERSTANDING, PipelineState.EMBEDDING,
                             PipelineState.RETRIEVAL, PipelineState.RERANKING,
                             PipelineState.LLM_GENERATION],  # 回滚到任意阶段
}


@dataclass
class StateSnapshot:
    """状态快照：保存某个时刻的 Pipeline 状态和上下文
    
    属性：
      - state: Pipeline 状态
      - context: PipelineContext 的深拷贝
      - timestamp: 快照时间
      - stage_times: 各阶段耗时
    """
    state: PipelineState
    context: Any  # PipelineContext 的深拷贝
    timestamp: float = field(default_factory=time.time)
    stage_times: Dict[str, float] = field(default_factory=dict)


class PipelineStateMachine:
    """Pipeline 状态机：管理运行生命周期
    
    使用方式：
        sm = PipelineStateMachine()
        
        # 开始执行
        sm.transition(PipelineState.QUERY_UNDERSTANDING, ctx)
        
        # 保存快照
        sm.save_snapshot(ctx)
        
        # 状态转换
        sm.transition(PipelineState.EMBEDDING, ctx)
        
        # 失败时回滚
        if error:
            ctx = sm.rollback(PipelineState.QUERY_UNDERSTANDING)
        
        # 完成
        sm.transition(PipelineState.COMPLETED, ctx)
    
    设计对比 HiveWard：
      HiveWard 的 state_machine.py 实现了更复杂的状态管理：
      - 支持并行状态（多个 Slot 同时执行）
      - 支持审批节点（暂停等待人工确认）
      - 支持状态持久化（保存到数据库）
      
      我们的 PipelineStateMachine 实现了核心的"状态转换+快照回滚"功能，
      是 HiveWard 状态机的简化版本。
    """
    
    def __init__(self):
        """初始化状态机"""
        self.state = PipelineState.PENDING
        self.snapshots: Dict[PipelineState, StateSnapshot] = {}
        self.stage_times: Dict[str, float] = {}
        self.current_stage_start: Optional[float] = None
        
        # 事件回调
        self.on_state_change: Optional[Callable[[PipelineState, PipelineState], None]] = None
        self.on_rollback: Optional[Callable[[PipelineState, PipelineState], None]] = None
    
    def transition(self, new_state: PipelineState, context: Any = None):
        """状态转换
        
        Args:
            new_state: 目标状态
            context: 当前 PipelineContext（用于保存快照）
            
        Raises:
            ValueError: 如果状态转换不合法
        """
        # 检查转换是否合法
        if new_state not in VALID_TRANSITIONS.get(self.state, []):
            raise ValueError(
                f"非法状态转换: {self.state.value} → {new_state.value}。"
                f"合法的目标状态: {[s.value for s in VALID_TRANSITIONS.get(self.state, [])]}"
            )
        
        # 记录上一个状态
        old_state = self.state
        
        # 计算上一个阶段的耗时
        if self.current_stage_start is not None:
            elapsed = time.time() - self.current_stage_start
            self.stage_times[old_state.value] = elapsed
            logger.info(f"⏱️ 阶段 {old_state.value} 耗时: {elapsed:.3f}s")
        
        # 转换状态
        self.state = new_state
        self.current_stage_start = time.time()
        
        # 保存快照（如果有上下文）
        if context is not None:
            self.save_snapshot(context)
        
        # 触发回调
        if self.on_state_change:
            self.on_state_change(old_state, new_state)
        
        logger.info(f"🔄 状态转换: {old_state.value} → {new_state.value}")
    
    def save_snapshot(self, context: Any):
        """保存快照
        
        Args:
            context: 当前 PipelineContext
        """
        # 深拷贝上下文
        snapshot = StateSnapshot(
            state=self.state,
            context=copy.deepcopy(context),
            stage_times=dict(self.stage_times)
        )
        
        self.snapshots[self.state] = snapshot
        logger.debug(f"📸 保存快照: {self.state.value}")
    
    def rollback(self, target_state: PipelineState) -> Any:
        """回滚到指定状态
        
        Args:
            target_state: 目标状态
            
        Returns:
            PipelineContext: 回滚后的上下文
            
        Raises:
            ValueError: 如果目标状态没有快照
        """
        if target_state not in self.snapshots:
            raise ValueError(
                f"无法回滚到 {target_state.value}：没有该状态的快照。"
                f"可用的快照: {[s.value for s in self.snapshots.keys()]}"
            )
        
        # 获取快照
        snapshot = self.snapshots[target_state]
        
        # 记录回滚
        old_state = self.state
        self.state = PipelineState.ROLLBACK
        
        # 触发回调
        if self.on_rollback:
            self.on_rollback(old_state, target_state)
        
        logger.info(f"⏪ 回滚: {old_state.value} → {target_state.value}")
        
        # 恢复状态
        self.state = target_state
        self.stage_times = dict(snapshot.stage_times)
        self.current_stage_start = time.time()
        
        # 返回深拷贝的上下文
        return copy.deepcopy(snapshot.context)
    
    def get_progress(self) -> Dict[str, Any]:
        """获取执行进度
        
        Returns:
            Dict[str, Any]: 进度信息
        """
        # 定义阶段顺序
        stages = [
            PipelineState.QUERY_UNDERSTANDING,
            PipelineState.EMBEDDING,
            PipelineState.RETRIEVAL,
            PipelineState.RERANKING,
            PipelineState.LLM_GENERATION
        ]
        
        # 计算当前阶段索引
        current_idx = -1
        if self.state in stages:
            current_idx = stages.index(self.state)
        
        # 计算进度百分比
        progress = 0
        if self.state == PipelineState.COMPLETED:
            progress = 100
        elif self.state == PipelineState.FAILED:
            progress = (current_idx / len(stages)) * 100 if current_idx >= 0 else 0
        elif current_idx >= 0:
            progress = (current_idx / len(stages)) * 100
        
        return {
            "state": self.state.value,
            "progress": progress,
            "current_stage": self.state.value if current_idx >= 0 else None,
            "stage_times": dict(self.stage_times),
            "total_time": sum(self.stage_times.values())
        }
    
    def is_completed(self) -> bool:
        """是否已完成"""
        return self.state == PipelineState.COMPLETED
    
    def is_failed(self) -> bool:
        """是否失败"""
        return self.state == PipelineState.FAILED
    
    def is_running(self) -> bool:
        """是否正在运行"""
        return self.state not in [
            PipelineState.PENDING,
            PipelineState.COMPLETED,
            PipelineState.FAILED
        ]
    
    def reset(self):
        """重置状态机"""
        self.state = PipelineState.PENDING
        self.snapshots.clear()
        self.stage_times.clear()
        self.current_stage_start = None
        logger.info("🔄 状态机已重置")


# ============ 辅助函数 ============

def create_state_machine() -> PipelineStateMachine:
    """创建状态机实例"""
    return PipelineStateMachine()


def with_state_machine(sm: PipelineStateMachine):
    """装饰器：为 Pipeline 添加状态机管理
    
    使用方式：
        sm = create_state_machine()
        
        @with_state_machine(sm)
        async def run_pipeline(ctx):
            # 执行 Pipeline
            return ctx
    """
    def decorator(func):
        async def wrapper(ctx):
            try:
                # 开始执行
                sm.transition(PipelineState.QUERY_UNDERSTANDING, ctx)
                
                # 执行 Pipeline
                result = await func(ctx)
                
                # 完成
                sm.transition(PipelineState.COMPLETED, result)
                return result
                
            except Exception as e:
                # 失败
                sm.transition(PipelineState.FAILED, ctx)
                raise e
        
        return wrapper
    return decorator
