"""
backend/app/exceptions.py — 异常处理模块
=========================================
作用：定义企业级异常体系，统一错误码和错误信息格式
原理：
  1. 使用枚举(Enum)定义所有错误码，避免魔法数字散落在代码各处
  2. 错误码分类：1xxxx=系统错误、2xxxx=认证错误、3xxxx=业务错误、4xxxx=RAG错误
  3. BusinessException继承FastAPI的HTTPException，兼容FastAPI的错误处理
  4. 前端可以根据error_code做不同的展示（如"检索为空"和"服务异常"显示不同提示）

好处：
  - 统一错误格式，前端处理更简单
  - 错误码有分类，快速定位问题领域
  - BusinessException抛出的异常会被FastAPI自动捕获并转为结构化JSON
"""

from enum import Enum
from fastapi import HTTPException


class ErrorCode(Enum):
    """
    作用：错误码枚举，定义系统中所有可能的错误
    格式：（错误码数字，错误信息，HTTP状态码）
    分类规则：
      1xxxx — 系统级错误（服务器内部、超时、不可用）
      2xxxx — 认证授权错误（未登录、无权限、令牌过期）
      3xxxx — 业务逻辑错误（参数错误、资源不存在、重复）
      4xxxx — RAG领域错误（文档不存在、检索为空、生成失败）
    """

    # =============================================================================
    # 系统错误 1xxxx
    # =============================================================================
    INTERNAL_ERROR = (10001, "Internal Server Error", 500)          # 服务器内部错误
    SERVICE_UNAVAILABLE = (10002, "Service Unavailable", 503)       # 服务不可用
    TIMEOUT_ERROR = (10003, "Request Timeout", 504)                 # 请求超时

    # =============================================================================
    # 认证错误 2xxxx
    # =============================================================================
    UNAUTHORIZED = (20001, "Unauthorized", 401)                     # 未认证
    FORBIDDEN = (20002, "Forbidden", 403)                           # 无权限
    TOKEN_EXPIRED = (20003, "Token Expired", 401)                   # 令牌过期

    # =============================================================================
    # 业务错误 3xxxx
    # =============================================================================
    INVALID_PARAMETER = (30001, "Invalid Parameter", 400)           # 参数无效
    RESOURCE_NOT_FOUND = (30002, "Resource Not Found", 404)         # 资源不存在
    DUPLICATE_RESOURCE = (30003, "Duplicate Resource", 409)        # 资源重复

    # =============================================================================
    # RAG错误 4xxxx
    # =============================================================================
    DOCUMENT_NOT_FOUND = (40001, "Document Not Found", 404)          # 文档不存在
    RETRIEVAL_EMPTY = (40002, "No Relevant Content Found", 404)    # 检索结果为空
    LLM_GENERATION_FAILED = (40003, "Answer Generation Failed", 500) # LLM生成失败

    def __init__(self, code: int, message: str, status_code: int):
        """
        作用：初始化错误码
        原理：枚举的每个成员都是一个元组(code, message, status_code)
              这里手动赋值给属性，方便访问
        参数：
            code: 错误码数字，如40002
            message: 错误信息，如"No Relevant Content Found"
            status_code: HTTP状态码，如404
        """
        self.code = code                # 错误码数字
        self.message = message          # 错误信息
        self.status_code = status_code  # HTTP状态码


class BusinessException(HTTPException):
    """
    作用：业务异常类，统一业务异常格式
    原理：继承FastAPI的HTTPException，FastAPI会自动捕获并返回结构化JSON
    使用示例：
        raise BusinessException(ErrorCode.RETRIEVAL_EMPTY)
        raise BusinessException(ErrorCode.LLM_GENERATION_FAILED, "SGLang服务无响应")
    返回格式：
        {
            "code": 40002,
            "message": "No Relevant Content Found",
            "detail": "未找到与问题相关的信息"
        }
    """

    def __init__(self, error_code: ErrorCode, detail: str = None):
        """
        作用：初始化业务异常
        参数：
            error_code: 错误码枚举
            detail: 详细的错误描述（可选，默认使用error_code的message）
        """
        self.error_code = error_code
        super().__init__(
            status_code=error_code.status_code,
            detail={
                "code": error_code.code,                    # 错误码
                "message": error_code.message,               # 错误信息
                "detail": detail or error_code.message,      # 详细描述
            }
        )


# =============================================================================
# 便捷函数
# =============================================================================

def raise_if_empty(result: list, message: str = "未找到相关数据"):
    """
    作用：如果结果列表为空则抛出业务异常
    原理：在检索结果为空时直接中断流程，避免把空结果传给LLM
    参数：
        result: 查询结果列表
        message: 错误信息
    异常：
        BusinessException: 如果结果为空
    """
    if not result:
        raise BusinessException(ErrorCode.RETRIEVAL_EMPTY, message)


def raise_if_not_found(obj, message: str = "资源不存在"):
    """
    作用：如果对象为空则抛出业务异常
    参数：
        obj: 查询对象
        message: 错误信息
    异常：
        BusinessException: 如果对象为空
    """
    if obj is None:
        raise BusinessException(ErrorCode.RESOURCE_NOT_FOUND, message)
