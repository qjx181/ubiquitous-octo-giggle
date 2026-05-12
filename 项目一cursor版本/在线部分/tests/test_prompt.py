"""测试 prompt.py：Prompt 组装逻辑"""
import pytest

from services.prompt import build_general_system_prompt, build_professional_rag_prompt


class TestBuildGeneralSystemPrompt:
    def test_basic_structure(self):
        """基本结构应包含角色提示和风格说明"""
        result = build_general_system_prompt("律师", "严谨专业")
        assert "你是一个专业、友好、可靠的中文智能助手" in result
        assert "严谨专业" in result
        assert "普通交流、问候、功能咨询" in result

    def test_custom_role_hint(self):
        """自定义角色提示应生效"""
        hint = "你是一名资深法律顾问"
        result = build_general_system_prompt("律师", "默认", hint)
        assert hint in result

    def test_different_styles(self):
        """不同风格应出现在 prompt 中"""
        for style in ["严谨专业", "温和耐心", "简洁干练", "亲切易懂", "默认"]:
            result = build_general_system_prompt("医生", style)
            assert style in result


class TestBuildProfessionalRagPrompt:
    def test_basic_structure(self):
        """RAG prompt 应包含检索资料和规则"""
        context = "[1] 民法典第xxx条"
        result = build_professional_rag_prompt("律师", "默认", context)
        assert "检索资料" in result
        assert context in result
        assert "引用编号" in result

    def test_different_roles_have_different_rules(self):
        """律师和医生的角色规则不同"""
        lawyer_prompt = build_professional_rag_prompt("律师", "默认", "context")
        doctor_prompt = build_professional_rag_prompt("医生", "默认", "context")
        assert "胜诉" in lawyer_prompt
        assert "用药" in doctor_prompt

    def test_from_web_flag(self):
        """from_web=True 时应标记为互联网资料"""
        result = build_professional_rag_prompt("律师", "默认", "context", from_web=True)
        assert "互联网" in result

    def test_from_web_false(self):
        """from_web=False 时应标记为知识库资料"""
        result = build_professional_rag_prompt("律师", "默认", "context", from_web=False)
        assert "知识库" in result
