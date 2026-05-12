"""测试 role_manager.py：角色提示和风格"""
import pytest

from role_manager import (
    get_role_prompt,
    get_style_prompt,
    get_role_style_prompt,
    ROLE_PROMPTS,
    ROLE_STYLES,
    DEFAULT_ROLE,
    DEFAULT_STYLE,
)


class TestRolePrompt:
    def test_lawyer_prompt(self):
        """律师角色提示包含法律相关关键词"""
        prompt = get_role_prompt("律师")
        assert "法律" in prompt or "法律顾问" in prompt or "律师" in prompt

    def test_doctor_prompt(self):
        """医生角色提示包含医学相关关键词"""
        prompt = get_role_prompt("医生")
        assert "医学" in prompt or "医生" in prompt or "诊断" in prompt

    def test_unknown_role(self):
        """未知角色返回默认（医生）"""
        prompt = get_role_prompt("程序员")
        default_prompt = get_role_prompt(DEFAULT_ROLE)
        assert prompt == default_prompt

    def test_all_roles_defined(self):
        """所有配置的角色都有定义"""
        for role in ["律师", "医生"]:
            assert role in ROLE_PROMPTS


class TestStylePrompt:
    def test_all_styles_have_content(self):
        """所有风格都有描述"""
        for style in ["严谨专业", "温和耐心", "简洁干练", "亲切易懂", "默认"]:
            prompt = get_style_prompt(style)
            assert len(prompt) > 10

    def test_unknown_style(self):
        """未知风格返回默认"""
        prompt = get_style_prompt("不存在风格")
        assert prompt == get_style_prompt(DEFAULT_STYLE)


class TestRoleStylePrompt:
    def test_combined(self):
        """角色风格组合"""
        result = get_role_style_prompt("律师", "严谨专业")
        assert "法律顾问" in result or "法律" in result
        assert "严谨" in result

    def test_default_style(self):
        """不传 style 时使用默认"""
        result = get_role_style_prompt("医生")
        default_style = get_style_prompt(DEFAULT_STYLE)
        # 默认风格的内容应该在结果中（作为"说话风格要求"的一部分）
        assert "说话风格" in result or default_style in result
