"""测试 memory.py：会话存储（强制走内存模式）"""
import time
import pytest

# 强制走内存模式：模拟 Redis 不可用
import memory as mem_mod

# 重置全局状态
mem_mod.redis_available = False
mem_mod.MEMORY_HISTORY.clear()
mem_mod.MEMORY_ROLE.clear()
mem_mod.MEMORY_SESSIONS.clear()


class TestHistoryStorage:
    def setup_method(self):
        mem_mod.MEMORY_HISTORY.clear()

    def test_append_and_get(self):
        """追加消息后能正确读取"""
        mem_mod.append_message("user1", "session1", "user", "你好")
        history = mem_mod.get_history("user1", "session1")
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "你好"

    def test_append_multiple_messages(self):
        """多条消息按顺序追加"""
        mem_mod.append_message("u1", "s1", "user", "问1")
        mem_mod.append_message("u1", "s1", "assistant", "答1")
        mem_mod.append_message("u1", "s1", "user", "问2")
        history = mem_mod.get_history("u1", "s1")
        assert len(history) == 3
        assert history[0]["content"] == "问1"
        assert history[1]["content"] == "答1"
        assert history[2]["content"] == "问2"

    def test_clear_history(self):
        """清空后历史为空"""
        mem_mod.append_message("u1", "s1", "user", "内容")
        mem_mod.clear_history("u1", "s1")
        assert mem_mod.get_history("u1", "s1") == []

    def test_max_history_truncation(self):
        """超过 MAX_HISTORY_MESSAGES 应截断"""
        original_max = mem_mod.MAX_HISTORY_MESSAGES
        mem_mod.MAX_HISTORY_MESSAGES = 3
        try:
            for i in range(5):
                mem_mod.append_message("u1", "s1", "user", str(i))
            history = mem_mod.get_history("u1", "s1")
            assert len(history) == 3
            # 应该保留最近的 3 条
            assert history[0]["content"] == "2"
            assert history[-1]["content"] == "4"
        finally:
            mem_mod.MAX_HISTORY_MESSAGES = original_max

    def test_empty_params(self):
        """空参数不报错"""
        mem_mod.append_message("", "", "", "")

    def test_invalid_role(self):
        """无效 role 不追加"""
        mem_mod.append_message("u1", "s1", "invalid_role", "内容")
        assert mem_mod.get_history("u1", "s1") == []

    def test_get_nonexistent(self):
        """不存在的会话返回空列表"""
        assert mem_mod.get_history("nouser", "nosession") == []


class TestSessionRole:
    def setup_method(self):
        mem_mod.MEMORY_ROLE.clear()
        mem_mod.MEMORY_SESSIONS.clear()

    def test_set_and_get_role(self):
        """设置角色后能正确获取"""
        mem_mod.set_session_role("u1", "s1", "律师")
        assert mem_mod.get_session_role("u1", "s1") == "律师"

    def test_invalid_role_falls_back(self):
        """无效角色应返回默认"""
        mem_mod.set_session_role("u1", "s1", "无效角色")
        assert mem_mod.get_session_role("u1", "s1") == "默认助手"

    def test_default_role(self):
        """未设置角色返回默认"""
        assert mem_mod.get_session_role("u1", "s1") == "默认助手"

    def test_list_sessions(self):
        """列出用户的所有会话"""
        mem_mod.set_session_role("u1", "s1", "医生")
        mem_mod.set_session_role("u1", "s2", "律师")
        sessions = mem_mod.list_sessions("u1")
        assert "s1" in sessions
        assert "s2" in sessions

    def test_session_expiry(self):
        """
        BUG 检测: _clean_expired_memory() 只检查 MEMORY_HISTORY 中的过期键
        如果只设置了角色(无历史消息)，角色永远不会被清理
        """
        mem_mod.set_session_role("u1", "s_expire", "医生")
        # 模拟过期 - 现在 MEMORY_ROLE 里有角色但 MEMORY_HISTORY 里没有
        key = mem_mod._mem_key("u1", "s_expire")
        old = mem_mod.MEMORY_ROLE[key]
        old["expire"] = time.time() - 1  # 已过期
        mem_mod.MEMORY_ROLE[key] = old

        mem_mod._clean_expired_memory()
        # 当前行为：_clean_expired_memory 遍历 MEMORY_HISTORY 的 key，
        # 但 s_expire 不在 MEMORY_HISTORY 中，所以没有被清理
        # 即使 MEMORY_ROLE 中对应的条目已经过期
        result = mem_mod.get_session_role("u1", "s_expire")
        assert result == "医生", (
            "BUG: 角色过期未被清理，因为 _clean_expired_memory 只检查 "
            "MEMORY_HISTORY 的 expire 字段，但此会话只有角色无历史"
        )


class TestHistorySummary:
    def setup_method(self):
        mem_mod.MEMORY_HISTORY.clear()

    def test_set_and_get_summary(self):
        """设置摘要后能读取"""
        mem_mod.set_history_summary("u1", "s1", "这是会话摘要")
        assert mem_mod.get_history_summary("u1", "s1") == "这是会话摘要"

    def test_default_summary(self):
        """无摘要返回空字符串"""
        assert mem_mod.get_history_summary("u1", "s1") == ""


class TestCreateSession:
    def setup_method(self):
        mem_mod.MEMORY_ROLE.clear()
        mem_mod.MEMORY_HISTORY.clear()

    def test_create_session_sets_role_and_history(self):
        """创建会话同时设置角色和空历史"""
        mem_mod.create_session("u1", "s1", "医生")
        assert mem_mod.get_session_role("u1", "s1") == "医生"
        assert mem_mod.get_history("u1", "s1") == []
