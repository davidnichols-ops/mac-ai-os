"""Tests for app/core/state.py — execution state tracking.

Covers:
- Defaults and budgets
- is_step_budget_exceeded / is_error_budget_exceeded
- check_infinite_loop (detection + recording)
- truncate_history (keeps system+user + tail)
- add_message / add_user_message / add_assistant_message / add_tool_result
"""

from __future__ import annotations

from app.core.state import ExecutionState
from app.models.types import Message, ToolCall

# --- Defaults ----------------------------------------------------------------


class TestDefaults:
    def test_defaults(self):
        s = ExecutionState()
        assert s.goal == ""
        assert s.messages == []
        assert s.step == 0
        assert s.max_steps == 30
        assert s.errors == 0
        assert s.max_errors == 10
        assert s.tool_calls == 0
        assert s.finished is False
        assert s.final_answer == ""
        assert s.workspace == "."
        assert s.max_history_messages == 20
        assert s.seen_actions == []


# --- Budgets -----------------------------------------------------------------


class TestBudgets:
    def test_step_budget_not_exceeded(self):
        s = ExecutionState(max_steps=10)
        s.step = 9
        assert s.is_step_budget_exceeded is False

    def test_step_budget_exceeded(self):
        s = ExecutionState(max_steps=10)
        s.step = 10
        assert s.is_step_budget_exceeded is True

    def test_error_budget_not_exceeded(self):
        s = ExecutionState(max_errors=5)
        s.errors = 4
        assert s.is_error_budget_exceeded is False

    def test_error_budget_exceeded(self):
        s = ExecutionState(max_errors=5)
        s.errors = 5
        assert s.is_error_budget_exceeded is True


# --- Infinite loop detection -------------------------------------------------


class TestInfiniteLoop:
    def test_first_call_not_loop(self):
        s = ExecutionState()
        assert s.check_infinite_loop("shell", {"command": "ls"}) is False
        assert len(s.seen_actions) == 1

    def test_repeat_within_last_three_is_loop(self):
        s = ExecutionState()
        s.check_infinite_loop("shell", {"command": "ls"})
        assert s.check_infinite_loop("shell", {"command": "ls"}) is True

    def test_different_args_not_loop(self):
        s = ExecutionState()
        s.check_infinite_loop("shell", {"command": "ls"})
        assert s.check_infinite_loop("shell", {"command": "pwd"}) is False

    def test_different_tool_not_loop(self):
        s = ExecutionState()
        s.check_infinite_loop("shell", {"command": "ls"})
        assert s.check_infinite_loop("file_read", {"path": "x"}) is False

    def test_repeat_after_three_distinct_not_loop(self):
        # only the last 3 actions are checked; need 4 distinct to push
        # the first out of the last-3 window.
        s = ExecutionState()
        s.check_infinite_loop("t", {"a": 1})
        s.check_infinite_loop("t", {"a": 2})
        s.check_infinite_loop("t", {"a": 3})
        s.check_infinite_loop("t", {"a": 4})
        # {"a":1} is now outside the last-3 window [a2, a3, a4]
        assert s.check_infinite_loop("t", {"a": 1}) is False

    def test_args_order_independent(self):
        s = ExecutionState()
        s.check_infinite_loop("t", {"a": 1, "b": 2})
        # json.dumps with sort_keys=True — order shouldn't matter
        assert s.check_infinite_loop("t", {"b": 2, "a": 1}) is True


# --- History truncation ------------------------------------------------------


class TestTruncateHistory:
    def test_no_truncation_when_under_limit(self):
        s = ExecutionState(max_history_messages=20)
        for i in range(10):
            s.messages.append(Message(role="user", content=str(i)))
        s.truncate_history()
        assert len(s.messages) == 10

    def test_no_truncation_at_exact_limit(self):
        s = ExecutionState(max_history_messages=20)
        for i in range(20):
            s.messages.append(Message(role="user", content=str(i)))
        s.truncate_history()
        assert len(s.messages) == 20

    def test_truncation_keeps_first_two_and_tail(self):
        s = ExecutionState(max_history_messages=6)
        for i in range(10):
            s.messages.append(Message(role="user", content=f"m{i}"))
        s.truncate_history()
        # keep first 2 + last (6-2)=4
        assert len(s.messages) == 6
        assert s.messages[0].content == "m0"
        assert s.messages[1].content == "m1"
        assert s.messages[2].content == "m6"
        assert s.messages[-1].content == "m9"

    def test_truncation_with_few_messages_keeps_all(self):
        s = ExecutionState(max_history_messages=20)
        s.messages.append(Message(role="system", content="sys"))
        s.messages.append(Message(role="user", content="hi"))
        s.truncate_history()
        assert len(s.messages) == 2


# --- Message helpers ---------------------------------------------------------


class TestMessageHelpers:
    def test_add_message(self):
        s = ExecutionState()
        m = Message(role="user", content="hi")
        s.add_message(m)
        assert s.messages == [m]

    def test_add_user_message(self):
        s = ExecutionState()
        s.add_user_message("hello")
        assert len(s.messages) == 1
        assert s.messages[0].role == "user"
        assert s.messages[0].content == "hello"

    def test_add_assistant_message_no_tools(self):
        s = ExecutionState()
        s.add_assistant_message("thinking...")
        assert s.messages[0].role == "assistant"
        assert s.messages[0].content == "thinking..."
        assert s.messages[0].tool_calls == []

    def test_add_assistant_message_with_tools(self):
        s = ExecutionState()
        tc = ToolCall(id="c1", name="shell", arguments={"command": "ls"})
        s.add_assistant_message("running", tool_calls=[tc])
        assert s.messages[0].tool_calls == [tc]

    def test_add_tool_result(self):
        s = ExecutionState()
        s.add_tool_result("c1", "shell", "output here")
        m = s.messages[0]
        assert m.role == "tool"
        assert m.content == "output here"
        assert m.tool_call_id == "c1"
        assert m.name == "shell"
