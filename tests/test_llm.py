from types import SimpleNamespace
from unittest.mock import MagicMock

import groq
import httpx
import pytest

from rag import llm


def fake_client(reply: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
    )
    return client


def make_history(n: int) -> list[dict]:
    return [
        {"role": "user", "content": f"q{i}"} if i % 2 == 0
        else {"role": "assistant", "content": f"a{i}", "sources": [{"label": "x"}]}
        for i in range(n)
    ]


def test_generate_answer_sends_system_recent_history_and_context():
    client = fake_client("ได้ค่าชดเชย 180 วัน [1]")
    out = llm.generate_answer(client, make_history(10), "คำถาม", "<context>x</context>")
    assert out == "ได้ค่าชดเชย 180 วัน [1]"
    kwargs = client.chat.completions.create.call_args.kwargs
    msgs = kwargs["messages"]
    assert kwargs["model"] == llm.MODEL
    assert msgs[0]["role"] == "system"
    assert len(msgs) == 1 + llm.HISTORY_MESSAGES + 1
    assert [m["content"] for m in msgs[1:-1]] == ["q4", "a5", "q6", "a7", "q8", "a9"]
    assert all(set(m) == {"role", "content"} for m in msgs)
    assert "<context>x</context>" in msgs[-1]["content"] and "คำถาม" in msgs[-1]["content"]


def test_rewrite_query_returns_first_line_only():
    client = fake_client("ทำงาน 5 ปีถูกเลิกจ้างได้ค่าชดเชยเท่าไร\nคำอธิบายเพิ่ม")
    history = [{"role": "user", "content": "ถูกเลิกจ้างได้ค่าชดเชยเท่าไร"}, {"role": "assistant", "content": "..."}]
    assert llm.rewrite_query(client, history, "แล้วถ้า 5 ปีล่ะ") == "ทำงาน 5 ปีถูกเลิกจ้างได้ค่าชดเชยเท่าไร"
    msgs = client.chat.completions.create.call_args.kwargs["messages"]
    assert "แล้วถ้า 5 ปีล่ะ" in msgs[-1]["content"]
    assert "ถูกเลิกจ้างได้ค่าชดเชยเท่าไร" in msgs[-1]["content"]


def test_rewrite_query_empty_reply_falls_back_to_question():
    assert llm.rewrite_query(fake_client("   "), make_history(2), "แล้วถ้า 5 ปีล่ะ") == "แล้วถ้า 5 ปีล่ะ"


def test_groq_errors_become_llm_error():
    client = MagicMock()
    client.chat.completions.create.side_effect = groq.APIConnectionError(
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    )
    with pytest.raises(llm.LLMError):
        llm.generate_answer(client, [], "q", "<context></context>")
    with pytest.raises(llm.LLMError):
        llm.rewrite_query(client, make_history(2), "q")
