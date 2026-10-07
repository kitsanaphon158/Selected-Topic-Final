from unittest.mock import MagicMock

import pytest

from rag import pipeline
from rag.llm import LLMError
from rag.prompt import NOT_FOUND
from rag.retriever import Hit
from tests.helpers import make_chunk

HISTORY = [
    {"role": "user", "content": "ถูกเลิกจ้างได้ค่าชดเชยเท่าไร"},
    {"role": "assistant", "content": "ขึ้นกับอายุงาน [1]"},
]


class FakeRetriever:
    def __init__(self, cosine: float):
        self.cosine = cosine
        self.queries: list[str] = []

    def search(self, query, k=4, laws=None):
        self.queries.append(query)
        if laws is not None and not laws:
            return []
        return [Hit(make_chunk("labor", "118", "มาตรา 118 ค่าชดเชย"), 0.03, self.cosine)]


def run(retriever, history=(), question="ทำงาน 3 ปีถูกเลิกจ้างได้เท่าไร", **kw):
    client = MagicMock()
    result = pipeline.answer_question(question, list(history), retriever, client, threshold=0.8, **kw)
    return result, client


def test_below_threshold_refuses_without_calling_llm():
    result, client = run(FakeRetriever(0.5))
    assert result.answer == NOT_FOUND and result.hits == [] and result.error is None
    client.chat.completions.create.assert_not_called()


def test_no_law_selected_refuses_without_calling_llm():
    result, client = run(FakeRetriever(0.95), laws=set())
    assert result.answer == NOT_FOUND
    client.chat.completions.create.assert_not_called()


def test_no_law_selected_skips_rewrite_even_with_history(monkeypatch):
    def no_rewrite(*a):
        raise AssertionError("ไม่ควรเรียก LLM เมื่อไม่ได้เลือกกฎหมาย")

    monkeypatch.setattr(pipeline.llm, "rewrite_query", no_rewrite)
    result, client = run(FakeRetriever(0.95), history=HISTORY, question="แล้วถ้า 5 ปีล่ะ", laws=set())
    assert result.answer == NOT_FOUND
    client.chat.completions.create.assert_not_called()


def test_relevant_hits_produce_grounded_answer(monkeypatch):
    seen = {}

    def fake_generate(client, history, question, context):
        seen["context"] = context
        return "ได้ค่าชดเชย 180 วัน [1]"

    monkeypatch.setattr(pipeline.llm, "generate_answer", fake_generate)
    result, _ = run(FakeRetriever(0.9))
    assert result.answer == "ได้ค่าชดเชย 180 วัน [1]"
    assert len(result.hits) == 1 and result.error is None
    assert "[1] พระราชบัญญัติคุ้มครองแรงงาน" in seen["context"]


def test_answer_is_cleaned_of_repeated_disclaimers(monkeypatch):
    monkeypatch.setattr(
        pipeline.llm, "generate_answer",
        lambda *a: "ได้ 180 วัน [1]\nหมายเหตุ: ข้อมูลเบื้องต้น ไม่ใช่คำปรึกษาทางกฎหมาย\nหมายเหตุ: ข้อมูลเบื้องต้น ไม่ใช่คำปรึกษาทางกฎหมาย",
    )
    result, _ = run(FakeRetriever(0.9))
    assert result.answer == "ได้ 180 วัน [1]"


def test_answer_that_is_only_disclaimers_counts_as_an_error(monkeypatch):
    monkeypatch.setattr(pipeline.llm, "generate_answer", lambda *a: "หมายเหตุ: ไม่ใช่คำปรึกษาทางกฎหมาย\nหมายเหตุ:")
    result, _ = run(FakeRetriever(0.9))
    assert result.answer is None and result.error and result.hits


def test_llm_refusal_with_prefix_hides_sources(monkeypatch):
    monkeypatch.setattr(pipeline.llm, "generate_answer", lambda *a: "ขออภัย ไม่พบข้อมูลในเอกสารที่มีอยู่")
    result, _ = run(FakeRetriever(0.9))
    assert result.answer == NOT_FOUND and result.hits == []


def test_llm_failure_keeps_retrieved_sections(monkeypatch):
    def boom(*a):
        raise LLMError("rate limited")

    monkeypatch.setattr(pipeline.llm, "generate_answer", boom)
    result, _ = run(FakeRetriever(0.9))
    assert result.answer is None
    assert result.error == "rate limited"
    assert [h.chunk.section for h in result.hits] == ["118"]


def test_first_message_skips_rewrite(monkeypatch):
    def no_rewrite(*a):
        raise AssertionError("ไม่ควร rewrite คำถามแรก")

    monkeypatch.setattr(pipeline.llm, "rewrite_query", no_rewrite)
    monkeypatch.setattr(pipeline.llm, "generate_answer", lambda *a: "ok [1]")
    retriever = FakeRetriever(0.9)
    result, _ = run(retriever)
    assert retriever.queries == ["ทำงาน 3 ปีถูกเลิกจ้างได้เท่าไร"]
    assert result.search_query == "ทำงาน 3 ปีถูกเลิกจ้างได้เท่าไร"


def test_follow_up_searches_with_rewritten_query(monkeypatch):
    monkeypatch.setattr(pipeline.llm, "rewrite_query", lambda c, h, q: "ทำงาน 5 ปีถูกเลิกจ้างได้ค่าชดเชยเท่าไร")
    monkeypatch.setattr(pipeline.llm, "generate_answer", lambda *a: "ok [1]")
    retriever = FakeRetriever(0.9)
    result, _ = run(retriever, history=HISTORY, question="แล้วถ้า 5 ปีล่ะ")
    assert retriever.queries == ["ทำงาน 5 ปีถูกเลิกจ้างได้ค่าชดเชยเท่าไร"]
    assert result.search_query == "ทำงาน 5 ปีถูกเลิกจ้างได้ค่าชดเชยเท่าไร"


def test_rewrite_failure_falls_back_to_original_question(monkeypatch):
    def boom(*a):
        raise LLMError("down")

    monkeypatch.setattr(pipeline.llm, "rewrite_query", boom)
    monkeypatch.setattr(pipeline.llm, "generate_answer", lambda *a: "ok [1]")
    retriever = FakeRetriever(0.9)
    result, _ = run(retriever, history=HISTORY, question="แล้วถ้า 5 ปีล่ะ")
    assert retriever.queries == ["แล้วถ้า 5 ปีล่ะ"]
    assert result.answer == "ok [1]"


def test_non_thai_first_message_is_rewritten_into_a_thai_search_query(monkeypatch):
    monkeypatch.setattr(pipeline.llm, "rewrite_query", lambda c, h, q: "ลูกจ้างมีสิทธิหยุดพักผ่อนประจำปีกี่วัน")
    monkeypatch.setattr(pipeline.llm, "generate_answer", lambda *a: "At least 6 working days [1]")
    retriever = FakeRetriever(0.9)
    result, _ = run(retriever, question="How many days of annual leave do I get?")
    assert retriever.queries == ["ลูกจ้างมีสิทธิหยุดพักผ่อนประจำปีกี่วัน"]
    assert result.answer == "At least 6 working days [1]"


@pytest.mark.parametrize("bad", ["", "   ", "ก" * 1001])
def test_validate_question_rejects_blank_and_too_long(bad):
    assert pipeline.validate_question(bad)
    with pytest.raises(ValueError):
        pipeline.answer_question(bad, [], FakeRetriever(0.9), MagicMock())


def test_validate_question_accepts_normal_text():
    assert pipeline.validate_question("ลาป่วยได้กี่วัน") is None
    assert pipeline.validate_question("ก" * 1000) is None
