"""ประกอบขั้นตอน RAG ต่อหนึ่งคำถาม: rewrite → retrieve → threshold → generate → ตรวจคำตอบ"""
from __future__ import annotations

from dataclasses import dataclass, field

from rag import llm
from rag.prompt import NOT_FOUND, build_context, is_not_found
from rag.retriever import DEFAULT_THRESHOLD, Hit, is_relevant

MAX_QUESTION_CHARS = 1000


@dataclass
class RagResult:
    answer: str | None  # None เฉพาะเมื่อ error
    hits: list[Hit] = field(default_factory=list)
    search_query: str = ""
    error: str | None = None


def validate_question(question: str) -> str | None:
    if not question.strip():
        return "กรุณาพิมพ์คำถาม"
    if len(question) > MAX_QUESTION_CHARS:
        return f"คำถามยาวเกินไป (สูงสุด {MAX_QUESTION_CHARS:,} ตัวอักษร)"
    return None


def answer_question(
    question: str,
    history: list[dict],
    retriever,
    client,
    *,
    k: int = 4,
    laws: set[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> RagResult:
    problem = validate_question(question)
    if problem:
        raise ValueError(problem)
    question = question.strip()

    search_query = question
    if history:
        try:
            search_query = llm.rewrite_query(client, history, question) or question
        except llm.LLMError:
            search_query = question

    hits = retriever.search(search_query, k=k, laws=laws)
    if not is_relevant(hits, threshold):
        return RagResult(NOT_FOUND, [], search_query)

    try:
        answer = llm.generate_answer(client, history, question, build_context(hits))
    except llm.LLMError as e:
        return RagResult(None, hits, search_query, error=str(e))

    if is_not_found(answer):
        return RagResult(NOT_FOUND, [], search_query)
    return RagResult(answer, hits, search_query)
