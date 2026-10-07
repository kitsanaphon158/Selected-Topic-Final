"""เรียก Groq LLM สำหรับ query rewriting และสร้างคำตอบ"""
from __future__ import annotations

from groq import Groq, GroqError

from rag.prompt import REWRITE_PROMPT, SYSTEM_PROMPT

MODEL = "llama-3.3-70b-versatile"
HISTORY_MESSAGES = 6  # 3 รอบล่าสุด
REWRITE_HISTORY_CHARS = 500


class LLMError(RuntimeError):
    """Groq ใช้งานไม่ได้ (เครือข่าย, rate limit, key ผิด ฯลฯ)"""


def get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key, max_retries=2, timeout=30)


def _recent(history: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in history[-HISTORY_MESSAGES:]]


def _chat(client, messages: list[dict], temperature: float, max_tokens: int) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
    except GroqError as e:
        raise LLMError(str(e)) from e
    return (resp.choices[0].message.content or "").strip()


def rewrite_query(client, history: list[dict], question: str) -> str:
    convo = "\n".join(
        f"{'ผู้ใช้' if m['role'] == 'user' else 'ผู้ช่วย'}: {m['content'][:REWRITE_HISTORY_CHARS]}"
        for m in _recent(history)
    )
    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": f"ประวัติการสนทนา:\n{convo}\n\nคำถามล่าสุด: {question}"},
    ]
    out = _chat(client, messages, temperature=0.0, max_tokens=200)
    return out.splitlines()[0].strip() if out else question


def generate_answer(client, history: list[dict], question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_recent(history),
        {"role": "user", "content": f"{context}\n\nคำถาม: {question}"},
    ]
    return _chat(client, messages, temperature=0.1, max_tokens=1024)
