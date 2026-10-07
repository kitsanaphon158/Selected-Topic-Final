"""เรียก Groq LLM สำหรับ query rewriting และสร้างคำตอบ"""
from __future__ import annotations

from groq import Groq, GroqError

from rag.prompt import REWRITE_PROMPT, SYSTEM_PROMPT

# llama-3.3-70b-versatile ถูกถอดจาก Groq แล้ว; gpt-oss-120b ตอบตัวบทภาษาไทยได้ถูกต้องที่สุดในโมเดลที่เหลือ
MODEL = "openai/gpt-oss-120b"
HISTORY_MESSAGES = 6  # 3 รอบล่าสุด
REWRITE_HISTORY_CHARS = 500


class LLMError(RuntimeError):
    """Groq ใช้งานไม่ได้ (เครือข่าย, rate limit, key ผิด ฯลฯ)"""


def get_client(api_key: str, max_retries: int = 2) -> Groq:
    # Groq แบบฟรีจำกัด 8,000 token/นาที — SDK รอแล้วลองใหม่เมื่อเจอ 429; ในแอปลองไม่กี่ครั้งเพื่อไม่ให้ผู้ใช้รอนาน
    return Groq(api_key=api_key, max_retries=max_retries, timeout=30)


def _recent(history: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in history[-HISTORY_MESSAGES:]]


def _chat(client, messages: list[dict], temperature: float, max_tokens: int) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            # โมเดลนี้คิดก่อนตอบ และโทเคนที่ใช้คิดนับรวมใน max_tokens จึงตั้งเพดานเผื่อไว้
            max_tokens=max_tokens,
            reasoning_effort="low",
        )
    except GroqError as e:
        raise LLMError(str(e)) from e
    return (resp.choices[0].message.content or "").strip()


def rewrite_query(client, history: list[dict], question: str) -> str:
    convo = "\n".join(
        f"{'ผู้ใช้' if m['role'] == 'user' else 'ผู้ช่วย'}: {m['content'][:REWRITE_HISTORY_CHARS]}"
        for m in _recent(history)
    ) or "(ไม่มี)"
    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": f"ประวัติการสนทนา:\n{convo}\n\nคำถามล่าสุด: {question}"},
    ]
    out = _chat(client, messages, temperature=0.0, max_tokens=512)
    return out.splitlines()[0].strip() if out else question


def generate_answer(client, history: list[dict], question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_recent(history),
        {"role": "user", "content": f"{context}\n\nคำถาม: {question}"},
    ]
    answer = _chat(client, messages, temperature=0.1, max_tokens=1536)
    if not answer:  # โมเดลใช้โทเคนไปกับการคิดจนหมด — ถือเป็นความล้มเหลว ไม่ใช่คำตอบว่าง
        raise LLMError("LLM คืนคำตอบว่าง")
    return answer
