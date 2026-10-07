"""ประเมินระบบด้วย test_questions.csv

  python evaluate.py --no-llm     # วัด retrieval + ดูคะแนน cosine เพื่อเลือก threshold
  python evaluate.py              # รวมการเรียก Groq: วัดการปฏิเสธ (ต้องมี GROQ_API_KEY)
"""
from __future__ import annotations

import argparse
import csv
import os
import tomllib
from pathlib import Path

from rag.llm import get_client
from rag.loader import load_chunks
from rag.pipeline import answer_question
from rag.prompt import is_not_found
from rag.retriever import DEFAULT_THRESHOLD, Retriever


def parse_sources(value: str) -> set[tuple[str, str]]:
    return {tuple(s.split(":", 1)) for s in value.split(";") if s.strip()}


def load_api_key() -> str | None:
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    path = Path(".streamlit/secrets.toml")
    if path.exists():
        return tomllib.loads(path.read_text(encoding="utf-8")).get("GROQ_API_KEY")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="test_questions.csv")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="พิมพ์คำตอบของ LLM")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    retriever = Retriever(load_chunks("data"))
    client = None
    if not args.no_llm:
        key = load_api_key()
        if not key:
            raise SystemExit("ไม่พบ GROQ_API_KEY (env หรือ .streamlit/secrets.toml) — ใช้ --no-llm ได้")
        client = get_client(key)

    ans_cos, unans_cos = [], []
    hit_ok = refusal_ok = false_refusal = 0
    for row in rows:
        answerable = row["answerable"].strip().lower() == "true"
        result = None
        search_query = row["question"]
        if client:
            result = answer_question(row["question"], [], retriever, client, k=args.k, threshold=args.threshold)
            search_query = result.search_query  # วัด retrieval ด้วยคำค้นที่ pipeline ใช้จริง (เช่น หลังแปลคำถามภาษาอังกฤษ)
        hits = retriever.search(search_query, k=args.k)
        best = max((h.cosine for h in hits), default=0.0)
        line = f"{row['id']:>3} {'A' if answerable else 'U'} cos={best:.3f}"
        if answerable:
            ans_cos.append(best)
            found = {(h.chunk.law, h.chunk.section) for h in hits}
            hit = bool(parse_sources(row["expected_source"]) & found)
            hit_ok += hit
            line += f" hit={'Y' if hit else 'N'}"
        else:
            unans_cos.append(best)
        if result:
            refused = result.answer is not None and is_not_found(result.answer)
            if answerable and refused:
                false_refusal += 1
            if not answerable and refused:
                refusal_ok += 1
            line += f" refused={'Y' if refused else 'N'}"
            if result.error:
                line += f" ERROR={result.error}"
        print(line, "|", row["question"])
        if result and args.verbose and result.answer:
            print("     ", result.answer.replace("\n", "\n      "))

    print("\n=== สรุป ===")
    print(f"Retrieval hit rate@{args.k}: {hit_ok}/{len(ans_cos)}")
    if client:
        print(f"ปฏิเสธถูกต้อง (ข้อที่ไม่มีคำตอบ): {refusal_ok}/{len(unans_cos)}")
        print(f"ปฏิเสธผิด (ข้อที่มีคำตอบแต่ตอบไม่พบ): {false_refusal}/{len(ans_cos)}")
    if ans_cos and unans_cos:
        lo, hi = min(ans_cos), max(unans_cos)
        print(f"cosine ต่ำสุดของข้อที่มีคำตอบ = {lo:.3f}, สูงสุดของข้อที่ไม่มีคำตอบ = {hi:.3f}")
        if lo > hi:
            print(f"threshold แนะนำ = {(lo + hi) / 2:.2f} (แยกสองกลุ่มได้ด้วย threshold)")
        else:
            print(f"threshold แนะนำ = {lo - 0.02:.2f} (สองกลุ่มซ้อนกัน — ให้ prompt เป็นด่านปฏิเสธหลัก)")


if __name__ == "__main__":
    main()
