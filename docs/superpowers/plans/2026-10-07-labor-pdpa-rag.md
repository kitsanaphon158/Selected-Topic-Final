# ผู้ช่วยกฎหมายสำหรับคนทำงาน (Labor + PDPA RAG) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** แชตบอต RAG บน Streamlit Community Cloud ที่ตอบคำถามสิทธิลูกจ้าง (พ.ร.บ.คุ้มครองแรงงาน) และ PDPA จากตัวบทเท่านั้น อ้างอิงระดับมาตรา และปฏิเสธเมื่อไม่มีข้อมูล

**Architecture:** ตัวบท `.txt` ใน `data/` (ไฟล์ละหมวด) → ตัด chunk ตามมาตรา → ค้นแบบ hybrid (FAISS บน e5-small + BM25 บน pythainlp) รวมด้วย RRF → threshold gate → Groq LLM พร้อม system prompt บังคับตอบจาก context; คำถามต่อเนื่องผ่าน query rewriting. ตรรกะ RAG ทั้งหมดอยู่ใน `rag/` ส่วน `app.py` เป็น UI ล้วน

**Tech Stack:** Python 3.11, streamlit, sentence-transformers (`intfloat/multilingual-e5-small`), faiss-cpu, rank-bm25, pythainlp, groq, pytest

**Spec:** `docs/superpowers/specs/2026-10-07-labor-pdpa-rag-design.md`

## Global Constraints

- ห้ามมี API key ใด ๆ ในไฟล์ที่ commit; เรียกผ่าน `st.secrets["GROQ_API_KEY"]` เท่านั้น; `.streamlit/secrets.toml` อยู่ใน `.gitignore`; **ผู้ช่วย AI ห้ามเขียน key จริงลงไฟล์ใด ๆ — ผู้ใช้สร้าง `secrets.toml` เอง**
- Embedding model: `intfloat/multilingual-e5-small` พร้อม prefix `query: ` / `passage: ` และ `normalize_embeddings=True`
- LLM: Groq `llama-3.3-70b-versatile`, temperature 0.1 (ตอบ), 0.0 (rewrite); ส่งประวัติ 6 ข้อความล่าสุด
- ข้อความปฏิเสธ: `ไม่พบข้อมูลในเอกสารที่มีอยู่` (ตรงตัว)
- chunk: `max_chars=800`, `overlap=100`; ค้นแต่ละวิธี top-8; RRF k=60; ค่าเริ่ม k=4 (slider 2–8)
- input สูงสุด 1,000 ตัวอักษร
- `data/` ≥10 ไฟล์ และรวม ≥15,000 ตัวอักษร; ห้ามพิมพ์ตัวบทจากความจำ
- `test_questions.csv` 12 ข้อ, unanswerable 2 ข้อ
- โมเดลและ index โหลดครั้งเดียวด้วย `@st.cache_resource`
- รันคำสั่ง Python บนเครื่องนี้หลัง `source scripts/dev_env.sh` เสมอ (Python ของเครื่องนี้ต่อ https ไม่ผ่านถ้าไม่ตั้ง cert bundle)
- ทุก commit message ลงท้ายด้วย `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. ผู้ใช้พิมพ์เลขไทย เช่น "มาตรา ๑๑๘" → ต้องได้ผลเหมือนพิมพ์ "มาตรา 118" (test ใน Task 3)
2. ผู้ใช้ติ๊กออกทั้งสองกฎหมาย → ไม่ค้น ไม่เรียก LLM ไม่ crash (test ใน Task 3 และ Task 5)
3. คำถามต่อเนื่องแต่ rewrite ล้มเหลวหรือคืนค่าว่าง → ค้นด้วยคำถามเดิม (test ใน Task 4 และ Task 5)
4. LLM ปฏิเสธด้วยประโยคที่มีคำนำหน้า เช่น "ขออภัย ไม่พบข้อมูล..." → ต้องนับเป็นปฏิเสธและซ่อนแหล่งอ้างอิง (test ใน Task 4 และ Task 5)
5. Groq ล่มตอนสร้างคำตอบ → ยังคืนมาตราที่ค้นได้ พร้อมข้อความ error ไม่ใช่ traceback (test ใน Task 5)

---

## File Structure

| ไฟล์ | หน้าที่ |
|---|---|
| `rag/loader.py` | `LAWS`, `Document`, `Chunk`, `clean_text`, `parse_document`, `load_documents`, `chunk_by_section`, `load_chunks` |
| `rag/retriever.py` | `Hit`, `Retriever.search`, `is_relevant`, `tokenize`, `DEFAULT_THRESHOLD` |
| `rag/prompt.py` | `NOT_FOUND`, `SYSTEM_PROMPT`, `REWRITE_PROMPT`, `source_label`, `build_context`, `is_not_found` |
| `rag/llm.py` | `MODEL`, `LLMError`, `get_client`, `rewrite_query`, `generate_answer` |
| `rag/pipeline.py` | `RagResult`, `validate_question`, `answer_question` — ประกอบทุกขั้นต่อหนึ่งคำถาม (เพิ่มจาก spec เพื่อให้ `app.py` ไม่มีตรรกะและทดสอบ flow ได้) |
| `app.py` | Streamlit UI |
| `scripts/split_law.py` | แตกข้อความดิบฉบับเต็มเป็นไฟล์ละหมวด (แทน `fetch_laws.py` ใน spec — การดาวน์โหลดใช้ `curl` บันทึกคำสั่งใน README) |
| `scripts/dev_env.sh` | (ไม่ commit) activate venv + cert bundle สำหรับเครื่องนี้ |
| `evaluate.py` | ประเมินด้วย `test_questions.csv` + แนะนำ threshold |
| `tests/helpers.py` | `make_chunk` ใช้ร่วมกันใน test |

ชื่อไฟล์ใน `data/` ใช้รูปแบบ `labor_ch11.txt`, `pdpa_ch03.txt`, `labor_ch00.txt` (ก่อนหมวด 1), `labor_ch99.txt` (บทเฉพาะกาล) — ชื่อหมวดอยู่ในหัวไฟล์ `CHAPTER:` แทนการเดา slug ภาษาอังกฤษ

---

### Task 1: Scaffold โปรเจกต์ + Loader (clean / parse / chunk)

**Files:**
- Create: `.gitignore`, `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, `.streamlit/secrets.toml.example`, `scripts/dev_env.sh`, `rag/__init__.py`, `rag/loader.py`, `tests/__init__.py`, `tests/helpers.py`, `tests/test_loader.py`

**Interfaces:**
- Consumes: —
- Produces:
  - `LAWS: dict[str, str]` = `{"labor": "พ.ร.บ.คุ้มครองแรงงาน", "pdpa": "พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล (PDPA)"}`
  - `Document(text, law, law_name, chapter, source_file, source_url)` (frozen dataclass)
  - `Chunk(id, text, law, law_name, chapter, section: str | None, source_file, source_url)` (frozen dataclass)
  - `clean_text(text: str) -> str`
  - `parse_document(raw: str, filename: str) -> Document`
  - `load_documents(data_dir: str | Path) -> list[Document]`
  - `chunk_by_section(doc: Document, max_chars: int = 800, overlap: int = 100) -> list[Chunk]`
  - `load_chunks(data_dir: str | Path) -> list[Chunk]`
  - `SECTION_RE` (compiled regex, group 1 = เลขมาตรา)
  - `tests.helpers.make_chunk(law: str, section: str | None, text: str, chapter: str = "หมวดทดสอบ") -> Chunk`

- [ ] **Step 1: สร้างไฟล์ config ของโปรเจกต์**

`.gitignore`:
```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.DS_Store
.claude/
.streamlit/secrets.toml
scripts/raw/
scripts/dev_env.sh
```

`requirements.txt`:
```text
--extra-index-url https://download.pytorch.org/whl/cpu
torch
streamlit>=1.38
sentence-transformers>=3.0
faiss-cpu>=1.8
rank-bm25>=0.2.2
pythainlp>=5.0
groq>=0.11
numpy
```

`requirements-dev.txt`:
```text
-r requirements.txt
pytest>=8
pypdf>=4
```

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

`.streamlit/secrets.toml.example`:
```toml
# คัดลอกไฟล์นี้เป็น .streamlit/secrets.toml แล้วใส่ key จริง (ไฟล์จริงอยู่ใน .gitignore)
GROQ_API_KEY = "ใส่-key-ของคุณที่นี่"
```

`scripts/dev_env.sh` (ไม่ commit — อยู่ใน .gitignore):
```bash
# ใช้: cd ~/Downloads/cloud/labor-pdpa-rag && source scripts/dev_env.sh
# เครื่องนี้ Python ต่อ https ไม่ผ่าน (cert ของพร็อกซีอยู่ใน Keychain) จึงส่งออก cert เป็น bundle ให้ pip / huggingface / groq ใช้
source .venv/bin/activate
if [ ! -s .venv/ca.pem ]; then
  security find-certificate -a -p \
    /System/Library/Keychains/SystemRootCertificates.keychain \
    /Library/Keychains/System.keychain \
    "$HOME/Library/Keychains/login.keychain-db" > .venv/ca.pem
fi
export SSL_CERT_FILE="$PWD/.venv/ca.pem"
export REQUESTS_CA_BUNDLE="$PWD/.venv/ca.pem"
export PIP_CERT="$PWD/.venv/ca.pem"
```

`rag/__init__.py` และ `tests/__init__.py`: ไฟล์ว่าง

- [ ] **Step 2: สร้าง venv และติดตั้ง dependency**

Run:
```bash
cd ~/Downloads/cloud/labor-pdpa-rag
python3.11 -m venv .venv
source scripts/dev_env.sh
pip install -r requirements-dev.txt
python -c "import faiss, rank_bm25, pythainlp, groq, sentence_transformers; print('ok')"
```
Expected: บรรทัดสุดท้ายพิมพ์ `ok`. ถ้า pip ขึ้น `CERTIFICATE_VERIFY_FAILED` ให้ตรวจว่า `.venv/ca.pem` ไม่ว่าง (`wc -c .venv/ca.pem`) แล้วรันใหม่

- [ ] **Step 3: เขียน test helper และ failing tests ของ loader**

`tests/helpers.py`:
```python
from rag.loader import Chunk

LAW_NAMES = {
    "labor": "พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541",
    "pdpa": "พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562",
}


def make_chunk(law: str, section: str | None, text: str, chapter: str = "หมวดทดสอบ") -> Chunk:
    return Chunk(
        id=f"{law}_test#{section or 'pre'}-0",
        text=text,
        law=law,
        law_name=LAW_NAMES[law],
        chapter=chapter,
        section=section,
        source_file=f"{law}_test.txt",
        source_url="https://example.go.th/law",
    )
```

`tests/test_loader.py`:
```python
import pytest

from rag.loader import (
    Document,
    chunk_by_section,
    clean_text,
    load_chunks,
    load_documents,
    parse_document,
)

SAMPLE = """LAW: พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541
CHAPTER: หมวด 11 ค่าชดเชย
SOURCE: https://example.go.th/labor.pdf
มาตรา ๑๑๘[๓๑] ให้นายจ้างจ่ายค่าชดเชยให้แก่ลูกจ้างซึ่งเลิกจ้าง
- ๑๒ -
มาตรา ๑๑๘/๑ การเกษียณอายุของลูกจ้าง
"""


def test_clean_text_converts_thai_digits_and_drops_noise():
    raw = (
        "มาตรา ๑๑๘[๓๑]   ให้นายจ้าง\n\n- ๑๒ -\nหน้า 3\n"
        "[๓๑] มาตรา ๑๑๘ แก้ไขโดย พ.ร.บ. ฉบับที่ 7\nบรรทัดสุดท้าย"
    )
    assert clean_text(raw) == "มาตรา 118 ให้นายจ้าง\nบรรทัดสุดท้าย"


def test_parse_document_reads_header_and_cleans_body():
    doc = parse_document(SAMPLE, "labor_ch11.txt")
    assert doc.law == "labor"
    assert doc.law_name == "พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541"
    assert doc.chapter == "หมวด 11 ค่าชดเชย"
    assert doc.source_url == "https://example.go.th/labor.pdf"
    assert doc.source_file == "labor_ch11.txt"
    assert doc.text.startswith("มาตรา 118 ให้นายจ้าง")
    assert "- 12 -" not in doc.text


def test_parse_document_rejects_missing_header():
    raw = "LAW: x\nCHAPTER: y\nมาตรา 1 ข้อความ"
    with pytest.raises(ValueError, match="SOURCE"):
        parse_document(raw, "labor_ch01.txt")


def test_parse_document_rejects_unknown_law_prefix():
    with pytest.raises(ValueError, match="tax"):
        parse_document(SAMPLE, "tax_ch01.txt")


def test_chunk_by_section_one_chunk_per_section():
    chunks = chunk_by_section(parse_document(SAMPLE, "labor_ch11.txt"))
    assert [c.section for c in chunks] == ["118", "118/1"]
    assert chunks[0].id == "labor_ch11#118-0"
    assert chunks[1].text.startswith("มาตรา 118/1")
    assert all(c.law == "labor" and c.chapter == "หมวด 11 ค่าชดเชย" for c in chunks)


def test_long_section_is_split_with_continuation_prefix():
    doc = Document(
        text="มาตรา 5 " + "ก" * 1500,
        law="pdpa",
        law_name="พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562",
        chapter="หมวด 1",
        source_file="pdpa_ch01.txt",
        source_url="u",
    )
    chunks = chunk_by_section(doc, max_chars=800, overlap=100)
    assert len(chunks) == 3
    assert [c.id for c in chunks] == ["pdpa_ch01#5-0", "pdpa_ch01#5-1", "pdpa_ch01#5-2"]
    assert all(c.section == "5" for c in chunks)
    assert chunks[1].text.startswith("(ต่อ) มาตรา 5\n")
    assert all(len(c.text) <= 800 + len("(ต่อ) มาตรา 5\n") for c in chunks)


def test_tiny_text_before_first_section_is_dropped():
    doc = Document(
        text="ส่วนที่ 1\nมาตรา 7 ข้อความของมาตรา",
        law="labor", law_name="L", chapter="หมวด 1", source_file="labor_ch01.txt", source_url="u",
    )
    assert [c.section for c in chunk_by_section(doc)] == ["7"]


def test_long_text_before_first_section_is_kept():
    preamble = "พระราชบัญญัตินี้มีบทบัญญัติบางประการเกี่ยวกับการจำกัดสิทธิและเสรีภาพของบุคคล " * 2
    doc = Document(
        text=f"{preamble}\nมาตรา 1 พระราชบัญญัตินี้เรียกว่า",
        law="labor", law_name="L", chapter="ก่อนหมวด 1", source_file="labor_ch00.txt", source_url="u",
    )
    chunks = chunk_by_section(doc)
    assert [c.section for c in chunks] == [None, "1"]
    assert chunks[0].id == "labor_ch00#pre-0"


def test_load_documents_and_chunks_read_sorted_txt(tmp_path):
    (tmp_path / "pdpa_ch01.txt").write_text(
        "LAW: P\nCHAPTER: หมวด 1\nSOURCE: u\nมาตรา 19 ความยินยอม", encoding="utf-8"
    )
    (tmp_path / "labor_ch11.txt").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "notes.md").write_text("ไม่ใช่ข้อมูล", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert [d.source_file for d in docs] == ["labor_ch11.txt", "pdpa_ch01.txt"]
    assert [c.section for c in load_chunks(tmp_path)] == ["118", "118/1", "19"]
```

- [ ] **Step 4: รัน test ให้ fail**

Run: `source scripts/dev_env.sh && pytest tests/test_loader.py -v`
Expected: FAIL ด้วย `ModuleNotFoundError: No module named 'rag.loader'`

- [ ] **Step 5: เขียน `rag/loader.py`**

```python
"""โหลดตัวบทกฎหมายจาก data/ ทำความสะอาด และตัดเป็น chunk ละหนึ่งมาตรา"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pythainlp.util import normalize, thai_digit_to_arabic_digit

# prefix ชื่อไฟล์ -> ชื่อย่อที่แสดงใน UI
LAWS = {
    "labor": "พ.ร.บ.คุ้มครองแรงงาน",
    "pdpa": "พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล (PDPA)",
}
HEADER_KEYS = ("LAW", "CHAPTER", "SOURCE")
MIN_PREAMBLE_CHARS = 50

SECTION_RE = re.compile(
    r"^มาตรา\s+(\d+(?:/\d+)?(?:\s+(?:ทวิ|ตรี|จัตวา|เบญจ))?)", re.MULTILINE
)
_FOOTNOTE_LINE_RE = re.compile(r"^\s*\[\d+\].*$", re.MULTILINE)
_FOOTNOTE_MARK_RE = re.compile(r"\[\d+\]")
_PAGE_NUM_RE = re.compile(r"^(?:-\s*\d+\s*-|\d+|หน้า\s*\d+(?:\s*/\s*\d+)?)$")
_SPACES_RE = re.compile(r"[ \t ​]+")


@dataclass(frozen=True)
class Document:
    text: str
    law: str  # "labor" | "pdpa"
    law_name: str  # ชื่อเต็มจากหัวไฟล์ LAW:
    chapter: str
    source_file: str
    source_url: str


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    law: str
    law_name: str
    chapter: str
    section: str | None
    source_file: str
    source_url: str


def clean_text(text: str) -> str:
    """เลขไทย→อารบิก, normalize, ลบเชิงอรรถ/เลขหน้า/บรรทัดว่าง/ช่องว่างซ้ำ"""
    text = thai_digit_to_arabic_digit(text.replace("\r\n", "\n").replace("\r", "\n"))
    text = normalize(text)
    text = _FOOTNOTE_LINE_RE.sub("", text)
    text = _FOOTNOTE_MARK_RE.sub("", text)
    lines = []
    for line in text.split("\n"):
        line = _SPACES_RE.sub(" ", line).strip()
        if line and not _PAGE_NUM_RE.match(line):
            lines.append(line)
    return "\n".join(lines)


def parse_document(raw: str, filename: str) -> Document:
    lines = raw.replace("\r\n", "\n").split("\n")
    header: dict[str, str] = {}
    i = 0
    while i < len(lines):
        key, sep, value = lines[i].partition(":")
        if not sep or key.strip() not in HEADER_KEYS:
            break
        header[key.strip()] = value.strip()
        i += 1
    missing = [k for k in HEADER_KEYS if k not in header]
    if missing:
        raise ValueError(f"{filename}: หัวไฟล์ขาด {', '.join(missing)}")
    law = filename.split("_", 1)[0]
    if law not in LAWS:
        raise ValueError(f"{filename}: prefix '{law}' ไม่อยู่ใน {sorted(LAWS)}")
    return Document(
        text=clean_text("\n".join(lines[i:])),
        law=law,
        law_name=header["LAW"],
        chapter=header["CHAPTER"],
        source_file=filename,
        source_url=header["SOURCE"],
    )


def load_documents(data_dir: str | Path) -> list[Document]:
    paths = sorted(Path(data_dir).glob("*.txt"))
    return [parse_document(p.read_text(encoding="utf-8"), p.name) for p in paths]


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    windows, start = [], 0
    while True:
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if cut > start + max_chars // 2:
                end = cut
        windows.append(text[start:end].strip())
        if end >= len(text):
            return windows
        start = max(end - overlap, start + 1)


def chunk_by_section(doc: Document, max_chars: int = 800, overlap: int = 100) -> list[Chunk]:
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap ต้องอยู่ระหว่าง 0 ถึง max_chars-1")
    matches = list(SECTION_RE.finditer(doc.text))
    pieces: list[tuple[str | None, str]] = []
    first = matches[0].start() if matches else len(doc.text)
    preamble = doc.text[:first].strip()
    if len(preamble) >= MIN_PREAMBLE_CHARS:
        pieces.append((None, preamble))
    for idx, m in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(doc.text)
        pieces.append((_SPACES_RE.sub(" ", m.group(1)), doc.text[m.start():end].strip()))

    stem = Path(doc.source_file).stem
    chunks = []
    for section, body in pieces:
        for part, window in enumerate(_split_long(body, max_chars, overlap)):
            text = f"(ต่อ) มาตรา {section}\n{window}" if part and section else window
            chunks.append(
                Chunk(
                    id=f"{stem}#{section or 'pre'}-{part}",
                    text=text,
                    law=doc.law,
                    law_name=doc.law_name,
                    chapter=doc.chapter,
                    section=section,
                    source_file=doc.source_file,
                    source_url=doc.source_url,
                )
            )
    return chunks


def load_chunks(data_dir: str | Path) -> list[Chunk]:
    return [c for doc in load_documents(data_dir) for c in chunk_by_section(doc)]
```

- [ ] **Step 6: รัน test ให้ผ่าน**

Run: `source scripts/dev_env.sh && pytest tests/test_loader.py -v`
Expected: PASS ทั้ง 9 test. ถ้า `test_clean_text...` fail เพราะ `normalize` ของ pythainlp เปลี่ยนข้อความ ให้พิมพ์ค่า `repr(clean_text(raw))` ดูว่าต่างตรงไหนก่อนแก้ — ห้ามแก้ test ให้ตรงกับผลที่ผิด

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt requirements-dev.txt pyproject.toml .streamlit/secrets.toml.example rag tests
git status --short   # ต้องไม่เห็น scripts/dev_env.sh หรือ .streamlit/secrets.toml
git commit -m "feat: scaffold project and section-aware loader

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: คลังตัวบทใน `data/`

**Files:**
- Create: `scripts/split_law.py`, `tests/test_split_law.py`, `tests/test_data.py`, `data/*.txt` (สร้างโดยสคริปต์)
- Local only (gitignored): `scripts/raw/labor.pdf|txt`, `scripts/raw/pdpa.pdf|txt`

**Interfaces:**
- Consumes: `clean_text`, `LAWS`, `SECTION_RE`, `load_documents`, `load_chunks` จาก Task 1
- Produces: `data/labor_chNN.txt`, `data/pdpa_chNN.txt` ที่ `parse_document` อ่านได้; `split_chapters(text: str) -> list[tuple[str, str, str]]` (รหัสหมวด, ชื่อหมวด, เนื้อหา)

- [ ] **Step 1: เขียน failing test ของ `split_chapters`**

`tests/test_split_law.py`:
```python
from scripts.split_law import chapter_filename, split_chapters

RAW = """พระราชบัญญัติ
คุ้มครองแรงงาน
มาตรา 1 พระราชบัญญัตินี้เรียกว่า "พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541"
หมวด 1
บททั่วไป
มาตรา 5 ในพระราชบัญญัตินี้
หมวด 11 ค่าชดเชย
มาตรา 118 ให้นายจ้างจ่ายค่าชดเชย
บทเฉพาะกาล
มาตรา 167 บรรดาคำร้อง
ผู้รับสนองพระบรมราชโองการ
นายกรัฐมนตรี
มาตรา 1 พระราชบัญญัตินี้เรียกว่า "พระราชบัญญัติคุ้มครองแรงงาน (ฉบับที่ 2)"
"""


def test_split_chapters_by_heading():
    chapters = split_chapters(RAW)
    assert [(code, title) for code, title, _ in chapters] == [
        ("00", "ก่อนหมวด 1"),
        ("01", "หมวด 1 บททั่วไป"),
        ("11", "หมวด 11 ค่าชดเชย"),
        ("99", "บทเฉพาะกาล"),
    ]
    assert chapters[2][2] == "มาตรา 118 ให้นายจ้างจ่ายค่าชดเชย"


def test_split_chapters_stops_at_signature_block():
    body = split_chapters(RAW)[-1][2]
    assert "ผู้รับสนอง" not in body
    assert "ฉบับที่ 2" not in body


def test_split_chapters_handles_sub_numbered_chapter():
    chapters = split_chapters("หมวด 6/1 กองทุน\nมาตรา 80 ข้อความ")
    assert chapters[0][:2] == ("06-1", "หมวด 6/1 กองทุน")


def test_chapter_filename():
    assert chapter_filename("labor", "11") == "labor_ch11.txt"
```

Run: `source scripts/dev_env.sh && pytest tests/test_split_law.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'scripts'`) — สร้าง `scripts/__init__.py` ว่างพร้อมสคริปต์ใน Step 2

- [ ] **Step 2: เขียน `scripts/split_law.py` (และ `scripts/__init__.py` ว่าง)**

```python
"""แตกข้อความดิบของ พ.ร.บ. ฉบับเต็มเป็นไฟล์ละหมวดใน data/

ใช้:
  python scripts/split_law.py scripts/raw/labor.txt --law labor \
      --name "พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541" --source "<URL ต้นทาง>"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.loader import LAWS, SECTION_RE, clean_text  # noqa: E402

CHAPTER_RE = re.compile(r"^หมวด\s+(\d+)(?:/(\d+))?\s*(.*)$")
TRANSITIONAL_RE = re.compile(r"^บทเฉพาะกาล\s*$")
# ท้ายตัวบท: ลายมือชื่อผู้รับสนอง / หมายเหตุ / พ.ร.บ.ฉบับแก้ไขที่แนบท้าย — ไม่นำเข้าคลัง
END_RE = re.compile(r"^(ผู้รับสนอง(พระบรม)?ราชโองการ|หมายเหตุ\s*[:\-]?\s*เหตุผล)")


def split_chapters(text: str) -> list[tuple[str, str, str]]:
    """คืน [(รหัสหมวด, ชื่อหมวด, เนื้อหา)] — '00' = ก่อนหมวด 1, '99' = บทเฉพาะกาล"""
    lines = clean_text(text).split("\n")
    chapters: list[tuple[str, str, list[str]]] = []
    code, title, body = "00", "ก่อนหมวด 1", []
    i = 0
    while i < len(lines):
        line = lines[i]
        if END_RE.match(line):
            break
        m = CHAPTER_RE.match(line)
        if m or TRANSITIONAL_RE.match(line):
            chapters.append((code, title, body))
            if m:
                num, sub, name = m.group(1), m.group(2), m.group(3).strip()
                if not name and i + 1 < len(lines):
                    i += 1
                    name = lines[i].strip()
                code = f"{int(num):02d}" + (f"-{sub}" if sub else "")
                title = f"หมวด {num}{'/' + sub if sub else ''} {name}".strip()
            else:
                code, title = "99", "บทเฉพาะกาล"
            body = []
        else:
            body.append(line)
        i += 1
    chapters.append((code, title, body))
    result = []
    for code, title, body_lines in chapters:
        content = "\n".join(body_lines).strip()
        if content and SECTION_RE.search(content):
            result.append((code, title, content))
    return result


def chapter_filename(law: str, code: str) -> str:
    return f"{law}_ch{code}.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("--law", required=True, choices=sorted(LAWS))
    ap.add_argument("--name", required=True, help="ชื่อเต็ม เช่น พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541")
    ap.add_argument("--source", required=True, help="URL ต้นทางของตัวบท")
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    args.out.mkdir(exist_ok=True)
    chapters = split_chapters(args.raw.read_text(encoding="utf-8"))
    for code, title, content in chapters:
        path = args.out / chapter_filename(args.law, code)
        path.write_text(
            f"LAW: {args.name}\nCHAPTER: {title}\nSOURCE: {args.source}\n{content}\n",
            encoding="utf-8",
        )
        print(f"{path.name:<20} {len(content):>7,} ตัวอักษร  {title}")
    print(f"รวม {len(chapters)} ไฟล์")


if __name__ == "__main__":
    main()
```

Run: `source scripts/dev_env.sh && pytest tests/test_split_law.py -v`
Expected: PASS ทั้ง 4 test

- [ ] **Step 3: Commit สคริปต์**

```bash
git add scripts/__init__.py scripts/split_law.py tests/test_split_law.py
git commit -m "feat: add script to split law text into chapter files

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: เขียน failing test คุณภาพคลังข้อมูล**

`tests/test_data.py`:
```python
import re
from collections import Counter
from pathlib import Path

from rag.loader import load_chunks, load_documents

DATA = Path(__file__).resolve().parents[1] / "data"
KEY_SECTIONS = {
    "labor": {"23", "30", "32", "41", "57", "61", "118"},
    "pdpa": {"19", "26", "30", "33", "37"},
}
SUFFIX = {"ทวิ": 1, "ตรี": 2, "จัตวา": 3, "เบญจ": 4}


def section_key(section: str) -> tuple[int, int]:
    m = re.match(r"(\d+)(?:/(\d+))?(?:\s+(\S+))?", section)
    return int(m.group(1)), int(m.group(2) or SUFFIX.get(m.group(3) or "", 0))


def sections_by_file() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in load_chunks(DATA):
        if c.section and c.id.endswith("-0"):
            out.setdefault(c.source_file, []).append(c.section)
    return out


def test_corpus_size_meets_assignment():
    docs = load_documents(DATA)
    assert len(docs) >= 10
    assert sum(len(d.text) for d in docs) >= 15_000


def test_both_laws_and_key_sections_present():
    found = {(c.law, c.section) for c in load_chunks(DATA)}
    for law, sections in KEY_SECTIONS.items():
        missing = {s for s in sections if (law, s) not in found}
        assert not missing, f"{law} ขาดมาตรา {sorted(missing)}"


def test_sections_increase_within_each_file():
    # ถ้า fail = บรรทัดที่ขึ้นต้นด้วย "มาตรา N" กลางย่อหน้าถูกตัดเป็นมาตราใหม่ ต้องแก้ข้อความดิบ
    for name, sections in sections_by_file().items():
        keys = [section_key(s) for s in sections]
        assert keys == sorted(keys) and len(set(keys)) == len(keys), f"{name}: {sections}"


def test_no_duplicate_sections_across_files():
    counts = Counter(
        (c.law, c.section) for c in load_chunks(DATA) if c.section and c.id.endswith("-0")
    )
    dupes = [k for k, n in counts.items() if n > 1]
    assert not dupes, dupes


def test_labor_text_is_consolidated_with_2019_amendment():
    text = {(c.law, c.section): c.text for c in load_chunks(DATA) if c.id.endswith("-0")}
    labor_118 = "".join(
        c.text for c in load_chunks(DATA) if c.law == "labor" and c.section == "118"
    )
    assert "สี่ร้อยวัน" in labor_118, "ม.118 ยังไม่ใช่ฉบับแก้ไข 2562 (ค่าชดเชย 400 วัน)"
    assert "เก้าสิบแปดวัน" in text[("labor", "41")], "ม.41 ยังไม่ใช่ฉบับแก้ไข 2562 (ลาคลอด 98 วัน)"
```

Run: `source scripts/dev_env.sh && pytest tests/test_data.py -v`
Expected: FAIL (`assert 0 >= 10` — ยังไม่มีไฟล์ใน `data/`)

- [ ] **Step 5: หาและดาวน์โหลดตัวบทฉบับรวมแก้ไข**

เป้าหมาย: ไฟล์ข้อความ (ไม่ใช่ภาพสแกน) ของ
1. พ.ร.บ.คุ้มครองแรงงาน พ.ศ. 2541 **ฉบับรวมแก้ไขถึงฉบับที่ 7 (2562) หรือใหม่กว่า**
2. พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562

ลำดับแหล่งที่จะลอง (ใช้ in-app browser หรือ `curl -sIL` ตรวจ URL ก่อน):
- สำนักงานคณะกรรมการกฤษฎีกา `https://www.krisdika.go.th` (ฐานข้อมูลกฎหมาย ฉบับ Update)
- กรมสวัสดิการและคุ้มครองแรงงาน `https://www.labour.go.th` / `https://lpa.labour.go.th` (พ.ร.บ.คุ้มครองแรงงาน ฉบับรวม)
- สำนักงานคณะกรรมการคุ้มครองข้อมูลส่วนบุคคล `https://www.pdpc.or.th` (ตัวบท PDPA)
- ราชกิจจานุเบกษา `https://ratchakitcha.soc.go.th`

```bash
mkdir -p scripts/raw
curl -sL -o scripts/raw/labor.pdf "<URL ที่ตรวจแล้ว>"
curl -sL -o scripts/raw/pdpa.pdf "<URL ที่ตรวจแล้ว>"
file scripts/raw/*.pdf   # ต้องเป็น "PDF document" ไม่ใช่ HTML
```
(ถ้าแหล่งเป็น HTML ให้บันทึกเป็น `.html` แล้วดึงข้อความด้วย `python -c "import html, re, sys; ..."` แบบเดียวกับ Step 6 แต่ตัดแท็ก)

บันทึก URL ที่ใช้จริงทั้งสองตัว — จะใช้ใน `--source` และ README

- [ ] **Step 6: ดึงข้อความจาก PDF และตรวจคุณภาพภาษาไทย**

```bash
source scripts/dev_env.sh
for law in labor pdpa; do
python - "$law" <<'EOF'
import sys
from pypdf import PdfReader
law = sys.argv[1]
reader = PdfReader(f"scripts/raw/{law}.pdf")
text = "\n".join(page.extract_text() or "" for page in reader.pages)
open(f"scripts/raw/{law}.txt", "w", encoding="utf-8").write(text)
print(law, len(reader.pages), "หน้า", len(text), "ตัวอักษร")
EOF
done
head -40 scripts/raw/labor.txt
grep -c "มาตรา" scripts/raw/labor.txt scripts/raw/pdpa.txt
grep -c "ค่าชดเชย" scripts/raw/labor.txt
grep -c "ความยินยอม" scripts/raw/pdpa.txt
```
Expected: แต่ละไฟล์หลายหมื่นตัวอักษร, `มาตรา` นับได้เป็นร้อย, คำค้นทั้งสองนับได้ > 0, และ 40 บรรทัดแรกอ่านเป็นภาษาไทยปกติ (สระ/วรรณยุกต์ไม่ลอยแยกหรือสลับที่)

**หยุดและแจ้งผู้ใช้** ถ้า: ได้ข้อความ < 1,000 ตัวอักษร (PDF สแกน), ภาษาไทยเพี้ยนจนอ่านไม่ออก, หรือหาแหล่งที่เป็นฉบับรวมแก้ไขไม่ได้ — **ห้ามพิมพ์ตัวบทเองจากความจำ**

- [ ] **Step 7: แตกเป็นไฟล์ละหมวด**

```bash
source scripts/dev_env.sh
python scripts/split_law.py scripts/raw/labor.txt --law labor \
  --name "พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541" --source "<URL labor จาก Step 5>"
python scripts/split_law.py scripts/raw/pdpa.txt --law pdpa \
  --name "พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562" --source "<URL pdpa จาก Step 5>"
ls data | wc -l
```
Expected: labor ประมาณ 15–18 ไฟล์, pdpa ประมาณ 8–9 ไฟล์ และชื่อหมวดที่พิมพ์ออกมาตรงกับสารบัญของ พ.ร.บ.

- [ ] **Step 8: รัน data test และแก้ข้อความดิบจนผ่าน**

Run: `source scripts/dev_env.sh && pytest tests/test_data.py -v`
Expected: PASS ทั้ง 5 test

ถ้า fail:
- `test_sections_increase_within_each_file` → เปิด `scripts/raw/<law>.txt` หาบรรทัดที่ขึ้นต้นด้วย `มาตรา N` ซึ่งเป็นการอ้างถึงมาตราอื่นกลางย่อหน้า แล้วต่อเข้ากับบรรทัดก่อนหน้า จากนั้นรัน Step 7 ใหม่ (ลบ `data/<law>_*.txt` ก่อน)
- `test_labor_text_is_consolidated...` → ได้ฉบับเก่า กลับไป Step 5 หาฉบับรวมแก้ไข
- ขาดมาตราสำคัญ → ตรวจว่า `END_RE` ตัดเนื้อหาเร็วเกินไปหรือ PDF ขาดหน้า

- [ ] **Step 9: Commit คลังข้อมูล**

```bash
git add data tests/test_data.py
git commit -m "data: add consolidated Labor Protection Act and PDPA texts split by chapter

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Retriever (FAISS + BM25 + RRF)

**Files:**
- Create: `rag/retriever.py`, `tests/test_retriever.py`

**Interfaces:**
- Consumes: `Chunk` (Task 1), `tests.helpers.make_chunk`
- Produces:
  - `DEFAULT_MODEL = "intfloat/multilingual-e5-small"`, `DEFAULT_THRESHOLD: float`
  - `Hit(chunk: Chunk, rrf_score: float, cosine: float)` (frozen dataclass)
  - `Retriever(chunks: list[Chunk], model_name: str = DEFAULT_MODEL)`
  - `Retriever.search(query: str, k: int = 4, laws: set[str] | None = None) -> list[Hit]` — `laws=None` = ทุกฉบับ, `set()` = ไม่ค้นเลย
  - `is_relevant(hits: list[Hit], threshold: float = DEFAULT_THRESHOLD) -> bool`
  - `tokenize(text: str) -> list[str]`

- [ ] **Step 1: เขียน failing tests**

`tests/test_retriever.py`:
```python
import pytest

from rag.retriever import Hit, Retriever, is_relevant, tokenize
from tests.helpers import make_chunk

CHUNKS = [
    make_chunk("labor", "118", "มาตรา 118 ให้นายจ้างจ่ายค่าชดเชยให้แก่ลูกจ้างซึ่งเลิกจ้าง ลูกจ้างซึ่งทำงานติดต่อกันครบสามปีแต่ไม่ครบหกปี ให้จ่ายไม่น้อยกว่าค่าจ้างอัตราสุดท้ายหนึ่งร้อยแปดสิบวัน"),
    make_chunk("labor", "32", "มาตรา 32 ให้ลูกจ้างมีสิทธิลาป่วยได้เท่าที่ป่วยจริง การลาป่วยตั้งแต่สามวันทำงานขึ้นไป นายจ้างอาจให้ลูกจ้างแสดงใบรับรองของแพทย์"),
    make_chunk("labor", "23", "มาตรา 23 ให้นายจ้างประกาศเวลาทำงานปกติ ซึ่งต้องไม่เกินวันละแปดชั่วโมง และสัปดาห์หนึ่งต้องไม่เกินสี่สิบแปดชั่วโมง"),
    make_chunk("pdpa", "19", "มาตรา 19 ผู้ควบคุมข้อมูลส่วนบุคคลจะกระทำการเก็บรวบรวม ใช้ หรือเปิดเผยข้อมูลส่วนบุคคลไม่ได้ หากเจ้าของข้อมูลส่วนบุคคลไม่ได้ให้ความยินยอมไว้ก่อนหรือในขณะนั้น"),
    make_chunk("pdpa", "30", "มาตรา 30 เจ้าของข้อมูลส่วนบุคคลมีสิทธิขอเข้าถึงและขอรับสำเนาข้อมูลส่วนบุคคลที่เกี่ยวกับตน"),
]


@pytest.fixture(scope="module")
def retriever():
    return Retriever(CHUNKS)


def test_tokenize_normalizes_digits_and_drops_spaces():
    tokens = tokenize("มาตรา ๑๑๘  ค่าชดเชย")
    assert "118" in tokens
    assert all(t.strip() for t in tokens)


def test_exact_section_number_ranks_first(retriever):
    hits = retriever.search("มาตรา 118 ว่าด้วยอะไร", k=3)
    assert (hits[0].chunk.law, hits[0].chunk.section) == ("labor", "118")


def test_thai_digit_query_matches_arabic_section(retriever):
    hits = retriever.search("มาตรา ๑๑๘ ว่าด้วยอะไร", k=3)
    assert hits[0].chunk.section == "118"


def test_semantic_query_finds_sick_leave(retriever):
    hits = retriever.search("ป่วยหยุดงานได้ไหม ต้องมีใบแพทย์หรือเปล่า", k=2)
    assert "32" in [h.chunk.section for h in hits]


def test_law_filter_restricts_results(retriever):
    hits = retriever.search("ข้อมูลส่วนบุคคล ความยินยอม", k=5, laws={"labor"})
    assert hits and all(h.chunk.law == "labor" for h in hits)


def test_empty_law_filter_returns_nothing(retriever):
    assert retriever.search("ค่าชดเชย", laws=set()) == []


def test_blank_query_returns_nothing(retriever):
    assert retriever.search("   ") == []


def test_k_limits_result_count(retriever):
    assert len(retriever.search("ค่าชดเชย", k=2)) == 2


def test_hits_sorted_by_rrf_and_carry_cosine(retriever):
    hits = retriever.search("ลูกจ้างถูกเลิกจ้าง", k=4)
    assert [h.rrf_score for h in hits] == sorted((h.rrf_score for h in hits), reverse=True)
    assert all(-1.0 <= h.cosine <= 1.0 for h in hits)


def test_off_topic_query_scores_lower_than_on_topic(retriever):
    on = max(h.cosine for h in retriever.search("ถูกเลิกจ้างได้ค่าชดเชยเท่าไร"))
    off = max(h.cosine for h in retriever.search("สูตรทำต้มยำกุ้งน้ำข้น"))
    assert off < on


def test_is_relevant_uses_best_cosine():
    c = CHUNKS[0]
    assert is_relevant([Hit(c, 0.03, 0.70), Hit(c, 0.02, 0.85)], 0.80)
    assert not is_relevant([Hit(c, 0.03, 0.70)], 0.80)
    assert not is_relevant([], 0.80)
```

- [ ] **Step 2: รัน test ให้ fail**

Run: `source scripts/dev_env.sh && pytest tests/test_retriever.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.retriever'`)

- [ ] **Step 3: เขียน `rag/retriever.py`**

```python
"""ค้นมาตราที่เกี่ยวข้องด้วย FAISS (ความหมาย) + BM25 (คำตรงตัว) แล้วรวมอันดับด้วย RRF"""
from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from pythainlp.tokenize import word_tokenize
from pythainlp.util import thai_digit_to_arabic_digit
from rank_bm25 import BM25Okapi

from rag.loader import Chunk

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
# ปรับจากผล evaluate.py (Task 7) — ค่าสุดท้ายบันทึกใน README
DEFAULT_THRESHOLD = 0.80
CANDIDATES = 8
RRF_K = 60


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    rrf_score: float
    cosine: float


def tokenize(text: str) -> list[str]:
    text = thai_digit_to_arabic_digit(text).lower()
    return [t for t in word_tokenize(text, engine="newmm", keep_whitespace=False) if t.strip()]


def _index_text(chunk: Chunk) -> str:
    return f"{chunk.law_name} {chunk.chapter}\n{chunk.text}"


class Retriever:
    def __init__(self, chunks: list[Chunk], model_name: str = DEFAULT_MODEL):
        if not chunks:
            raise ValueError("ไม่มี chunk ให้สร้าง index — ตรวจโฟลเดอร์ data/")
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model = SentenceTransformer(model_name, device="cpu")
        vectors = self._encode([f"passage: {_index_text(c)}" for c in chunks])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.bm25 = BM25Okapi([tokenize(_index_text(c)) for c in chunks])

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype="float32")

    def search(self, query: str, k: int = 4, laws: set[str] | None = None) -> list[Hit]:
        query = thai_digit_to_arabic_digit(query).strip()
        allowed = [i for i, c in enumerate(self.chunks) if laws is None or c.law in laws]
        if not query or not allowed or k <= 0:
            return []

        # FAISS: คลังเล็ก (ไม่กี่ร้อย chunk) จึงค้นทั้งหมดแล้วกรองตามกฎหมายที่เลือก
        scores, ids = self.index.search(self._encode([f"query: {query}"]), self.index.ntotal)
        cosine = np.zeros(len(self.chunks), dtype="float32")
        cosine[ids[0]] = scores[0]
        allowed_set = set(allowed)
        vector_rank = [int(i) for i in ids[0] if int(i) in allowed_set][:CANDIDATES]

        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_rank = [
            i for i in sorted(allowed, key=lambda i: bm25_scores[i], reverse=True)
            if bm25_scores[i] > 0
        ][:CANDIDATES]

        fused: dict[int, float] = {}
        for ranking in (vector_rank, bm25_rank):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(fused, key=fused.__getitem__, reverse=True)[:k]
        return [Hit(self.chunks[i], fused[i], float(cosine[i])) for i in top]


def is_relevant(hits: list[Hit], threshold: float = DEFAULT_THRESHOLD) -> bool:
    return bool(hits) and max(h.cosine for h in hits) >= threshold
```

- [ ] **Step 4: รัน test ให้ผ่าน**

Run: `source scripts/dev_env.sh && pytest tests/test_retriever.py -v`
Expected: PASS ทั้ง 11 test (ครั้งแรกจะดาวน์โหลดโมเดล ~470 MB). ถ้าดาวน์โหลดขึ้น SSL error ให้ตรวจว่า source `dev_env.sh` แล้ว

- [ ] **Step 5: Commit**

```bash
git add rag/retriever.py tests/test_retriever.py
git commit -m "feat: hybrid FAISS + BM25 retriever with RRF and law filter

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Prompt + LLM wrapper

**Files:**
- Create: `rag/prompt.py`, `rag/llm.py`, `tests/test_prompt.py`, `tests/test_llm.py`

**Interfaces:**
- Consumes: `Chunk` (Task 1), `Hit` (Task 3)
- Produces:
  - `NOT_FOUND = "ไม่พบข้อมูลในเอกสารที่มีอยู่"`, `DISCLAIMER`, `SYSTEM_PROMPT`, `REWRITE_PROMPT`
  - `source_label(chunk: Chunk) -> str` เช่น `"พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541 — หมวด 11 ค่าชดเชย — มาตรา 118"`
  - `build_context(hits: list[Hit]) -> str`
  - `is_not_found(answer: str) -> bool`
  - `MODEL`, `HISTORY_MESSAGES = 6`, `class LLMError(RuntimeError)`
  - `get_client(api_key: str) -> groq.Groq`
  - `rewrite_query(client, history: list[dict], question: str) -> str` (คืน `question` ถ้า LLM คืนค่าว่าง; โยน `LLMError` ถ้า API ล้มเหลว)
  - `generate_answer(client, history: list[dict], question: str, context: str) -> str` (โยน `LLMError`)
  - `history` = list ของ dict ที่มีอย่างน้อย `role` ("user"/"assistant") และ `content`; key อื่นถูกตัดทิ้งก่อนส่ง

- [ ] **Step 1: เขียน failing tests**

`tests/test_prompt.py`:
```python
from rag.prompt import NOT_FOUND, SYSTEM_PROMPT, build_context, is_not_found, source_label
from rag.retriever import Hit
from tests.helpers import make_chunk


def test_source_label_has_law_chapter_and_section():
    chunk = make_chunk("labor", "118", "มาตรา 118 ...", chapter="หมวด 11 ค่าชดเชย")
    assert source_label(chunk) == "พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541 — หมวด 11 ค่าชดเชย — มาตรา 118"


def test_source_label_without_section():
    chunk = make_chunk("labor", None, "คำปรารภ", chapter="ก่อนหมวด 1")
    assert source_label(chunk) == "พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541 — ก่อนหมวด 1"


def test_build_context_numbers_each_hit():
    hits = [
        Hit(make_chunk("labor", "118", "มาตรา 118 ค่าชดเชย"), 0.03, 0.9),
        Hit(make_chunk("pdpa", "19", "มาตรา 19 ความยินยอม"), 0.02, 0.8),
    ]
    ctx = build_context(hits)
    assert ctx.startswith("<context>\n[1] พระราชบัญญัติคุ้มครองแรงงาน")
    assert "\n[2] พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562" in ctx
    assert "มาตรา 19 ความยินยอม" in ctx
    assert ctx.endswith("</context>")


def test_system_prompt_states_rules():
    assert NOT_FOUND in SYSTEM_PROMPT
    assert "<context>" in SYSTEM_PROMPT
    assert "[1]" in SYSTEM_PROMPT


def test_is_not_found_variants():
    assert is_not_found(NOT_FOUND)
    assert is_not_found("  ไม่พบข้อมูลในเอกสารที่มีอยู่\n\nหมายเหตุ ...")
    assert is_not_found("ขออภัย ไม่พบข้อมูลในเอกสารที่มีอยู่")
    assert not is_not_found(
        "ตามมาตรา 118 ลูกจ้างที่ทำงานครบสามปีแต่ไม่ครบหกปีมีสิทธิได้รับค่าชดเชย 180 วัน [1] "
        "หากไม่พบข้อมูลเพิ่มเติมให้สอบถามพนักงานตรวจแรงงาน"
    )
```

`tests/test_llm.py`:
```python
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
```

- [ ] **Step 2: รัน test ให้ fail**

Run: `source scripts/dev_env.sh && pytest tests/test_prompt.py tests/test_llm.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.prompt'`)

- [ ] **Step 3: เขียน `rag/prompt.py`**

```python
"""Prompt ที่บังคับให้ LLM ตอบจากตัวบทที่ค้นได้เท่านั้น"""
from __future__ import annotations

from rag.loader import Chunk
from rag.retriever import Hit

NOT_FOUND = "ไม่พบข้อมูลในเอกสารที่มีอยู่"
DISCLAIMER = "หมายเหตุ: ข้อมูลเบื้องต้นจากตัวบทกฎหมาย ไม่ใช่คำปรึกษาทางกฎหมาย"

SYSTEM_PROMPT = f"""คุณคือ "ผู้ช่วยกฎหมายสำหรับคนทำงาน" ตอบคำถามเรื่องสิทธิลูกจ้างตามพระราชบัญญัติคุ้มครองแรงงาน และเรื่องข้อมูลส่วนบุคคลตามพระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล โดยใช้ตัวบทใน <context> ที่ให้มาเท่านั้น

กฎที่ต้องทำตามอย่างเคร่งครัด:
1. ใช้เฉพาะข้อมูลใน <context> ห้ามใช้ความรู้ภายนอก ห้ามเดา ห้ามแต่งตัวเลข อัตรา หรือเลขมาตราที่ไม่มีใน <context>
2. ทุกประโยคที่อ้างกฎหมาย ต้องใส่เลขอ้างอิงในวงเล็บเหลี่ยมตามลำดับใน <context> เช่น [1] และระบุชื่อ พ.ร.บ. กับเลขมาตรา
3. ถ้า <context> ไม่มีข้อมูลที่ตอบคำถามได้โดยตรง ให้ตอบประโยคนี้ประโยคเดียวเท่านั้น แม้คำถามเป็นภาษาอังกฤษ: "{NOT_FOUND}"
4. ตอบเป็นภาษาเดียวกับคำถาม (ไทยหรืออังกฤษ) ใช้ภาษาที่ประชาชนทั่วไปเข้าใจง่าย ถ้าตัวบทเขียนจำนวนเป็นตัวหนังสือ ให้เขียนเป็นตัวเลขด้วย เช่น "หนึ่งร้อยแปดสิบวัน (180 วัน)"
5. ถ้าตอบได้ ให้ปิดท้ายด้วยบรรทัด: "{DISCLAIMER}"
"""

REWRITE_PROMPT = (
    "จากประวัติการสนทนาและคำถามล่าสุด ให้เขียนคำถามล่าสุดใหม่เป็นคำถามที่สมบูรณ์ในตัวเอง "
    "อ่านเข้าใจได้โดยไม่ต้องดูประวัติ คงภาษาเดิมของคำถาม ห้ามตอบคำถาม "
    "ส่งกลับเฉพาะคำถามที่เขียนใหม่เพียงบรรทัดเดียว"
)


def source_label(chunk: Chunk) -> str:
    label = f"{chunk.law_name} — {chunk.chapter}"
    return f"{label} — มาตรา {chunk.section}" if chunk.section else label


def build_context(hits: list[Hit]) -> str:
    blocks = [f"[{n}] {source_label(h.chunk)}\n{h.chunk.text}" for n, h in enumerate(hits, 1)]
    return "<context>\n" + "\n\n".join(blocks) + "\n</context>"


def is_not_found(answer: str) -> bool:
    # LLM บางครั้งขึ้นต้นด้วย "ขออภัย" ก่อนประโยคปฏิเสธ จึงดูเฉพาะช่วงต้นของคำตอบ
    return "ไม่พบข้อมูล" in answer.strip()[:40]
```

- [ ] **Step 4: เขียน `rag/llm.py`**

```python
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
```

- [ ] **Step 5: รัน test ให้ผ่าน**

Run: `source scripts/dev_env.sh && pytest tests/test_prompt.py tests/test_llm.py -v`
Expected: PASS ทั้ง 9 test

- [ ] **Step 6: Commit**

```bash
git add rag/prompt.py rag/llm.py tests/test_prompt.py tests/test_llm.py
git commit -m "feat: grounded system prompt and Groq wrapper with query rewriting

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Pipeline ต่อหนึ่งคำถาม

**Files:**
- Create: `rag/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Retriever.search`, `is_relevant`, `DEFAULT_THRESHOLD`, `Hit` (Task 3); `build_context`, `is_not_found`, `NOT_FOUND` (Task 4); `llm.rewrite_query`, `llm.generate_answer`, `LLMError` (Task 4)
- Produces:
  - `MAX_QUESTION_CHARS = 1000`
  - `RagResult(answer: str | None, hits: list[Hit], search_query: str, error: str | None = None)` — `answer=None` เฉพาะเมื่อ `error` ไม่ใช่ None; ถ้า `answer == NOT_FOUND` แล้ว `hits == []`
  - `validate_question(question: str) -> str | None` (คืนข้อความเตือน หรือ None ถ้าผ่าน)
  - `answer_question(question, history, retriever, client, *, k=4, laws=None, threshold=DEFAULT_THRESHOLD) -> RagResult` (โยน `ValueError` ถ้า `validate_question` ไม่ผ่าน)

- [ ] **Step 1: เขียน failing tests**

`tests/test_pipeline.py`:
```python
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


@pytest.mark.parametrize("bad", ["", "   ", "ก" * 1001])
def test_validate_question_rejects_blank_and_too_long(bad):
    assert pipeline.validate_question(bad)
    with pytest.raises(ValueError):
        pipeline.answer_question(bad, [], FakeRetriever(0.9), MagicMock())


def test_validate_question_accepts_normal_text():
    assert pipeline.validate_question("ลาป่วยได้กี่วัน") is None
    assert pipeline.validate_question("ก" * 1000) is None
```

- [ ] **Step 2: รัน test ให้ fail**

Run: `source scripts/dev_env.sh && pytest tests/test_pipeline.py -v`
Expected: FAIL (`ImportError: cannot import name 'pipeline' from 'rag'`)

- [ ] **Step 3: เขียน `rag/pipeline.py`**

```python
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
```

- [ ] **Step 4: รัน test ทั้งชุดให้ผ่าน**

Run: `source scripts/dev_env.sh && pytest -v`
Expected: PASS ทุก test (loader, split_law, data, retriever, prompt, llm, pipeline)

- [ ] **Step 5: Commit**

```bash
git add rag/pipeline.py tests/test_pipeline.py
git commit -m "feat: RAG pipeline with threshold gate, rewrite fallback and error handling

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Streamlit UI (`app.py`)

**Files:**
- Create: `app.py`, `.claude/launch.json` (gitignored)

**Interfaces:**
- Consumes: `LAWS`, `load_chunks` (Task 1); `Retriever`, `DEFAULT_THRESHOLD` (Task 3); `get_client` (Task 4); `source_label` (Task 4); `answer_question`, `validate_question`, `RagResult` (Task 5)
- Produces: แอปที่ `streamlit run app.py` ได้

- [ ] **Step 1: ผู้ใช้สร้าง `.streamlit/secrets.toml` เอง**

ขอให้ผู้ใช้สร้างไฟล์ `.streamlit/secrets.toml` ที่มีบรรทัด `GROQ_API_KEY = "<key ของผู้ใช้>"` — **ผู้ช่วย AI ไม่เขียนไฟล์นี้**. ตรวจเฉพาะว่ามีไฟล์และถูก ignore:
```bash
test -f .streamlit/secrets.toml && git check-ignore .streamlit/secrets.toml
```
Expected: พิมพ์ `.streamlit/secrets.toml` (แปลว่าถูก ignore)

- [ ] **Step 2: เขียน `app.py`**

```python
"""ผู้ช่วยกฎหมายสำหรับคนทำงาน — แชตบอต RAG (พ.ร.บ.คุ้มครองแรงงาน + PDPA)"""
import streamlit as st

from rag.llm import get_client
from rag.loader import LAWS, load_chunks
from rag.pipeline import RagResult, answer_question, validate_question
from rag.prompt import source_label
from rag.retriever import DEFAULT_THRESHOLD, Retriever

DATA_DIR = "data"
EXAMPLES = [
    "ทำงานมา 3 ปีแล้วถูกเลิกจ้าง จะได้ค่าชดเชยเท่าไร",
    "ลาป่วยได้กี่วันต่อปีโดยยังได้รับค่าจ้าง",
    "บริษัทจะเก็บข้อมูลส่วนบุคคลของพนักงาน ต้องขอความยินยอมไหม",
    "ถ้าข้อมูลส่วนบุคคลรั่วไหล บริษัทต้องแจ้งภายในกี่ชั่วโมง",
]
AI_BUSY = "⚠️ ระบบ AI ไม่ว่างในขณะนี้ ลองใหม่อีกครั้ง — ด้านล่างคือมาตราที่ค้นได้"

st.set_page_config(page_title="ผู้ช่วยกฎหมายคนทำงาน", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="กำลังเตรียมคลังกฎหมาย (ครั้งแรกเท่านั้น)...")
def load_retriever() -> Retriever:
    return Retriever(load_chunks(DATA_DIR))


@st.cache_resource
def load_client(api_key: str):
    return get_client(api_key)


def get_api_key() -> str | None:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:  # ไม่มีไฟล์ secrets หรือไม่มี key
        return None


def to_message(result: RagResult) -> dict:
    return {
        "role": "assistant",
        "content": result.answer if result.error is None else AI_BUSY,
        "sources": [
            {
                "label": source_label(h.chunk),
                "cosine": h.cosine,
                "text": h.chunk.text,
                "url": h.chunk.source_url,
            }
            for h in result.hits
        ],
        "search_query": result.search_query,
    }


def render_message(msg: dict) -> None:
    st.markdown(msg["content"])
    if msg["role"] != "assistant":
        return
    sources = msg.get("sources", [])
    if sources:
        with st.expander(f"📚 แหล่งอ้างอิง ({len(sources)})"):
            for n, s in enumerate(sources, 1):
                st.markdown(f"**[{n}] {s['label']}** · ความคล้าย {s['cosine']:.2f}")
                st.text(s["text"])
                st.caption(f"ที่มา: {s['url']}")
    else:
        st.caption("ไม่มีแหล่งอ้างอิง เพราะไม่พบตัวบทที่ตอบคำถามนี้ได้")
    if msg.get("search_query"):
        with st.expander("🔎 คำค้นที่ใช้"):
            st.code(msg["search_query"], language=None)


with st.sidebar:
    st.title("⚖️ ผู้ช่วยกฎหมายคนทำงาน")
    st.caption("ตอบจากตัวบท พ.ร.บ.คุ้มครองแรงงาน และ PDPA ด้วยเทคนิค RAG")
    st.subheader("ขอบเขตกฎหมาย")
    laws = {code for code, name in LAWS.items() if st.checkbox(name, value=True, key=f"law_{code}")}
    k = st.slider("จำนวนมาตราที่ค้น", min_value=2, max_value=8, value=4)
    if st.button("🗑 ล้างแชต"):
        st.session_state.messages = []
        st.rerun()
    st.warning("ข้อมูลเบื้องต้นจากตัวบทกฎหมาย ไม่ใช่คำปรึกษาทางกฎหมาย")
    st.caption("แหล่งข้อมูล: ตัวบทฉบับรวมแก้ไขจากแหล่งทางการ — ดูรายละเอียดใน README ของโปรเจกต์")

st.header("ถามเรื่องสิทธิลูกจ้าง & ข้อมูลส่วนบุคคล")

api_key = get_api_key()
if not api_key:
    st.error(
        "ยังไม่ได้ตั้งค่า GROQ_API_KEY — บนเครื่อง: สร้าง `.streamlit/secrets.toml` "
        "บน Streamlit Cloud: Settings → Secrets แล้วใส่ `GROQ_API_KEY = \"...\"`"
    )
    st.stop()

retriever = load_retriever()
client = load_client(api_key)
st.session_state.setdefault("messages", [])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_message(msg)

pending = None
if not st.session_state.messages:
    st.markdown("**ลองถามคำถามตัวอย่าง:**")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, key=f"example_{i}"):
            pending = example

if not laws:
    st.info("เลือกกฎหมายอย่างน้อย 1 ฉบับที่แถบด้านซ้าย")
typed = st.chat_input("พิมพ์คำถามเกี่ยวกับสิทธิลูกจ้างหรือ PDPA...", disabled=not laws)
question = typed or pending

if question:
    problem = validate_question(question)
    if problem:
        st.warning(problem)
        st.stop()
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        render_message({"role": "user", "content": question})
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นตัวบท..."):
            result = answer_question(
                question, history, retriever, client, k=k, laws=laws, threshold=DEFAULT_THRESHOLD
            )
        msg = to_message(result)
        render_message(msg)
    st.session_state.messages.append(msg)
    if pending:
        st.rerun()  # ซ่อนปุ่มคำถามตัวอย่างหลังเริ่มแชต
```

- [ ] **Step 3: ตั้งค่า launch config สำหรับ preview**

`.claude/launch.json`:
```json
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "streamlit",
      "runtimeExecutable": "bash",
      "runtimeArgs": ["-lc", "source scripts/dev_env.sh && streamlit run app.py --server.port 8501 --server.headless true"],
      "port": 8501
    }
  ]
}
```

- [ ] **Step 4: รันแอปและตรวจด้วย in-app browser**

เปิดด้วย `preview_start` ชื่อ `streamlit` แล้วตรวจทีละข้อ (ใช้ `get_page_text` / screenshot):
1. หน้าแรกขึ้นปุ่มคำถามตัวอย่าง 4 ปุ่ม, sidebar มี checkbox 2 ตัว, slider, ปุ่มล้างแชต, คำเตือน
2. กดปุ่ม "ทำงานมา 3 ปีแล้วถูกเลิกจ้าง..." → คำตอบมี `[1]` และ "มาตรา 118", expander แหล่งอ้างอิงแสดง ม.118, ปุ่มตัวอย่างหายไป
3. พิมพ์ต่อ "แล้วถ้าทำงาน 10 ปีล่ะ" → expander "คำค้นที่ใช้" แสดงคำถามที่ rewrite แล้ว (มีคำว่า ค่าชดเชย/เลิกจ้าง)
4. พิมพ์ "สูตรทำต้มยำกุ้ง" → ตอบ `ไม่พบข้อมูลในเอกสารที่มีอยู่` และขึ้น caption ไม่มีแหล่งอ้างอิง
5. ติ๊กออกทั้งสองกฎหมาย → ช่องพิมพ์ถูก disable และขึ้นข้อความให้เลือกกฎหมาย
6. ถาม "How many days of annual leave do I get?" → ตอบเป็นภาษาอังกฤษพร้อมอ้าง ม.30
7. กด "ล้างแชต" → กลับสู่หน้าแรก

Expected: ผ่านครบ 7 ข้อ ถ้าข้อไหนไม่ผ่านให้ใช้ superpowers:systematic-debugging ก่อนแก้

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: Streamlit chat UI with sources, law filter and example questions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: คำถามทดสอบ + `evaluate.py` + ปรับ threshold

**Files:**
- Create: `test_questions.csv`, `evaluate.py`
- Modify: `rag/retriever.py` (ค่า `DEFAULT_THRESHOLD`)

**Interfaces:**
- Consumes: `load_chunks`, `Retriever`, `DEFAULT_THRESHOLD`, `answer_question`, `is_not_found`, `get_client`
- Produces: `test_questions.csv` คอลัมน์ `id,question,expected_answer,expected_source,answerable`; `expected_source` รูปแบบ `law:section` คั่นหลายค่าด้วย `;` (ว่างเมื่อ `answerable=false`)

- [ ] **Step 1: ร่าง `test_questions.csv`**

```csv
id,question,expected_answer,expected_source,answerable
1,ทำงานมา 3 ปีแล้วถูกเลิกจ้าง จะได้ค่าชดเชยเท่าไร,ได้ค่าชดเชยไม่น้อยกว่าค่าจ้างอัตราสุดท้าย 180 วัน (ทำงานครบ 3 ปีแต่ไม่ครบ 6 ปี),labor:118,true
2,ลาป่วยได้รับค่าจ้างปีละกี่วัน,ลาป่วยได้เท่าที่ป่วยจริง และได้รับค่าจ้างในวันลาป่วยปีหนึ่งไม่เกิน 30 วันทำงาน,labor:32;labor:57,true
3,ลูกจ้างหญิงมีสิทธิลาคลอดได้กี่วัน,ลาเพื่อคลอดบุตรครรภ์หนึ่งไม่เกิน 98 วัน (นับรวมวันหยุดในระหว่างลา),labor:41,true
4,เวลาทำงานปกติต้องไม่เกินวันละกี่ชั่วโมง,ไม่เกินวันละ 8 ชั่วโมง และสัปดาห์หนึ่งไม่เกิน 48 ชั่วโมง,labor:23,true
5,ทำงานล่วงเวลาในวันทำงานได้ค่าล่วงเวลากี่เท่า,ไม่น้อยกว่า 1.5 เท่าของอัตราค่าจ้างต่อชั่วโมงในวันทำงาน,labor:61,true
6,บริษัทจะเก็บรวบรวมข้อมูลส่วนบุคคลต้องได้รับความยินยอมก่อนหรือไม่,ต้องได้รับความยินยอมจากเจ้าของข้อมูลก่อนหรือในขณะเก็บรวบรวม เว้นแต่มีบทบัญญัติให้กระทำได้,pdpa:19,true
7,เจ้าของข้อมูลส่วนบุคคลขอสำเนาข้อมูลของตนเองได้ไหม,ได้ มีสิทธิขอเข้าถึงและขอรับสำเนาข้อมูลส่วนบุคคลที่เกี่ยวกับตน,pdpa:30,true
8,ข้อมูลส่วนบุคคลรั่วไหล ผู้ควบคุมข้อมูลต้องแจ้งสำนักงานภายในกี่ชั่วโมง,แจ้งสำนักงานโดยไม่ชักช้าภายใน 72 ชั่วโมงนับแต่ทราบเหตุ (เว้นแต่ไม่มีความเสี่ยง),pdpa:37,true
9,ข้อมูลชีวภาพและประวัติอาชญากรรมถือเป็นข้อมูลอ่อนไหวหรือไม่,ใช่ เป็นข้อมูลตามมาตรา 26 ที่ห้ามเก็บรวบรวมโดยไม่ได้รับความยินยอมโดยชัดแจ้ง เว้นแต่เข้าข้อยกเว้น,pdpa:26,true
10,How many days of annual leave am I entitled to?,After working continuously for 1 year the employee is entitled to at least 6 working days of annual holiday,labor:30,true
11,ค่าแรงขั้นต่ำของกรุงเทพมหานครปีนี้วันละเท่าไร,ไม่พบข้อมูลในเอกสารที่มีอยู่ (อัตราค่าจ้างขั้นต่ำกำหนดโดยประกาศคณะกรรมการค่าจ้าง ไม่อยู่ในตัวบท),,false
12,มนุษย์เงินเดือนต้องเสียภาษีเงินได้ในอัตราเท่าไร,ไม่พบข้อมูลในเอกสารที่มีอยู่ (เป็นเรื่องประมวลรัษฎากร),,false
```

- [ ] **Step 2: ตรวจคำตอบทุกข้อกับตัวบทใน `data/`**

```bash
source scripts/dev_env.sh
python - <<'EOF'
import csv
from rag.loader import load_chunks
idx = {}
for c in load_chunks("data"):
    idx.setdefault((c.law, c.section), []).append(c.text)
for row in csv.DictReader(open("test_questions.csv", encoding="utf-8")):
    print(f"\n=== {row['id']}. {row['question']}\nคาดหวัง: {row['expected_answer']}")
    for src in filter(None, row["expected_source"].split(";")):
        law, sec = src.split(":")
        print(f"--- {src}\n" + "\n".join(idx.get((law, sec), ["!!! ไม่พบมาตรานี้ใน data/"])))
EOF
```
อ่านตัวบทที่พิมพ์ออกมาเทียบกับ `expected_answer` ทีละข้อ ถ้าตัวเลข/เงื่อนไข/เลขมาตราไม่ตรงกับตัวบท **ให้แก้ CSV ตามตัวบท** (ไม่ใช่ตามความจำ). ข้อ 11–12 ตรวจด้วย `grep -l "ภาษีเงินได้\|ค่าจ้างขั้นต่ำ" data/*.txt` ว่าตัวบทไม่มีตัวเลขที่ตอบได้

- [ ] **Step 3: เขียน `evaluate.py`**

```python
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
        hits = retriever.search(row["question"], k=args.k)
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
        if client:
            result = answer_question(row["question"], [], retriever, client, k=args.k, threshold=args.threshold)
            refused = result.answer is not None and is_not_found(result.answer)
            if answerable and refused:
                false_refusal += 1
            if not answerable and refused:
                refusal_ok += 1
            line += f" refused={'Y' if refused else 'N'}"
            if result.error:
                line += f" ERROR={result.error}"
        print(line, "|", row["question"])
        if client and args.verbose and result.answer:
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
```

- [ ] **Step 4: วัด retrieval และเลือก threshold**

Run: `source scripts/dev_env.sh && python evaluate.py --no-llm`
Expected: hit rate ≥ 8/10. ถ้าต่ำกว่า ให้ดูข้อที่ `hit=N` แล้วตรวจว่ามาตราที่คาดหวังถูกตัด chunk ถูกต้องไหม (อย่าแก้คำถามให้ง่ายขึ้นเพื่อให้ผ่าน)

นำค่า "threshold แนะนำ" ไปแก้ `DEFAULT_THRESHOLD` ใน `rag/retriever.py` (ปัดทศนิยม 2 ตำแหน่ง)

- [ ] **Step 5: วัดแบบเต็มด้วย Groq**

Run: `source scripts/dev_env.sh && python evaluate.py --verbose`
Expected: `ปฏิเสธถูกต้อง 2/2`, `ปฏิเสธผิด 0/10`, ไม่มี `ERROR=`. อ่านคำตอบ verbose ว่ามี `[n]` และเลขมาตราจริง. ถ้าข้อ 11–12 ไม่ถูกปฏิเสธ ให้ปรับข้อ 3 ของ `SYSTEM_PROMPT` ให้เข้มขึ้น แล้วรัน `pytest tests/test_prompt.py` + evaluate ซ้ำ. **เก็บผลสรุปบล็อกสุดท้ายไว้ใส่ README (Task 8)**

- [ ] **Step 6: รัน test ทั้งชุดอีกครั้งแล้ว commit**

```bash
source scripts/dev_env.sh && pytest -q
git add test_questions.csv evaluate.py rag/retriever.py
git commit -m "test: add test questions, evaluation script and tuned threshold

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: README + ตรวจความปลอดภัยก่อน push

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: URL แหล่งตัวบท (Task 2 Step 5), ผล evaluate (Task 7 Step 5), ค่า threshold (Task 7 Step 4)
- Produces: README ครบหัวข้อที่โจทย์กำหนด

- [ ] **Step 1: เขียน `README.md`**

ต้องมีหัวข้อต่อไปนี้ครบ (เขียนเนื้อหาจริงทุกหัวข้อ):

```markdown
# ⚖️ ผู้ช่วยกฎหมายสำหรับคนทำงาน (Labor & PDPA RAG Chatbot)

แชตบอตตอบคำถามเรื่องสิทธิลูกจ้างและข้อมูลส่วนบุคคล จากตัวบท พ.ร.บ.คุ้มครองแรงงาน พ.ศ. 2541 (ฉบับรวมแก้ไข) และ พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562 ด้วยเทคนิค RAG

🔗 เว็บแอป: <ใส่หลัง deploy ใน Task 9>

## แนวคิดของ Domain
(ทำไมเลือกกฎหมายสองฉบับนี้: คำถามที่คนทำงานเจอบ่อย — ค่าชดเชย ลา ทำงานล่วงเวลา และข้อมูลส่วนบุคคลของพนักงานที่บริษัทเก็บ; ตัวบทเป็นแหล่งที่น่าเชื่อถือที่สุดและอ้างอิงเป็นมาตราได้)

## วิธีใช้งาน
- บนเว็บ: เลือกกฎหมายที่ sidebar → พิมพ์คำถามหรือกดคำถามตัวอย่าง → เปิด "📚 แหล่งอ้างอิง" เพื่อดูตัวบท
- รันบนเครื่อง: python -m venv .venv / pip install -r requirements.txt / คัดลอก .streamlit/secrets.toml.example เป็น secrets.toml แล้วใส่ key / streamlit run app.py
- Deploy: Streamlit Community Cloud → New app → เลือก repo, main, app.py → Advanced: Python 3.11 → Secrets: GROQ_API_KEY

## สถาปัตยกรรม RAG
(แผนภาพ flow: คำถาม → query rewriting → FAISS + BM25 → RRF → threshold → Groq → คำตอบ + อ้างอิง)
- Document Loading & Chunking: หัวไฟล์ LAW/CHAPTER/SOURCE, clean_text (เลขไทย→อารบิก, ลบเชิงอรรถ/เลขหน้า), ตัด 1 chunk = 1 มาตรา, มาตรา > 800 ตัวอักษรแบ่งย่อย overlap 100 — เหตุผล: มาตราคือหน่วยที่อ้างอิงได้
- Embedding & Vector Search: multilingual-e5-small + FAISS IndexFlatIP — เหตุผล: เล็ก (~470 MB) รองรับไทย/อังกฤษ
- Hybrid Retrieval: BM25 (pythainlp newmm) + RRF k=60 — เหตุผล: จับเลขมาตราและศัพท์กฎหมายตรงตัว
- Prompt Engineering: (ยก SYSTEM_PROMPT และ REWRITE_PROMPT จาก rag/prompt.py)
- การปฏิเสธ 2 ชั้น: threshold cosine = <ค่าจาก Task 7> + prompt rule
- LLM: Groq llama-3.3-70b-versatile, temperature 0.1

## แหล่งที่มาของเอกสาร
(ตาราง: ไฟล์ใน data/ → หมวด → URL ต้นทางจาก Task 2; วิธีสร้างซ้ำด้วย curl + scripts/split_law.py)

## ผลการทดสอบ
(วางบล็อก "=== สรุป ===" จาก python evaluate.py และอธิบายสั้น ๆ; อธิบายไฟล์ test_questions.csv)

## ตัวอย่าง Prompt ที่ใช้สั่ง AI ระหว่างพัฒนา
(ยก prompt ที่ใช้กับ Claude Code จริง: โจทย์ที่วางให้ brainstorm, การเลือกหัวข้อ C, แนวทาง 1, การขอเขียน spec/plan, และ prompt ระหว่าง implement)

## ข้อจำกัด
ไม่รวมกฎกระทรวง/ประกาศ (เช่น ค่าแรงขั้นต่ำ), ข้อมูลเบื้องต้นไม่ใช่คำปรึกษากฎหมาย, LLM อาจผิดพลาดได้ — ตรวจตัวบทในแหล่งอ้างอิงเสมอ
```
ข้อความในวงเล็บคือคำสั่งให้เขียนเนื้อหาจริงแทน — ไฟล์สุดท้ายต้องไม่มีวงเล็บคำสั่งหลงเหลือ ยกเว้นบรรทัด URL เว็บแอปที่เติมใน Task 9

- [ ] **Step 2: ตรวจว่าไม่มี secret ใน repo**

```bash
git add README.md
git grep -n -i "gsk_" -- . ; echo "exit=$?"
git ls-files | grep -E "secrets\.toml$" ; echo "exit=$?"
```
Expected: ทั้งสองคำสั่ง `exit=1` (ไม่พบอะไร) — ถ้าพบ **หยุด** ห้าม commit/push

- [ ] **Step 3: Commit**

```bash
git commit -m "docs: add README with architecture, sources, evaluation and prompts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Deploy + เตรียมสิ่งที่ต้องส่ง

**Files:**
- Modify: `README.md` (URL เว็บแอป)
- Create: `docs/submission.md`

**Interfaces:**
- Consumes: repo ที่ผ่าน Task 1–8
- Produces: GitHub repo ส่วนตัว, URL Streamlit ที่ใช้ได้จริง, ข้อความสำหรับ Colab, checklist ภาพหน้าจอ

- [ ] **Step 1: ผู้ใช้สร้าง GitHub repo และ push**

ผู้ใช้เลือกทางใดทางหนึ่ง (ถามก่อน — การสร้าง repo/push เป็นการกระทำภายนอก):
- ผู้ใช้ login `gh auth login` เองแล้วให้ผู้ช่วยรัน: `gh repo create labor-pdpa-rag --private --source . --push`
- หรือผู้ใช้สร้าง repo ว่างบนเว็บ GitHub แล้วรัน `git remote add origin <url>` และ `git push -u origin main`

ตรวจหลัง push (repo private ต้องใช้ `gh`):
```bash
gh repo view --json url,visibility
```

**หมายเหตุ:** Streamlit Community Cloud deploy จาก repo private ได้ (ต้องอนุญาตสิทธิ์ GitHub ตอนเชื่อมบัญชี) แต่ผู้ตรวจต้องเห็นซอร์สโค้ด — ถามผู้ใช้ว่าจะเพิ่มอาจารย์เป็น collaborator หรือเปลี่ยนเป็น public ก่อนส่ง

- [ ] **Step 2: ผู้ใช้ deploy บน Streamlit Community Cloud**

ผู้ใช้ทำเอง: share.streamlit.io → Create app → repo `labor-pdpa-rag`, branch `main`, file `app.py` → Advanced settings: Python 3.11, Secrets: `GROQ_API_KEY = "..."` → Deploy

ถ้า build ล้มเหลวที่ขั้นติดตั้ง torch จากบรรทัด `--extra-index-url` (ดู log บนหน้า app) ให้ลบบรรทัดนั้นออกจาก `requirements.txt`, commit, push แล้ว reboot app

- [ ] **Step 3: ทดสอบ URL จริงจากเบราว์เซอร์ที่ไม่ได้ล็อกอิน**

เปิด URL ด้วย in-app browser (ไม่มี session GitHub/Streamlit) แล้วทำรายการตรวจ 7 ข้อเดียวกับ Task 6 Step 4
Expected: ผ่านครบ และไม่มีหน้าขอ login

- [ ] **Step 4: ใส่ URL ใน README และเขียน `docs/submission.md`**

`docs/submission.md`:
```markdown
# สิ่งที่ต้องส่ง

## ข้อความสำหรับวางในไฟล์ Colab
- หัวข้อที่เลือก: ผู้ช่วยกฎหมายสำหรับประชาชน — "ผู้ช่วยกฎหมายสำหรับคนทำงาน" ตอบคำถามสิทธิลูกจ้าง (พ.ร.บ.คุ้มครองแรงงาน พ.ศ. 2541 ฉบับรวมแก้ไข) และ PDPA (พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562) จากตัวบทกฎหมาย
- URL หน้าเว็บ Streamlit: <URL จาก Step 2>
- GitHub Repository URL: <URL จาก Step 1>

## ภาพหน้าจอสำหรับ PDF (แคปตามลำดับ พร้อมคำอธิบายใต้ภาพ)
1. หน้าแรก — sidebar (เลือกกฎหมาย, จำนวนมาตรา, ล้างแชต, คำเตือน) และปุ่มคำถามตัวอย่าง
2. ถามค่าชดเชย 3 ปี — คำตอบอ้าง [1] มาตรา 118 พร้อมเปิด expander แหล่งอ้างอิง (Retrieval + Citation)
3. ถามต่อ "แล้วถ้าทำงาน 10 ปีล่ะ" — เปิด expander คำค้นที่ใช้ แสดง query rewriting (คุยต่อเนื่อง)
4. ถามเรื่อง PDPA (ข้อมูลรั่วไหล 72 ชั่วโมง) — อ้างมาตรา 37
5. ถามภาษาอังกฤษเรื่อง annual leave — ตอบเป็นภาษาอังกฤษ
6. ถามค่าแรงขั้นต่ำ — ตอบ "ไม่พบข้อมูลในเอกสารที่มีอยู่" (ปฏิเสธเมื่อไม่มีข้อมูล)
7. ถามนอกเรื่อง (ต้มยำกุ้ง) — ปฏิเสธด้วย threshold
8. ติ๊กเลือกเฉพาะ PDPA แล้วถามเรื่องลาป่วย — แสดงการกรองตามกฎหมาย
9. หน้า GitHub repo — โครงสร้างไฟล์ app.py, requirements.txt, README.md, data/, test_questions.csv
10. ผลรัน python evaluate.py (สรุป hit rate และการปฏิเสธ)
11. หน้า Streamlit Cloud Settings → Secrets (บังค่า key ก่อนแคป) — แสดงว่าเก็บ key นอก repo
```
แทน `<URL ...>` ด้วยค่าจริงก่อน commit

- [ ] **Step 5: Commit และ push**

```bash
git grep -n -i "gsk_" -- . ; echo "exit=$?"   # ต้อง exit=1
git add README.md docs/submission.md
git commit -m "docs: add deployed URL and submission checklist

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```
Expected: Streamlit Cloud redeploy อัตโนมัติ และ URL ยังใช้งานได้
