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
