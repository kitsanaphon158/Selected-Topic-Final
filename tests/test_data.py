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


def test_labor_text_is_latest_consolidated_version():
    def section_text(law: str, section: str) -> str:
        return "".join(c.text for c in load_chunks(DATA) if c.law == law and c.section == section)

    assert "สี่ร้อยวัน" in section_text("labor", "118"), "ม.118 ยังไม่ใช่ฉบับแก้ไข 2562 (ค่าชดเชย 400 วัน)"
    assert "หนึ่งร้อยยี่สิบวัน" in section_text("labor", "41"), "ม.41 ยังไม่ใช่ฉบับแก้ไข 2568 (ลาคลอด 120 วัน)"
