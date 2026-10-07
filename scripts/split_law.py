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
