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
from pythainlp.util import thai_digit_to_arabic_digit  # noqa: E402

from rag.loader import LAWS, SECTION_RE, clean_text  # noqa: E402

CHAPTER_RE = re.compile(r"^หมวด\s+(\d+)(?:/(\d+))?\s*(.*)$")
TRANSITIONAL_RE = re.compile(r"^บทเฉพาะกาล\s*$")
# ท้ายตัวบท: ลายมือชื่อผู้รับสนอง / หมายเหตุ / พ.ร.บ.ฉบับแก้ไขที่แนบท้าย — ไม่นำเข้าคลัง
END_RE = re.compile(r"^(ผู้รับสนอง(พระบรม)?ราชโองการ|หมายเหตุ\s*[:\-]?\s*เหตุผล)")

_MONTHS = "มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม"
# หัว/ท้ายกระดาษของฉบับพิมพ์กฤษฎีกาและราชกิจจานุเบกษา
NOISE_RES = [
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4} \d{1,2}:\d{2}$"),
    re.compile(r"^สำนักงานคณะกรรมการกฤษฎีกา$"),
    re.compile(r"^about:blank$"),
    re.compile(r"^\d+/\d+$"),
    re.compile(r"^[่้]?เล่?ม \d+ ตอนที่ \d+ \S+$"),
    re.compile(r"^ราชกิจจานุเบกษา$"),
    re.compile(rf"^\d{{1,2}} ({_MONTHS}) \d{{4}}$"),
    re.compile(r"^[่้]?ห้?น้?า \d+[่้]?$"),
]
FOOTNOTE_MARK_LINE_RE = re.compile(r"^\[\d+\]$")
HEADER_RE = re.compile(r"^มาตรา (\d+)(?:/(\d+))?(?: (ทวิ|ตรี|จัตวา|เบญจ))?(?= |$)")
_SUFFIX = {"ทวิ": 1, "ตรี": 2, "จัตวา": 3, "เบญจ": 4}
MAX_SECTION_GAP = 3
# บรรทัดที่ขึ้นต้นแบบนี้คือย่อหน้าใหม่ ไม่ต่อกับบรรทัดก่อนหน้า
NEW_PARAGRAPH_RE = re.compile(r"^(หมวด \d|บทเฉพาะกาล|ส่วนที่ \d|\(\d+\)|\([ก-ฮ]\)|“|ผู้รับสนอง|หมายเหตุ)")


def prepare_text(text: str) -> str:
    """แก้ข้อความที่ดึงจาก PDF: ลบหัว/ท้ายกระดาษ, แก้สระอำแตก, ต่อบรรทัดที่ถูกตัด
    และต่อบรรทัดที่ขึ้นต้นด้วย "มาตรา N" ซึ่งเป็นการอ้างถึงมาตราอื่นกลางย่อหน้า"""
    text = thai_digit_to_arabic_digit(text.replace("\r\n", "\n"))
    # ฟอนต์ราชกิจจาฯ ทำให้ "ำ" ถูกดึงออกมาเป็น " า" — ภาษาไทยไม่มีคำที่ขึ้นต้นด้วย า จึงแก้กลับได้เสมอ
    text = re.sub(r"(?<=[ก-ฮ่้๊๋]) า", "ำ", text)

    rows = []  # (ข้อความ, มีช่องว่างนำหน้า, มีช่องว่างต่อท้าย)
    for raw in text.split("\n"):
        # วรรณยุกต์ที่หลุดจากหัวกระดาษมาค้างท้ายบรรทัด (" ้") ไม่ใช่เนื้อความ
        raw = re.sub(r"\s[่้๊๋]\s*$", " ", raw)
        line = re.sub(r"\s+", " ", raw).strip()
        if line and not any(r.match(line) for r in NOISE_RES):
            rows.append((line, raw[:1].isspace(), raw[-1:].isspace()))
    # "ที่" ที่อยู่ก่อนเลขเชิงอรรถ [n] คือเศษของตัวยกเชิงอรรถ ไม่ใช่เนื้อความ
    rows = [
        r for i, r in enumerate(rows)
        if not FOOTNOTE_MARK_LINE_RE.match(r[0])
        and not (r[0] == "ที่" and i + 1 < len(rows) and FOOTNOTE_MARK_LINE_RE.match(rows[i + 1][0]))
    ]

    out: list[str] = []
    prev_key = (0, 0)
    prev_trailing = prev_bare_heading = False
    for line, leading, trailing in rows:
        m = HEADER_RE.match(line)
        is_reference = False
        if m:
            key = (int(m.group(1)), int(m.group(2) or _SUFFIX.get(m.group(3) or "", 0)))
            if prev_key == (0, 0) or (key > prev_key and key[0] - prev_key[0] <= MAX_SECTION_GAP):
                prev_key = key
            else:
                is_reference = True
        new_line = (
            not out
            or (m and not is_reference)
            or NEW_PARAGRAPH_RE.match(line)
            or prev_bare_heading
        )
        if new_line:
            out.append(line)
        else:
            sep = " " if (is_reference or leading or prev_trailing or HEADER_RE.fullmatch(out[-1])) else ""
            out[-1] = out[-1] + sep + line
        prev_trailing = trailing
        prev_bare_heading = bool(re.fullmatch(r"หมวด \d+(?:/\d+)?", line))
    return "\n".join(out)


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
    chapters = split_chapters(prepare_text(args.raw.read_text(encoding="utf-8")))
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
