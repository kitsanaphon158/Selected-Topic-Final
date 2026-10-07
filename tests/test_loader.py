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


def test_every_part_of_a_long_section_carries_the_full_section_text():
    body = "มาตรา 118 ให้นายจ้างจ่ายค่าชดเชย\n" + "\n".join(f"({i}) ลูกจ้างซึ่งทำงานติดต่อกันครบ {i} ปี ให้จ่ายไม่น้อยกว่าค่าจ้างอัตราสุดท้าย" * 3 for i in range(1, 7))
    doc = Document(text=body, law="labor", law_name="L", chapter="หมวด 11", source_file="labor_ch11.txt", source_url="u")
    chunks = chunk_by_section(doc)
    assert len(chunks) > 1
    assert all(c.section_text == body for c in chunks)


def test_continuation_windows_start_at_a_word_boundary():
    words = ["ลูกจ้าง", "นายจ้าง", "ค่าชดเชย", "วันทำงาน"]
    body = "มาตรา 9 " + " ".join(words[i % 4] for i in range(400))
    doc = Document(text=body, law="labor", law_name="L", chapter="หมวด 1", source_file="labor_ch01.txt", source_url="u")
    for c in chunk_by_section(doc)[1:]:
        first_word = c.text.split("\n", 1)[1].split(" ", 1)[0]
        assert first_word in words, first_word
