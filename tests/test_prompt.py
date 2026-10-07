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
    assert "ใช้เฉพาะบางกรณี" in SYSTEM_PROMPT
    assert "ไม่เกี่ยวกับ" in SYSTEM_PROMPT  # คำถามนอกเรื่องต้องถูกปฏิเสธ  # มาตราแบบมีเงื่อนไข (เช่น ม.122) ต้องแยกเป็นข้อสังเกต


def test_is_not_found_variants():
    assert is_not_found(NOT_FOUND)
    assert is_not_found("  ไม่พบข้อมูลในเอกสารที่มีอยู่\n\nหมายเหตุ ...")
    assert is_not_found("ขออภัย ไม่พบข้อมูลในเอกสารที่มีอยู่")
    assert not is_not_found(
        "ตามมาตรา 118 ลูกจ้างที่ทำงานครบสามปีแต่ไม่ครบหกปีมีสิทธิได้รับค่าชดเชย 180 วัน [1] "
        "หากไม่พบข้อมูลเพิ่มเติมให้สอบถามพนักงานตรวจแรงงาน"
    )


def test_build_context_sends_the_full_section_not_the_window():
    chunk = make_chunk("labor", "118", "(ต่อ) มาตรา 118 ส่วนที่สอง", part=1, section_text="มาตรา 118 ส่วนแรก ส่วนที่สอง")
    ctx = build_context([Hit(chunk, 0.03, 0.9)])
    assert "มาตรา 118 ส่วนแรก ส่วนที่สอง" in ctx
    assert "(ต่อ)" not in ctx


def test_build_context_caps_each_section_and_the_total_size():
    from rag.prompt import MAX_CONTEXT_CHARS, MAX_SECTION_CHARS
    hits = [Hit(make_chunk("labor", str(n), "x", section_text=f"มาตรา {n} " + "ก" * 3000), 0.03, 0.9) for n in range(1, 7)]
    ctx = build_context(hits)
    assert len(ctx) <= MAX_CONTEXT_CHARS + 600
    assert "ก" * (MAX_SECTION_CHARS + 1) not in ctx
    assert "(ตัดทอน)" in ctx
    assert "[1] " in ctx
