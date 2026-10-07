from scripts.split_law import chapter_filename, prepare_text, split_chapters

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


def test_prepare_text_drops_page_headers_footers_and_footnote_anchors():
    raw = "\n".join([
        "หน่วยงานดังกล่าว",
        "ที่",
        "[1]",
        "11/12/68 13:54",
        "สำนักงานคณะกรรมการกฤษฎีกา",
        "about:blank",
        "1/52",
        "",
        "เป็นผู้ควบคุม",
        "เลม   ๑๓๖   ตอนที่   ๖๙    ก",
        "ราชกิจจานุเบกษา",
        "๒๗   พฤษภาคม   ๒๕๖๒",
        "้หนา   ๕๒",
        "และดูแล",
    ])
    assert prepare_text(raw) == "หน่วยงานดังกล่าวเป็นผู้ควบคุมและดูแล"


def test_prepare_text_fixes_broken_sara_am():
    assert prepare_text("ให้ส านักงานด าเนินการ ไม่ต่ ากว่า") == "ให้สำนักงานดำเนินการ ไม่ต่ำกว่า"


def test_prepare_text_joins_wrapped_lines_and_reference_lines():
    raw = "\n".join([
        "มาตรา ๓๒  ให้ลูกจ้างมีสิทธิลาป่วยได้เท่าที่ป่วยจริง การลาป่วยตั้งแต่สามวัน",
        "ทำงานขึ้นไป ",
        "นายจ้างอาจให้แสดงใบรับรองแพทย์",
        "มาตรา ๔๑ มิให้ถือเป็นวันลาป่วย",
        "ตามมาตรานี้",
        "มาตรา ๓๔",
        "   ให้ลูกจ้างมีสิทธิลาเพื่อกิจธุระ",
        "(๑) กรณีหนึ่ง",
        "“วันลา” หมายความว่า วันที่ลูกจ้างลา",
    ])
    assert prepare_text(raw) == "\n".join([
        "มาตรา 32 ให้ลูกจ้างมีสิทธิลาป่วยได้เท่าที่ป่วยจริง การลาป่วยตั้งแต่สามวันทำงานขึ้นไป "
        "นายจ้างอาจให้แสดงใบรับรองแพทย์ มาตรา 41 มิให้ถือเป็นวันลาป่วยตามมาตรานี้",
        "มาตรา 34 ให้ลูกจ้างมีสิทธิลาเพื่อกิจธุระ",
        "(1) กรณีหนึ่ง",
        "“วันลา” หมายความว่า วันที่ลูกจ้างลา",
    ])


def test_prepare_text_keeps_chapter_heading_and_title_lines():
    raw = "มาตรา ๓๗  ห้ามยกของหนัก\nหมวด ๓\nการใช้แรงงานหญิง\nมาตรา ๓๘  ห้ามนายจ้าง\nบทเฉพาะกาล\nมาตรา ๓๙  ข้อความ"
    out = prepare_text(raw)
    assert out == "มาตรา 37 ห้ามยกของหนัก\nหมวด 3\nการใช้แรงงานหญิง\nมาตรา 38 ห้ามนายจ้าง\nบทเฉพาะกาล\nมาตรา 39 ข้อความ"
    assert [(c, t) for c, t, _ in split_chapters(out)] == [
        ("00", "ก่อนหมวด 1"), ("03", "หมวด 3 การใช้แรงงานหญิง"), ("99", "บทเฉพาะกาล"),
    ]


def test_prepare_text_accepts_sub_numbered_sections_in_order():
    raw = "มาตรา ๔ ข้อความ\nมาตรา ๔/๑ ข้อความ\nมาตรา ๕ ข้อความ\nมาตรา ๔ อ้างถึง"
    assert prepare_text(raw) == "มาตรา 4 ข้อความ\nมาตรา 4/1 ข้อความ\nมาตรา 5 ข้อความ มาตรา 4 อ้างถึง"


def test_prepare_text_drops_gazette_page_number_with_displaced_tone_mark():
    raw = "การให้บริการนั้น ๆ ้\nหนา   ๖๐่\nเลม   ๑๓๖   ตอนที่   ๖๙    ก\nราชกิจจานุเบกษา\n๒๗   พฤษภาคม   ๒๕๖๒\n\nเจ้าของข้อมูล"
    assert prepare_text(raw) == "การให้บริการนั้น ๆ เจ้าของข้อมูล"
