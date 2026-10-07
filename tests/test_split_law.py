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
