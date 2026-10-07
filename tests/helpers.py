from rag.loader import Chunk

LAW_NAMES = {
    "labor": "พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541",
    "pdpa": "พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562",
}


def make_chunk(
    law: str, section: str | None, text: str, chapter: str = "หมวดทดสอบ", part: int = 0, section_text: str | None = None
) -> Chunk:
    return Chunk(
        id=f"{law}_test#{section or 'pre'}-{part}",
        text=text,
        section_text=section_text or text,
        law=law,
        law_name=LAW_NAMES[law],
        chapter=chapter,
        section=section,
        source_file=f"{law}_test.txt",
        source_url="https://example.go.th/law",
    )
