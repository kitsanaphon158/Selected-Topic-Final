import pytest

from rag.retriever import Hit, Retriever, expand_query, is_relevant, tokenize
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


def test_search_returns_one_hit_per_section():
    full = "มาตรา 118 ค่าชดเชย ส่วนแรก ส่วนที่สอง"
    parts = [
        make_chunk("labor", "118", "มาตรา 118 ให้นายจ้างจ่ายค่าชดเชย ลูกจ้างทำงานครบสามปี", part=0, section_text=full),
        make_chunk("labor", "118", "(ต่อ) มาตรา 118 ลูกจ้างทำงานครบสิบปี ค่าชดเชยสามร้อยวัน", part=1, section_text=full),
        make_chunk("labor", "32", "มาตรา 32 ลาป่วย"),
    ]
    hits = Retriever(parts).search("ค่าชดเชย ทำงานสิบปี", k=3)
    assert [(h.chunk.section) for h in hits].count("118") == 1
    assert len(hits) == 2


def test_default_threshold_tuned_from_evaluation():
    from rag.retriever import DEFAULT_THRESHOLD
    assert DEFAULT_THRESHOLD == 0.81


def test_expand_query_adds_statutory_terms_for_everyday_words():
    assert "การละเมิดข้อมูลส่วนบุคคล" in expand_query("ข้อมูลลูกค้ารั่วไหล ต้องแจ้งใคร")
    assert "หยุดพักผ่อนประจำปี" in expand_query("ลาพักร้อนได้กี่วัน")
    assert expand_query("ลาป่วยได้กี่วัน") == "ลาป่วยได้กี่วัน"


def test_everyday_wording_finds_the_statutory_section():
    chunks = [
        make_chunk("pdpa", "37", "มาตรา 37 ผู้ควบคุมข้อมูลส่วนบุคคลมีหน้าที่แจ้งเหตุการละเมิดข้อมูลส่วนบุคคลแก่สำนักงานภายในเจ็ดสิบสองชั่วโมง"),
        make_chunk("pdpa", "23", "มาตรา 23 ในการเก็บรวบรวมข้อมูลส่วนบุคคล ผู้ควบคุมข้อมูลส่วนบุคคลต้องแจ้งให้เจ้าของข้อมูลส่วนบุคคลทราบ"),
        make_chunk("pdpa", "25", "มาตรา 25 ห้ามมิให้ผู้ควบคุมข้อมูลส่วนบุคคลเก็บรวบรวมข้อมูลส่วนบุคคลจากแหล่งอื่นที่ไม่ใช่จากเจ้าของข้อมูลส่วนบุคคลโดยตรง"),
    ]
    hits = Retriever(chunks).search("ข้อมูลลูกค้ารั่วไหล บริษัทต้องรีบบอกใคร", k=3)
    assert hits[0].chunk.section == "37"
