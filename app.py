"""ผู้ช่วยกฎหมายสำหรับคนทำงาน — แชตบอต RAG (พ.ร.บ.คุ้มครองแรงงาน + PDPA)"""
import streamlit as st

from rag.llm import get_client
from rag.loader import LAWS, load_chunks
from rag.pipeline import RagResult, answer_question, validate_question
from rag.prompt import source_label
from rag.retriever import DEFAULT_THRESHOLD, Retriever

DATA_DIR = "data"
EXAMPLES = [
    "ทำงาน 3 ปีถูกเลิกจ้าง ได้ค่าชดเชยเท่าไร",
    "ลาป่วยได้ค่าจ้างปีละกี่วัน",
    "เก็บข้อมูลพนักงาน ต้องขอความยินยอมไหม",
    "ข้อมูลรั่วไหล ต้องแจ้งภายในกี่ชั่วโมง",
]
AI_BUSY = "ระบบ AI ไม่ว่างในขณะนี้ ลองใหม่อีกครั้ง — ด้านล่างคือมาตราที่ค้นได้"
AVATARS = {"user": ":material/person:", "assistant": ":material/gavel:"}

st.set_page_config(page_title="ผู้ช่วยกฎหมายคนทำงาน", page_icon=":material/gavel:", layout="wide")


@st.cache_resource(show_spinner="กำลังเตรียมคลังกฎหมาย (ครั้งแรกเท่านั้น)...")
def load_retriever() -> Retriever:
    return Retriever(load_chunks(DATA_DIR))


@st.cache_resource
def load_client(api_key: str):
    return get_client(api_key)


def get_api_key() -> str | None:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:  # ไม่มีไฟล์ secrets หรือไม่มี key
        return None


def to_message(result: RagResult) -> dict:
    return {
        "role": "assistant",
        "content": result.answer if result.error is None else AI_BUSY,
        "sources": [
            {
                "label": source_label(h.chunk),
                "cosine": h.cosine,
                "text": h.chunk.section_text,
                "url": h.chunk.source_url,
            }
            for h in result.hits
        ],
        "search_query": result.search_query,
    }


def render_message(msg: dict) -> None:
    st.markdown(msg["content"])
    if msg["role"] != "assistant":
        return
    sources = msg.get("sources", [])
    if sources:
        with st.expander(f"แหล่งอ้างอิง ({len(sources)} มาตรา)", icon=":material/menu_book:"):
            for n, s in enumerate(sources, 1):
                with st.container(border=True):
                    st.markdown(f"**[{n}] {s['label']}**")
                    st.caption(f"ความคล้ายกับคำถาม {s['cosine']:.2f}")
                    st.markdown(s["text"].replace("\n", "  \n"))
                    st.caption(f"[เปิดตัวบทต้นฉบับ]({s['url']})")
    else:
        st.caption(":material/info: ไม่มีแหล่งอ้างอิง เพราะไม่พบตัวบทที่ตอบคำถามนี้ได้")
    if msg.get("search_query"):
        with st.expander("คำค้นที่ระบบใช้", icon=":material/search:"):
            st.code(msg["search_query"], language=None)


with st.sidebar:
    st.title(":material/gavel: ผู้ช่วยกฎหมายคนทำงาน")
    st.caption("ตอบจากตัวบท พ.ร.บ.คุ้มครองแรงงาน และ PDPA ด้วยเทคนิค RAG")
    st.subheader("ขอบเขตกฎหมาย")
    laws = {code for code, name in LAWS.items() if st.checkbox(name, value=True, key=f"law_{code}")}
    k = st.slider("จำนวนมาตราที่ค้น", min_value=2, max_value=6, value=4)
    if st.button("เริ่มแชตใหม่", icon=":material/refresh:", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.warning("ข้อมูลเบื้องต้นจากตัวบทกฎหมาย ไม่ใช่คำปรึกษาทางกฎหมาย", icon=":material/warning:")
    st.caption("แหล่งข้อมูล: ตัวบทฉบับรวมแก้ไขจากแหล่งทางการ — ดูรายละเอียดใน README ของโปรเจกต์")

st.header("ถามเรื่องสิทธิลูกจ้าง & ข้อมูลส่วนบุคคล")
st.caption("ตอบจากตัวบทกฎหมายเท่านั้น พร้อมบอกเลขมาตราที่ใช้ตอบทุกครั้ง")

api_key = get_api_key()
if not api_key:
    st.error(
        icon=":material/key_off:",
        body="ยังไม่ได้ตั้งค่า GROQ_API_KEY — บนเครื่อง: สร้าง `.streamlit/secrets.toml` "
        "บน Streamlit Cloud: Settings → Secrets แล้วใส่ `GROQ_API_KEY = \"...\"`"
    )
    st.stop()

retriever = load_retriever()
client = load_client(api_key)
st.session_state.setdefault("messages", [])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        render_message(msg)

pending = None
if not st.session_state.messages:
    st.markdown("**ลองถามคำถามตัวอย่าง:**")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, key=f"example_{i}", icon=":material/help:", width="stretch", disabled=not laws):
            pending = example

if not laws:
    st.info("เลือกกฎหมายอย่างน้อย 1 ฉบับที่แถบด้านซ้าย", icon=":material/info:")
typed = st.chat_input("พิมพ์คำถามเกี่ยวกับสิทธิลูกจ้างหรือ PDPA...", disabled=not laws)
question = typed or pending

if question:
    problem = validate_question(question)
    if problem:
        st.warning(problem, icon=":material/warning:")
        st.stop()
    history = list(st.session_state.messages)
    with st.chat_message("user", avatar=AVATARS["user"]):
        render_message({"role": "user", "content": question})
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("กำลังค้นตัวบท..."):
            result = answer_question(
                question, history, retriever, client, k=k, laws=laws, threshold=DEFAULT_THRESHOLD
            )
            # บันทึกคำถาม+คำตอบพร้อมกันก่อนเรียก st.* ตัวถัดไป — ถ้าผู้ใช้แตะ widget ระหว่างรอ
            # Streamlit จะ rerun ที่การเรียก st.* ถัดไป และคำตอบที่ได้แล้วจะไม่หาย
            st.session_state.messages += [{"role": "user", "content": question}, to_message(result)]
    st.rerun()  # วาดหน้าใหม่จากประวัติ เพื่อซ่อนปุ่มคำถามตัวอย่างหลังเริ่มแชต  # วาดหน้าใหม่จากประวัติ เพื่อซ่อนปุ่มคำถามตัวอย่างหลังเริ่มแชต
