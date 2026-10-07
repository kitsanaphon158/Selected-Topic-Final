# ผู้ช่วยกฎหมายสำหรับคนทำงาน (Labor + PDPA RAG) — Design Spec

วันที่: 2026-10-07
สถานะ: รอผู้ใช้ตรวจ

## 1. เป้าหมาย

Web Application แชตบอตบน Streamlit Community Cloud ที่ตอบคำถามประชาชนเรื่อง
**สิทธิลูกจ้าง** (พ.ร.บ.คุ้มครองแรงงาน พ.ศ. 2541 และฉบับแก้ไข) และ
**ข้อมูลส่วนบุคคล** (พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562) จากตัวบทกฎหมายเท่านั้น
ด้วยเทคนิค RAG — งานส่งรายวิชา

### เกณฑ์ความสำเร็จ (จากโจทย์)
- โค้ดอยู่บน GitHub repo ส่วนตัว และเปิด URL Streamlit Cloud จากเบราว์เซอร์ที่ไม่ได้ล็อกอินได้จริง
- ไม่มี API key ใดใน repo (เรียกผ่าน `st.secrets["GROQ_API_KEY"]`, `.streamlit/secrets.toml` อยู่ใน `.gitignore`)
- มี `app.py`, `requirements.txt`, `README.md`, `data/` (≥10 ไฟล์ และ ≥15,000 ตัวอักษร), `test_questions.csv` (≥10 ข้อ มีข้อไม่มีคำตอบ ≥2)
- ครบ 5 เทคนิค: Loading & Chunking, Embedding & Vector Search (FAISS), Prompt Engineering, LLM ผ่าน API, Chatbot Interface ที่คุยต่อเนื่องและแสดงอ้างอิงทุกครั้ง
- ตอบโดยอิงเอกสาร อ้างแหล่งที่มา และตอบ "ไม่พบข้อมูล" เมื่อเอกสารไม่มีคำตอบ

### ขอบเขตที่ไม่ทำ (YAGNI)
- ไม่มีระบบอัปโหลดเอกสารโดยผู้ใช้, ไม่มีบัญชีผู้ใช้, ไม่เก็บประวัติแชตข้ามเซสชัน
- ไม่รวมกฎหมายลำดับรอง (กฎกระทรวง ประกาศ) และเอกสารอธิบายภาษาง่าย — ใช้ตัวบท พ.ร.บ. เท่านั้น

## 2. การตัดสินใจหลัก

| เรื่อง | เลือก | เหตุผล |
|---|---|---|
| หัวข้อ | แรงงาน + PDPA รวมกัน | สร้างสรรค์กว่าหัวข้อเดี่ยว, คำถามแรงงานมีคำตอบเป็นตัวเลขตรวจง่าย |
| แหล่งเอกสาร | ตัวบทจากแหล่งทางการ (กฤษฎีกา / ราชกิจจานุเบกษา) ฉบับรวมแก้ไข | อ้างอิงระดับมาตราได้แม่น |
| Chunking | ตามมาตรา (regex `มาตรา \d+`), มาตรา >800 ตัวอักษรแบ่งย่อย overlap ~100 | 1 chunk = 1 หน่วยกฎหมายที่อ้างได้ |
| Embedding | `intfloat/multilingual-e5-small` (prefix `query: ` / `passage: `), normalize | เล็กพอสำหรับ Streamlit Cloud, รองรับไทย/อังกฤษ |
| Vector DB | FAISS `IndexFlatIP` (cosine จากเวกเตอร์ normalize) | ข้อมูลไม่กี่ร้อย chunk ไม่ต้องใช้ index ประมาณค่า |
| Keyword search | BM25 (`rank-bm25`) ตัดคำด้วย `pythainlp` (`newmm`) | จับคำเฉพาะ/เลขมาตราได้ดีกว่าเวกเตอร์ |
| Fusion | Reciprocal Rank Fusion (k=60) | รวมสองอันดับโดยไม่ต้องปรับสเกลคะแนน |
| LLM | Groq `llama-3.3-70b-versatile`, temperature 0.1 | ฟรี เร็ว ภาษาไทยพอใช้ |
| คุยต่อเนื่อง | Query rewriting ด้วย LLM ก่อนค้น (ข้ามเมื่อเป็นคำถามแรก) | คำถามต่อเนื่องอย่าง "แล้วถ้า 5 ปีล่ะ" ค้นตรง ๆ ไม่ได้ |

## 3. โครงสร้างไฟล์

```
labor-pdpa-rag/
├── app.py                          # Streamlit UI เท่านั้น
├── rag/
│   ├── __init__.py
│   ├── loader.py                   # load → clean → chunk
│   ├── retriever.py                # embed + FAISS + BM25 + RRF + threshold
│   ├── prompt.py                   # system prompt, context builder, rewrite prompt
│   └── llm.py                      # Groq client wrapper
├── data/                           # ตัวบท .txt แยกไฟล์ละหมวด (~15–20 ไฟล์)
├── scripts/fetch_laws.py           # ดึง/แปลงตัวบทจากแหล่งทางการ (ทำซ้ำได้)
├── tests/                          # pytest
├── evaluate.py                     # รัน test_questions.csv → hit rate / refusal
├── test_questions.csv
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/secrets.toml.example
```

## 4. ส่วนประกอบและ interface

### 4.1 `rag/loader.py`
- `load_documents(data_dir: str) -> list[Document]` — อ่านทุก `.txt`, แยกหัวไฟล์ 3 บรรทัด (`LAW:`, `CHAPTER:`, `SOURCE:`) เป็น metadata
- `clean_text(text: str) -> str` — ลบเลขหน้า เชิงอรรถ ช่องว่างซ้ำ, แปลงเลขไทย→อารบิก, `pythainlp.util.normalize`
- `chunk_by_section(doc, max_chars=800, overlap=100) -> list[Chunk]`
- `Chunk` = `{id, text, law, chapter, section, source_file, source_url}`; `law` ∈ {`labor`, `pdpa`}
- ข้อความก่อนมาตราแรกของไฟล์ (ถ้ามี) เป็น chunk ที่ `section=None`

### 4.2 `rag/retriever.py`
- `class Retriever(chunks, model_name)` — สร้าง embedding + FAISS + BM25 ตอน init
- `search(query, k=4, laws={"labor","pdpa"}) -> list[Hit]` — ค้น FAISS top-8 และ BM25 top-8 ภายใน `laws` ที่เลือก → RRF → top-k
- `Hit` = `{chunk, rrf_score, cosine}`
- `is_relevant(hits, threshold) -> bool` — `max(cosine) >= threshold`
- ค่า `threshold` เริ่มที่ 0.80 แล้ว**ปรับจาก `evaluate.py`** บันทึกค่าสุดท้ายใน README

### 4.3 `rag/prompt.py`
- `SYSTEM_PROMPT` — กฎ: ตอบจาก `<context>` เท่านั้น, อ้าง `[n]` + ชื่อ พ.ร.บ. + มาตรา,
  ถ้าไม่มีคำตอบให้ตอบเพียง `ไม่พบข้อมูลในเอกสารที่มีอยู่`, ตอบภาษาเดียวกับคำถาม,
  ปิดท้ายว่าเป็นข้อมูลเบื้องต้นไม่ใช่คำปรึกษากฎหมาย
- `build_context(hits) -> str` — รูปแบบ `[n] <ชื่อ พ.ร.บ.> — <หมวด> — มาตรา <x>\n<ตัวบท>` ห่อใน `<context>`
- `REWRITE_PROMPT` — "เขียนคำถามล่าสุดใหม่ให้สมบูรณ์ในตัวเองจากประวัติแชต ตอบเฉพาะคำถาม"
- `NOT_FOUND = "ไม่พบข้อมูลในเอกสารที่มีอยู่"`

### 4.4 `rag/llm.py`
- `get_client(api_key)`, `rewrite_query(client, history, question) -> str`,
  `generate_answer(client, history, question, context) -> str`
- ส่งประวัติแชตล่าสุด 3 รอบ (6 ข้อความ)

### 4.5 `app.py`
- `@st.cache_resource` โหลด chunks + `Retriever` ครั้งเดียว
- Sidebar: ชื่อแอป, checkbox กฎหมาย 2 ตัว, slider จำนวนมาตรา (2–8, ค่าเริ่ม 4), ปุ่มล้างแชต, คำเตือนไม่ใช่คำปรึกษากฎหมาย, แหล่งข้อมูล
- หน้าหลัก: ปุ่มคำถามตัวอย่าง 4 ปุ่ม (แสดงเมื่อแชตว่าง), `st.chat_message` + `st.chat_input`, ประวัติใน `st.session_state`
- ใต้ทุกคำตอบ: expander "📚 แหล่งอ้างอิง (n)" (ชื่อ พ.ร.บ., มาตรา, cosine, ตัวบทเต็ม) และ expander "🔎 คำค้นที่ใช้"
- ไม่มีตรรกะ RAG ใน `app.py`

## 5. Data flow ต่อหนึ่งคำถาม

1. validate input (ว่าง / >1,000 ตัวอักษร → แจ้งเตือน ไม่ประมวลผล)
2. ถ้ามีประวัติ → `rewrite_query` ; ไม่มี → ใช้คำถามเดิม
3. `retriever.search(rewritten, k, laws)`
4. `is_relevant` ไม่ผ่าน → ตอบ `NOT_FOUND` ทันที ไม่เรียก LLM, ไม่แสดงแหล่งอ้างอิง
5. ผ่าน → `generate_answer` ด้วย `SYSTEM_PROMPT` + `build_context(hits)` + ประวัติ 3 รอบ
6. ถ้าคำตอบขึ้นต้นด้วย `ไม่พบข้อมูล` → ไม่แสดงแหล่งอ้างอิง; มิฉะนั้นแสดงทุกมาตราที่ส่งเป็น context
7. บันทึกคำตอบ + hits + คำค้นลง `session_state`

## 6. ข้อมูล (`data/`)

- ดึงจากเว็บกฤษฎีกา/ราชกิจจานุเบกษา **ฉบับรวมแก้ไขล่าสุด** ด้วย `scripts/fetch_laws.py`
- แยกไฟล์ละหมวด ตั้งชื่อ `labor_chXX_<topic>.txt`, `pdpa_chXX_<topic>.txt` (บทเฉพาะกาลอาจเป็นไฟล์แยก)
- หัวไฟล์:
  ```
  LAW: พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541
  CHAPTER: หมวด 11 ค่าชดเชย
  SOURCE: <URL>
  ```
- ห้ามพิมพ์ตัวบทจากความจำ — ถ้าดึงอัตโนมัติไม่ได้ (บล็อก / PDF สแกน) ให้หยุดแจ้งผู้ใช้
- ตรวจหลังสร้าง: จำนวนไฟล์ ≥10, รวม ≥15,000 ตัวอักษร, ตรวจว่ามาตราต่อเนื่องไม่หาย

## 7. `test_questions.csv`

- คอลัมน์: `id,question,expected_answer,expected_source,answerable`
- 12 ข้อ: answerable แรงงาน 5, PDPA 4, อังกฤษ 1; ไม่มีคำตอบ 2 (เช่น ค่าแรงขั้นต่ำปีปัจจุบัน, อัตราภาษีเงินได้)
- `expected_answer` และ `expected_source` ยกจากตัวบทใน `data/` โดยตรงและตรวจซ้ำ

## 8. การจัดการข้อผิดพลาด

| สถานการณ์ | พฤติกรรม |
|---|---|
| ไม่มี `GROQ_API_KEY` ใน secrets | `st.error` พร้อมวิธีตั้งค่า แล้ว `st.stop()` (ไม่แสดง traceback) |
| Groq error / 429 | ข้อความ "ระบบ AI ไม่ว่าง ลองใหม่อีกครั้ง" + ยังแสดงมาตราที่ค้นได้ |
| rewrite ล้มเหลว | ใช้คำถามเดิมค้นต่อ |
| input ว่าง / ยาวเกิน | แจ้งเตือน ไม่ประมวลผล |
| โหลดครั้งแรกช้า | spinner "กำลังเตรียมคลังกฎหมาย (ครั้งแรกเท่านั้น)" |

## 9. การทดสอบ

- **pytest** (ไม่เรียก Groq จริง):
  - loader: ตัดตามมาตราถูก, metadata ครบ, เลขไทย→อารบิก, มาตรายาวถูกแบ่งย่อย
  - retriever: ค้น "มาตรา 118" ได้มาตรา 118 ของแรงงานอันดับ 1; ตัวกรอง `laws` ทำงาน; คำถามนอกเรื่องได้ cosine ต่ำกว่าคำถามในเรื่อง
  - prompt: `build_context` ติดป้าย `[n]` + ชื่อ พ.ร.บ. + มาตรา ถูกต้อง
  - llm/app flow: mock client — กรณี threshold ไม่ผ่านต้องไม่เรียก LLM
- **`evaluate.py`** (เรียก Groq จริง): retrieval hit rate@k, อัตราปฏิเสธถูกต้องในข้อ unanswerable, ใช้ปรับ threshold, ผลสรุปใส่ README

## 10. Deploy และสิ่งที่ต้องส่ง

- `requirements.txt`: `streamlit`, `sentence-transformers`, `faiss-cpu`, `rank-bm25`, `pythainlp`, `groq` + torch CPU (`--extra-index-url https://download.pytorch.org/whl/cpu`)
- `.gitignore` มี `.streamlit/secrets.toml`; มี `secrets.toml.example` ที่ไม่มีค่าจริง
- ก่อน push: `git grep -i "gsk_"` ต้องไม่พบอะไร
- **ผู้ใช้ทำเอง:** สร้าง GitHub repo ส่วนตัว (หรือ login `gh` ให้ผมช่วย), ตั้ง Secrets บน Streamlit Cloud, แคปหน้าจอ, ส่ง Colab
- **ผมเตรียมให้:** คำสั่ง push/deploy, ข้อความสำหรับ Colab (หัวข้อ, URL ทั้งสอง), รายการภาพหน้าจอที่ควรแคปพร้อมคำอธิบายสำหรับ PDF
- หลัง deploy: เปิด URL ด้วย in-app browser (ไม่ล็อกอิน) ทดสอบคำถามที่มีคำตอบ, ไม่มีคำตอบ, และคำถามต่อเนื่อง

## 11. README.md ต้องมี
วิธีใช้งาน (local + deploy), แนวคิดของ domain, แหล่งที่มาของเอกสาร (URL ทุกไฟล์), สถาปัตยกรรม RAG และเหตุผลการเลือก chunking/retrieval/threshold, ผล `evaluate.py`, ตัวอย่าง Prompt ที่ใช้สั่ง AI ระหว่างพัฒนา
