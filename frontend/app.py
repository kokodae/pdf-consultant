import os
import sys
import shutil
import gc
import time
import streamlit as st
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage, AIMessage
from backend.config import CHROMA_PERSIST_DIR
from backend.rag.embeddings import EmbeddingsGigaR
from backend.rag.llm_adapter import GigaChatLangChainAdapter
from backend.rag.graph import RAGPipeline
from backend.utils.pdf_processor import process_pdf

st.set_page_config(page_title="GigaRAG PDF Chat", layout="centered", initial_sidebar_state="collapsed")

# ================= STYLES =================
st.markdown("""
<style>
    :root {
        --bg-page: #FFFFFF;
        --bg-element: #F0F0F0;
        --txt-main: #000000;
        --txt-secondary: #444444;
        --border: #E0E0E0;
        --accent: #FFB6C1;
        --accent-hover: #FF9EB0;
    }
    .stApp, body, .main, .block-container, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: var(--bg-page) !important; background-image: none !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, div, label, li, button, input, textarea, a {
        color: var(--txt-main) !important;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown div { color: var(--txt-main) !important; }
    .stButton > button {
        background-color: var(--accent) !important; color: #000000 !important;
        border: 1px solid var(--border) !important; border-radius: 4px; font-weight: 500; transition: all 0.2s ease;
    }
    .stButton > button:hover { background-color: var(--accent-hover) !important; }
    .stTextInput > div > div > input, .stChatInput > div > div > input,
    .stChatMessage, .streamlit-expanderHeader, .streamlit-expanderContent {
        background-color: var(--bg-element) !important; color: var(--txt-main) !important;
        border: 1px solid var(--border) !important;
    }
    .stChatMessage {
        border-left: 3px solid var(--accent) !important; border-radius: 0 6px 6px 0; margin-bottom: 0.5rem;
    }
    .streamlit-expanderHeader { font-weight: 600 !important; }
    .streamlit-expanderContent { border-top: none !important; color: var(--txt-secondary) !important; }
    .streamlit-expanderContent blockquote {
        background: transparent !important; color: var(--txt-main) !important;
        border-left: 3px solid var(--accent) !important; padding: 0.5rem 0.8rem; margin: 0.4rem 0;
        font-style: normal !important; border-radius: 0 4px 4px 0;
    }
    .stProgress > div > div > div { background-color: var(--accent) !important; }
    hr { border-color: var(--border) !important; }
    a { color: var(--accent-hover) !important; }
    .stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ================= STATE =================
if "mode" not in st.session_state: st.session_state.mode = "upload"
if "pipeline" not in st.session_state: st.session_state.pipeline = None
if "history" not in st.session_state: st.session_state.history = []
if "filename" not in st.session_state: st.session_state.filename = None
if "book_title" not in st.session_state: st.session_state.book_title = None

def _safe_clear(dir_path: str):
    p = Path(dir_path)
    if not p.exists(): return
    gc.collect()
    time.sleep(0.1)
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)

def reset_app():
    st.session_state.update({"mode": "upload", "pipeline": None, "history": [], "filename": None, "book_title": None})
    _safe_clear(CHROMA_PERSIST_DIR)

def render_citations(citations: list):
    """Рендер цитат: номер страницы только в заголовке, текст — внутри"""
    if not citations: return
    st.markdown("---")
    st.subheader("Источники")
    for i, c in enumerate(citations, 1):
        page = c.get('source_page', '?')
        text = c.get('text', '').strip()
        # Заголовок: [1] Страница 27
        # Содержимое: только текст цитаты, без дублирования номера
        with st.expander(f"[{i}] Страница {page}"):
            if text:
                st.markdown(f"> {text}")
            else:
                st.markdown("> _Текст цитаты не извлечён_")

def _ai_title(text: str, llm) -> str:
    try:
        prompt = (
            "Извлеки точное название книги или отчёта из начала текста. "
            "Верни ТОЛЬКО название, без кавычек, авторов и пояснений. "
            "Если название неочевидно, верни 'Документ без названия'.\n"
            f"Текст:\n{text[:600]}"
        )
        r = llm.invoke([HumanMessage(content=prompt)])
        t = r.content.strip().strip('"\'').strip()
        return t if 2 < len(t) < 120 else "Документ без названия"
    except:
        return "Документ без названия"

# ================= UPLOAD =================
if st.session_state.mode == "upload":
    st.title("Загрузка документа")
    st.caption("Поддерживаются PDF 100-200 страниц. Индексация займет 15-40 секунд.")
    uploaded = st.file_uploader("Выберите PDF-файл", type=["pdf"])
    
    if uploaded:
        temp = f"./temp_{uploaded.name}"
        with open(temp, "wb") as f: f.write(uploaded.getbuffer())
        st.session_state.filename = uploaded.name
        
        with st.spinner("Обработка документа..."):
            progress = st.progress(0, text="Инициализация EmbeddingsGigaR...")
            embeddings = EmbeddingsGigaR.with_retrieval_instruction(instruction="Дан вопрос, необходимо найти абзац текста с ответом")
            
            progress.progress(0.2, text="Чтение и чанкинг текста...")
            from pypdf import PdfReader
            reader = PdfReader(temp)
            first_txt = reader.pages[0].extract_text() if reader.pages else ""
            
            progress.progress(0.4, text="Индексация в ChromaDB...")
            vs = process_pdf(temp, embeddings)
            os.remove(temp)
            
            progress.progress(0.6, text="Инициализация GigaChat...")
            llm = GigaChatLangChainAdapter()
            
            progress.progress(0.8, text="ИИ определяет название книги...")
            st.session_state.book_title = _ai_title(first_txt, llm)
            
            progress.progress(0.9, text="Сборка LangGraph пайплайна...")
            st.session_state.pipeline = RAGPipeline(llm=llm, vectorstore=vs)
            st.session_state.mode = "chat"
            st.rerun()

# ================= CHAT =================
else:
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 1.2rem; margin-top: 0.5rem;">
        <h1 style="font-size: 1.6rem; color: var(--txt-main); margin: 0; font-weight: 500;">{st.session_state.book_title}</h1>
        <div style="font-size: 0.78rem; color: var(--txt-secondary); text-align: left; margin-top: 0.3rem; font-weight: 400; font-family: monospace;">
            📄 Файл: {st.session_state.filename}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col1: pass
    with col2:
        if st.button("Сброс диалога", use_container_width=True): reset_app(); st.rerun()
        
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"): render_citations(msg["citations"])
            
    if prompt := st.chat_input("Задайте вопрос по документу..."):
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Анализирую документ..."):
                lc_msgs = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.history]
                state = {"messages": lc_msgs, "retrieved_docs": [], "citations": []}
                res = st.session_state.pipeline.invoke(state)
                
                ans = res["messages"][-1].content
                cits = res.get("citations", [])
                
                st.markdown(ans)
                if cits: render_citations(cits)
                st.session_state.history.append({"role": "assistant", "content": ans, "citations": cits})
