import json
import re
from typing import List, TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field


# ==================== SCHEMAS ====================
class Citation(BaseModel):
    text: str = Field(description="Полный текст цитаты без префиксов [Стр.X]")
    source_page: int = Field(description="Номер страницы")

class RAGResponse(BaseModel):
    answer: str = Field(description="Ответ на вопрос пользователя")
    citations: List[Citation] = Field(description="Список подтверждающих цитат")

class RAGState(TypedDict):
    messages: Annotated[list, add_messages]
    retrieved_docs: List[Document]
    citations: List[dict]


# ==================== PROMPTS ====================
QUERY_GEN_PROMPT = """Generate 3 different versions of the user's question to improve document retrieval.
Each version should focus on different aspects, synonyms, or phrasing.
Output ONLY the questions, separated by newlines. Do not add explanations.
Original question: {question}"""

SYSTEM_PROMPT = (
    "Ты - экспертный ассистент по анализу документов.\n"
    "Отвечай ТОЛЬКО на основе предоставленного контекста.\n"
    "Если информации недостаточно, скажи об этом прямо.\n"
    "Верни результат СТРОГО в формате JSON с полями 'answer' и 'citations'.\n"
    "ВАЖНО: В поле 'text' каждой цитаты верни ПОЛНЫЙ ТЕКСТ фрагмента из контекста.\n"
    "НЕ добавляй [Стр.X] или номера страниц внутрь текста цитаты — номер указывается только в поле 'source_page'.\n"
    "Пример: {\"text\": \"Параллельная архитектура ориентирована на исполнение алгоритмов...\", \"source_page\": 8}\n"
    "Не добавляй никаких пояснений, комментариев или markdown вне JSON."
).replace("{", "{{").replace("}", "}}")


# ==================== PIPELINE ====================
class RAGPipeline:
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        self.query_chain = ChatPromptTemplate.from_template(QUERY_GEN_PROMPT) | llm | StrOutputParser()
        self.graph = self._build_graph()

    def _retrieve_node(self, state: RAGState) -> dict:
        query = state["messages"][-1].content
        try:
            raw = self.query_chain.invoke({"question": query})
            queries = [q.strip() for q in raw.split("\n") if 3 < len(q.strip())]
        except Exception:
            queries = []
        queries.append(query)

        all_docs = []
        for q in queries:
            try:
                all_docs.extend(self.vectorstore.similarity_search(q, k=4))
            except Exception:
                continue

        seen, unique = set(), []
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique.append(doc)
        return {"retrieved_docs": unique[:10]}

    def _clean_citation_text(self, text: str) -> str:
        """Удаляет префиксы вида [Стр.2], Стр 2:, [Стр 2] и т.п. из начала цитаты"""
        text = re.sub(r'^\s*[\[\(]?[Сс]тр\.?\s*\.?\]?(\d+)[\]\)]?\s*:?\.?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*[\[\(]?[Сс]тр\.?\s*\.?\]?\d+[\]\)]?\s*:?\.?\s*', ' ', text, flags=re.IGNORECASE)
        return text.strip()

    def _generate_node(self, state: RAGState) -> dict:
        context = "\n\n".join([
            f"[Стр.{d.metadata.get('page', '?')}] {d.page_content}" for d in state["retrieved_docs"]
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Контекст:\n{context}\n\nВопрос: {question}")
        ])

        raw = (prompt | self.llm | (lambda m: m.content if hasattr(m, 'content') else str(m))).invoke({
            "context": context, "question": state["messages"][-1].content
        })

        answer_text = raw.strip()
        citations = []

        try:
            text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE).strip()

            start, end = text.find('{'), text.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(text[start:end])

                ans = data.get("answer", "").strip()
                if ans:
                    answer_text = ans
                else:
                    for v in data.values():
                        if isinstance(v, str) and len(v) > 20:
                            answer_text = v.strip()
                            break

                raw_cits = data.get("citations", data.get("citation", []))
                if not isinstance(raw_cits, list):
                    raw_cits = [raw_cits] if raw_cits else []
                
                for c in raw_cits:
                    txt = ""
                    pg = 0
                    
                    if isinstance(c, str):
                        page_match = re.search(r'[Сс]тр\.?\s*\.?(\d+)', c)
                        pg = int(page_match.group(1)) if page_match else 0
                        txt = self._clean_citation_text(c)
                    
                    elif isinstance(c, dict):
                        txt = c.get("text", c.get("quote", c.get("content", ""))) or ""
                        pg = c.get("source_page", c.get("page", c.get("page_number", 0))) or 0
                        txt = self._clean_citation_text(str(txt))
                    
                    # Фоллбэк: если текст пустой или слишком короткий, берём из контекста
                    if (not txt or len(txt) < 20) and pg:
                        for doc in state["retrieved_docs"]:
                            if doc.metadata.get("page") == pg:
                                txt = doc.page_content[:300]
                                if len(txt) > 297: txt += "..."
                                txt = self._clean_citation_text(txt)
                                break
                    
                    if txt and len(txt) >= 15:
                        citations.append({"text": txt, "source_page": int(pg) if pg else 0})
                        
        except Exception as e:
            print(f"⚠️ JSON parse fallback: {e}")

        if len(answer_text) > 4000:
            answer_text = answer_text[:3997] + "..."

        return {"messages": [AIMessage(content=answer_text)], "citations": citations}

    def _build_graph(self):
        wf = StateGraph(RAGState)
        wf.add_node("retrieve", self._retrieve_node)
        wf.add_node("generate", self._generate_node)
        wf.add_edge(START, "retrieve")
        wf.add_edge("retrieve", "generate")
        wf.add_edge("generate", END)
        return wf.compile()

    def invoke(self, state: dict) -> dict:
        return self.graph.invoke(state)
