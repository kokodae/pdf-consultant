"""
Модуль эмбеддингов для модели EmbeddingsGigaR от GigaChat.
Официальная документация:
- https://developers.sber.ru/docs/ru/gigachat/models/embeddings-giga-r
- https://developers.sber.ru/docs/ru/gigachat/guides/embeddings
"""
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from langchain_gigachat.embeddings import GigaChatEmbeddings

from backend.config import (
    GIGACHAT_API_KEY,
    GIGACHAT_AUTH_URL,
    GIGACHAT_SCOPE,
    MOCK_MODE
)


class EmbeddingsGigaR(Embeddings):
    """
    Реализация эмбеддингов через официальную модель EmbeddingsGigaR.
    
    Параметры модели:
    - Размер вектора: 1536
    - Контекст: до 2048 токенов на вход
    - Оптимизирована для retrieval-задач (поиск, RAG)
    """
    
    def __init__(
        self,
        credentials: Optional[str] = None,
        scope: Optional[str] = None,
        auth_url: Optional[str] = None,
        verify_ssl: bool = False,
        timeout: float = 30.0,
        model: str = "EmbeddingsGigaR",
        **kwargs
    ):
        self.credentials = credentials or GIGACHAT_API_KEY
        self.scope = scope or GIGACHAT_SCOPE
        self.auth_url = auth_url or GIGACHAT_AUTH_URL
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.model = model
        self.mock_mode = MOCK_MODE
        
        self._client: Optional[GigaChatEmbeddings] = None
        if not self.mock_mode and self.credentials:
            self._client = GigaChatEmbeddings(
                credentials=self.credentials,
                model=self.model,
                scope=self.scope,
                auth_url=self.auth_url,
                verify_ssl_certs=self.verify_ssl,
                timeout=self.timeout,
                **kwargs
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.mock_mode or not self._client:
            return [[0.0] * 1536 for _ in texts]
        return self._client.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        if self.mock_mode or not self._client:
            return [0.0] * 1536
        return self._client.embed_query(text)
    
    @staticmethod
    def with_retrieval_instruction(instruction: str) -> "EmbeddingsGigaR":
        instance = EmbeddingsGigaR()
        original_embed_query = instance.embed_query
        
        def wrapped_embed_query(text: str) -> List[float]:
            instructed_text = f"{instruction}\n{text}"
            return original_embed_query(instructed_text)
        
        instance.embed_query = wrapped_embed_query
        return instance
    
    @property
    def dimension(self) -> int:
        return 1536