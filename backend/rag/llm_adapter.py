from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import List, Any, Optional
from backend.gigachat.factory import get_giga_llm

class GigaChatLangChainAdapter(BaseChatModel):
    """Адаптер, совместимый с langchain-core >=0.2.0"""
    model: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = get_giga_llm()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        if not self.model:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="⚠️ LLM не инициализирована."))])

        try:
            # LangChain-клиент
            if hasattr(self.model, 'invoke') and 'langchain' in str(type(self.model)).lower():
                resp = self.model.invoke(messages)
                content = resp.content if hasattr(resp, 'content') else str(resp)
            # REST-клиент
            else:
                content = self.model.invoke(messages)
                
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
        except Exception as e:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"❌ Ошибка LLM: {str(e)}"))])

    @property
    def _llm_type(self) -> str:
        return "gigachat_adapter"