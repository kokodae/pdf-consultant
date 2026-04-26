from typing import Optional
from backend.config import GIGACHAT_API_KEY, MOCK_MODE, GIGACHAT_AUTH_URL

# LangChain imports
try:
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_gigachat import GigaChat as LangchainGigaChat
    LANGCHAIN_AVAILABLE = True
    GIGACHAT_LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    GIGACHAT_LANGCHAIN_AVAILABLE = False
    SystemMessage = HumanMessage = None
    LangchainGigaChat = None

from backend.gigachat.clients import GigaChatREST, GigaChatRESTCorp

def get_giga_llm():
    """Создание клиента GigaChat для корпоративного ключа"""
    if not GIGACHAT_API_KEY:
        print("⚠️ GigaChat API key missing")
        return None
    
    if MOCK_MODE:
        print("🔧 Работа в мок-режиме (GigaChat отключен)")
        return None

    corp_scopes = [
        "GIGACHAT_API_CORP",
        "GIGACHAT_API_CORP GIGACHAT_API_PERS",
        "https://api.sberbank.ru/gigachat/corp",
        ""
    ]

    for scope in corp_scopes:
        try:
            print(f"🔄 Пробуем scope: '{scope}' для CORP ключа")
            
            if GIGACHAT_LANGCHAIN_AVAILABLE and LangchainGigaChat:
                auth_urls = [
                    GIGACHAT_AUTH_URL,
                    "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                    "https://api.sberbank.ru/gigachat/oauth"
                ]
                
                for auth_url in auth_urls:
                    try:
                        llm = LangchainGigaChat(
                            credentials=GIGACHAT_API_KEY,
                            auth_url=auth_url,
                            scope=scope if scope else None,
                            model="GigaChat-2-Max",
                            verify_ssl_certs=False,
                            temperature=0.1,
                            profanity_check=False,
                            timeout=30
                        )
                        test_messages = [SystemMessage(content="Тест"), HumanMessage(content="OK")]
                        llm.invoke(test_messages)
                        print(f"✅ GigaChat CORP работает с auth_url={auth_url}, scope='{scope}'")
                        return llm
                    except Exception as e:
                        print(f"⚠️ auth_url={auth_url}, scope='{scope}' не подошел: {str(e)[:80]}")
                        continue
        except Exception as e:
            print(f"❌ Ошибка с scope '{scope}': {str(e)[:100]}")
            continue

    print("⚠️ Пробуем прямой REST для CORP ключа...")
    try:
        llm = GigaChatRESTCorp(GIGACHAT_API_KEY)
        print("✅ GigaChat REST для CORP инициализирован")
        return llm
    except Exception as e:
        print(f"❌ REST тоже не работает: {e}")

    print("❌ Не удалось подключиться к GigaChat с CORP ключом")
    return None