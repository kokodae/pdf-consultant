import time
import uuid
import requests
from urllib.parse import urlencode
from typing import Optional, List
from backend.config import GIGACHAT_AUTH_URL, GIGACHAT_API_URL, GIGACHAT_SCOPE
from backend.gigachat.auth import get_access_token

class GigaChatREST:
    """Прямой клиент для GigaChat REST API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.access_token = None
        self.token_expires_at = 0
        
    def _ensure_token(self) -> bool:
        current_time = time.time()
        if not self.access_token or self.token_expires_at - current_time < 60:
            self.access_token = get_access_token(self.api_key)
            if self.access_token:
                self.token_expires_at = current_time + 1800
                return True
            return False
        return True

    def invoke(self, messages: list) -> str:
        if not self._ensure_token():
            return "❌ Не удалось получить токен доступа к GigaChat API"
        
        try:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.access_token}'
            }
            
            formatted_messages = []
            for msg in messages:
                content = msg.content if hasattr(msg, 'content') else str(msg)
                role = "user"
                class_name = msg.__class__.__name__ if hasattr(msg, '__class__') else ''
                if 'System' in class_name:
                    role = "system"
                elif 'Human' in class_name:
                    role = "user"
                elif 'AI' in class_name:
                    role = "assistant"
                formatted_messages.append({"role": role, "content": content})
            
            data = {
                "model": "GigaChat",
                "messages": formatted_messages,
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            response = requests.post(
                f"{GIGACHAT_API_URL}/chat/completions",
                headers=headers,
                json=data,
                verify=False,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"❌ Ошибка API: {response.status_code}"
        except Exception as e:
            return f"❌ Ошибка: {str(e)}"


class GigaChatRESTCorp:
    """Специальный клиент для корпоративных ключей"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.access_token = None
        
    def _get_corp_token(self) -> Optional[str]:
        try:
            if len(self.api_key) > 100 and not self.api_key.startswith('Basic'):
                print("🔑 Ключ похож на Bearer токен, используем напрямую")
                return self.api_key
            
            auth_headers = [
                {'Authorization': f'Basic {self.api_key}'},
                {'Authorization': f'Bearer {self.api_key}'},
                {'Authorization': self.api_key},
            ]
            
            for headers in auth_headers:
                try:
                    rquid = str(uuid.uuid4())
                    headers.update({
                        'Content-Type': 'application/x-www-form-urlencoded',
                        'Accept': 'application/json',
                        'RqUID': rquid,
                    })
                    
                    data_options = [
                        {'scope': 'GIGACHAT_API_CORP'},
                        {'scope': 'GIGACHAT_API_PERS'},
                        {}
                    ]
                    
                    for data in data_options:
                        response = requests.post(
                            GIGACHAT_AUTH_URL,
                            headers=headers,
                            data=urlencode(data) if data else None,
                            verify=False,
                            timeout=30
                        )
                        if response.status_code == 200:
                            token = response.json().get('access_token')
                            print(f"✅ CORP токен получен")
                            return token
                except:
                    continue
            return None
        except Exception as e:
            print(f"❌ Ошибка получения CORP токена: {e}")
            return None

    def invoke(self, messages: list) -> str:
        token = self._get_corp_token()
        if not token:
            return "❌ Не удалось получить токен для CORP ключа"
        
        try:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Bearer {token}'
            }
            
            formatted_messages = []
            for msg in messages:
                content = msg.content if hasattr(msg, 'content') else str(msg)
                role = "user"
                class_name = msg.__class__.__name__ if hasattr(msg, '__class__') else ''
                if 'System' in class_name:
                    role = "system"
                elif 'AI' in class_name:
                    role = "assistant"
                formatted_messages.append({"role": role, "content": content})
            
            data = {
                "model": "GigaChat-2-Max",
                "messages": formatted_messages,
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            response = requests.post(
                f"{GIGACHAT_API_URL}/chat/completions",
                headers=headers,
                json=data,
                verify=False,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"❌ Ошибка CORP API: {response.status_code} - {response.text[:200]}"
        except Exception as e:
            return f"❌ Ошибка CORP: {str(e)}"