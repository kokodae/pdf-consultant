import uuid
import requests
from urllib.parse import urlencode
from typing import Optional
from backend.config import GIGACHAT_AUTH_URL, GIGACHAT_SCOPE

def get_access_token(api_key: str) -> Optional[str]:
    """Получение токена доступа через REST API"""
    try:
        rquid = str(uuid.uuid4())
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': rquid,
            'Authorization': f'Basic {api_key}'
        }
        data = {'scope': GIGACHAT_SCOPE}
        
        print(f"🔄 Запрос токена доступа к {GIGACHAT_AUTH_URL}")
        response = requests.post(
            GIGACHAT_AUTH_URL,
            headers=headers,
            data=urlencode(data),
            verify=False,
            timeout=30
        )
        
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data.get('access_token')
            print(f"✅ Токен получен успешно")
            return access_token
        else:
            print(f"❌ Ошибка получения токена: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Ошибка при получении токена: {e}")
        return None
