import os

from openai import OpenAI

from src.generator import SYSTEM_PROMPT


class MedicalGenerator:
    """Ollama LLM 文字生成（透過 OpenAI 相容 API）"""

    def __init__(self, model_name: str = None, ollama_url: str = None):
        self.model_name = model_name or os.getenv('OLLAMA_MODEL', 'llama3.2')
        ollama_url = ollama_url or os.getenv('OLLAMA_URL', 'http://localhost:11434')
        self.client = OpenAI(
            base_url=f'{ollama_url}/v1',
            api_key='ollama',
        )
        print(f'[Generator] Ollama 端點：{ollama_url}，模型：{self.model_name}')

    def generate(self, query: str, retrieved_contexts: list[str], max_new_tokens: int = 1024) -> str:
        """根據 query 和 retrieved contexts 生成回答"""
        context_block = '\n\n'.join(
            f'【相似病例 {i+1}】\n{ctx}' for i, ctx in enumerate(retrieved_contexts) if ctx
        )

        user_message = f"""以下是與當前查詢相關的歷史病例資料：

{context_block}

---
醫療人員查詢：{query}

請根據上述病例資料，提供診斷方向與治療參考。"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user',   'content': user_message},
            ],
            max_tokens=max_new_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
