import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = """你是一個醫療資訊輔助系統。你的任務是根據提供的歷史病例資料，
幫助醫療人員了解相似病例的診斷方向和治療方式。

重要規則：
1. 只能根據提供的病例資料回答，不得憑空捏造醫療資訊
2. 不得輸出任何病患個人識別資訊（姓名、身分證、電話、地址）
3. 所有建議僅供參考，最終臨床決策需由執業醫師判斷
4. 如果提供的資料不足以回答問題，請明確說明「資料庫中無充足相似病例」
5. 回答時需標注資訊來源（「根據相似病例顯示...」）
6. 若涉及用藥劑量，必須強調需經醫師確認"""


class MedicalGenerator:
    """Llama-3.2-3B-Instruct 文字生成（HuggingFace 本地載入）"""

    def __init__(self, model_name: str = 'meta-llama/Llama-3.2-3B-Instruct', use_4bit: bool = True):
        self.model_name = model_name
        hf_token = os.environ.get('HF_TOKEN')

        print(f'[Generator] 載入模型：{model_name}')

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        has_gpu = torch.cuda.is_available()

        if use_4bit and has_gpu:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map='auto',
                token=hf_token,
            )
        elif has_gpu:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto',
                token=hf_token,
            )
        else:
            print('[Generator] 無 GPU，使用 CPU float32（速度較慢）')
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                token=hf_token,
            )

        self.model.eval()
        print('[Generator] 模型載入完成')

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

        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_message},
        ]

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors='pt',
            return_dict=True,
        )

        device = next(self.model.parameters()).device
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return answer.strip()
