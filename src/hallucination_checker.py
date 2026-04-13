import re


# 醫療關鍵詞（疾病、症狀、藥物、檢查等常見詞彙）
_MEDICAL_TERMS = [
    # 疾病
    '糖尿病', '高血壓', '心臟病', '心肌梗塞', '腦中風', '肺炎', '氣喘', '過敏',
    '癌症', '腫瘤', '骨折', '貧血', '感染', '發炎', '腎衰竭', '肝硬化',
    '憂鬱症', '焦慮症', '失眠', '頭痛', '偏頭痛', '眩暈', '胃潰瘍', '腸炎',
    # 症狀
    '發燒', '咳嗽', '呼吸困難', '胸痛', '胸悶', '噁心', '嘔吐', '腹痛',
    '腹瀉', '便秘', '水腫', '疲勞', '虛弱', '暈厥', '抽搐', '麻木',
    # 藥物（通用名）
    'aspirin', 'metformin', 'insulin', 'warfarin', 'statin', 'amoxicillin',
    '阿斯匹靈', '胰島素', '抗生素', '抗凝血劑', '降血壓藥', '降血糖藥',
    # 檢查
    '血糖', '血壓', '心電圖', 'CT', 'MRI', '超音波', '血液檢查', '尿液檢查',
    'X光', '內視鏡', '切片', '培養', 'HbA1c', 'INR',
    # 處置
    '手術', '化療', '放療', '透析', '輸血', '住院', '急診', '手術治療',
]

# 強烈建議詞（觸發 LOW 信心）
_STRONG_SUGGESTION_PATTERNS = re.compile(
    r'建議立即|必須|緊急|立刻就醫|馬上|一定要|強烈建議'
)

# 劑量數字模式（觸發 LOW 信心）
_DOSAGE_PATTERN = re.compile(
    r'\d+\s*(?:mg|mcg|μg|mL|ml|IU|U|units?|錠|顆|瓶|包|袋|次/天|次/日|mg/kg|mg/day)'
)


class HallucinationChecker:
    """驗證 LLM 回答是否有依據（信心分級：HIGH / MEDIUM / LOW）"""

    def _extract_medical_terms(self, text: str) -> set[str]:
        """從文字中提取出現的醫療詞彙"""
        found = set()
        text_lower = text.lower()
        for term in _MEDICAL_TERMS:
            if term.lower() in text_lower:
                found.add(term)
        return found

    def _compute_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """
        計算 faithfulness score：
        回答中出現在任一 context 的醫療詞彙 / 回答中所有醫療詞彙
        """
        answer_terms = self._extract_medical_terms(answer)
        if not answer_terms:
            return 1.0  # 無醫療詞彙時不懲罰

        combined_context = ' '.join(contexts).lower()
        supported = sum(1 for t in answer_terms if t.lower() in combined_context)
        return supported / len(answer_terms)

    def check(
        self,
        query: str,
        answer: str,
        retrieved_contexts: list[str],
        retrieval_scores: list[float],
    ) -> dict:
        """
        回傳：
        {
            "confidence": "HIGH" | "MEDIUM" | "LOW",
            "retrieval_score": float,
            "faithfulness_score": float,
            "warnings": list[str],
            "requires_review": bool
        }
        """
        top_retrieval_score = max(retrieval_scores) if retrieval_scores else 0.0
        faithfulness_score = self._compute_faithfulness(answer, retrieved_contexts)

        warnings = []

        # 劑量數字
        if _DOSAGE_PATTERN.search(answer):
            warnings.append('回答包含具體劑量數字，請由執業醫師確認後再使用')

        # 強烈建議詞
        if _STRONG_SUGGESTION_PATTERNS.search(answer):
            warnings.append('回答含有強烈建議語氣，請謹慎評估臨床適用性')

        # 低 retrieval score
        if top_retrieval_score < 0.6:
            warnings.append(f'相似度分數偏低（{top_retrieval_score:.2f}），資料庫中可能無足夠相似病例')

        # 信心等級判斷
        has_dosage_or_strong = bool(
            _DOSAGE_PATTERN.search(answer) or _STRONG_SUGGESTION_PATTERNS.search(answer)
        )

        if (top_retrieval_score >= 0.75 and faithfulness_score >= 0.6
                and not has_dosage_or_strong):
            confidence = 'HIGH'
        elif (top_retrieval_score >= 0.6 or faithfulness_score >= 0.5) and not has_dosage_or_strong:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        requires_review = confidence == 'LOW' or len(warnings) > 0

        return {
            'confidence': confidence,
            'retrieval_score': top_retrieval_score,
            'faithfulness_score': faithfulness_score,
            'warnings': warnings,
            'requires_review': requires_review,
        }
