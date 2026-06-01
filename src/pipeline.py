import os

from src.deidentifier import MedicalDeidentifier
from src.embedder import MedicalEmbedder
from src.hallucination_checker import HallucinationChecker
from src.reranker import MedicalReranker
from src.retriever import MedicalRetriever

if os.getenv('USE_OLLAMA', '0') == '1':
    from src.generator_ollama import MedicalGenerator
else:
    from src.generator import MedicalGenerator

_NO_RESULT_ANSWER = (
    '資料庫中無充足相似病例，無法提供有依據的診斷方向。\n'
    '建議諮詢專科醫師或查閱相關醫學文獻。\n\n'
    '免責聲明：本系統提供的資訊僅供醫療專業人員參考，不構成診斷或治療建議。'
)

_DEFAULT_INDEX_DIR = 'data/faiss_index'


class MedicalRAGPipeline:
    """整合去識別化、Embedding、Retrieval、Reranking、Generation、Hallucination 檢查"""

    def __init__(
        self,
        index_dir: str = _DEFAULT_INDEX_DIR,
        model_name: str = 'meta-llama/Llama-3.2-3B-Instruct',
        use_4bit: bool = True,
        top_k: int = 5,
        retrieval_top_k: int = 20,
        min_score_threshold: float = 0.5,
    ):
        self.top_k = top_k
        self.retrieval_top_k = max(retrieval_top_k, top_k)
        self.min_score_threshold = min_score_threshold

        import torch
        reranker_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.deidentifier = MedicalDeidentifier()
        self.embedder = MedicalEmbedder()
        self.retriever = MedicalRetriever()
        self.reranker = MedicalReranker(device=reranker_device)
        if os.getenv('USE_OLLAMA', '0') == '1':
            self.generator = MedicalGenerator()
        else:
            self.generator = MedicalGenerator(model_name=model_name, use_4bit=use_4bit)
        self.checker = HallucinationChecker()

        index_path = os.path.join(index_dir, 'index.faiss')
        metadata_path = os.path.join(index_dir, 'metadata.pkl')
        self.retriever.load_index(index_path, metadata_path)

    def query(self, user_input: str) -> dict:
        """
        回傳：
        {
            "answer": str,
            "retrieved_cases": list,
            "confidence": str,
            "retrieval_score": float,
            "warnings": list[str],
            "requires_review": bool,
            "phi_detected_in_query": bool
        }
        """
        # 1. 掃描輸入 PHI
        cleaned_query, detected_phi = self.deidentifier.scan_query(user_input)
        phi_detected = len(detected_phi) > 0

        warnings = []
        if phi_detected:
            warnings.append(
                f'您的查詢中偵測到個人識別資訊（{", ".join(detected_phi)}），已自動遮蔽。'
            )

        # 2. Embedding
        query_embedding = self.embedder.embed_query(cleaned_query)

        # 3. Retrieval（取較多候選供 reranker 使用）
        retrieved = self.retriever.search(query_embedding, top_k=self.retrieval_top_k)

        # 4. 低相似度直接回傳（用 FAISS top-1 score 判斷，reranker 前）
        if not retrieved or retrieved[0]['score'] < self.min_score_threshold:
            return {
                'answer': _NO_RESULT_ANSWER,
                'retrieved_cases': [],
                'confidence': 'LOW',
                'retrieval_score': retrieved[0]['score'] if retrieved else 0.0,
                'warnings': warnings + ['相似度不足，無法提供有依據的回答'],
                'requires_review': True,
                'phi_detected_in_query': phi_detected,
            }

        # 5. Reranker 重排，縮減到 top_k
        retrieved = self.reranker.rerank(cleaned_query, retrieved, top_k=self.top_k)

        # 準備 context（使用 context chunks）
        contexts = [r['context'] for r in retrieved if r.get('context')]
        retrieval_scores = [r['score'] for r in retrieved]

        # 6. LLM 生成
        answer = self.generator.generate(cleaned_query, contexts)

        # 7. Hallucination 檢查
        check_result = self.checker.check(
            query=cleaned_query,
            answer=answer,
            retrieved_contexts=contexts,
            retrieval_scores=retrieval_scores,
        )
        warnings.extend(check_result['warnings'])

        # 8. 掃描輸出 PHI
        answer, has_phi_in_output = self.deidentifier.scan_output(answer)
        if has_phi_in_output:
            warnings.append('LLM 輸出中偵測到潛在個人識別資訊，已自動遮蔽。')

        # 組裝相似病例摘要（只回傳 query chunk 內容，不含原始 context）
        retrieved_cases = [
            {
                'rank': i + 1,
                'record_id': r['record_id'],
                'summary': r['content'],
                'context': r.get('context', ''),
                'score': r['score'],
                'rerank_score': r.get('rerank_score'),
                'metadata': r['metadata'],
            }
            for i, r in enumerate(retrieved)
        ]

        return {
            'answer': answer,
            'retrieved_cases': retrieved_cases,
            'confidence': check_result['confidence'],
            'retrieval_score': check_result['retrieval_score'],
            'faithfulness_score': check_result['faithfulness_score'],
            'warnings': warnings,
            'requires_review': check_result['requires_review'],
            'phi_detected_in_query': phi_detected,
        }
