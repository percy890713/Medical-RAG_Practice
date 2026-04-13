from sentence_transformers import CrossEncoder


class MedicalReranker:
    """Cross-encoder Reranker（第二層排序）

    流程：FAISS 先用向量搜尋取出較多候選（top-N），
    Reranker 對每個 (query, document) 配對直接計算相關性分數，
    重排後只取 top-k 給 LLM 生成。

    優點：精準度遠高於向量相似度，能正確處理「用詞不同但病情相似」的情況。
    代價：速度較慢（O(N) 次 forward pass），所以只用在縮小後的候選集上。

    模型選擇：mmarco-mMiniLMv2-L12-H384-v1 支援多語言（含繁中 + 醫學英文混合）
    """

    def __init__(
        self,
        model_name: str = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
        device: str = 'cpu',
    ):
        print(f'[Reranker] 載入模型：{model_name}')
        self.model = CrossEncoder(model_name, device=device)
        print('[Reranker] 模型載入完成')

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """
        對 FAISS 候選結果重新排序。

        candidates：MedicalRetriever.search() 的回傳值（list of dict）
        top_k：重排後保留幾筆

        回傳同樣格式的 list[dict]，依 rerank_score 降序，長度 <= top_k。
        每筆結果新增 'rerank_score' 欄位，保留原始 'score'（faiss cosine similarity）。
        """
        if not candidates:
            return []

        # cross-encoder 輸入：(query, document) 配對
        # 用 content（症狀+診斷摘要）做排序依據，語意最集中
        pairs = [(query, c['content']) for c in candidates]
        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate['rerank_score'] = float(score)

        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_k]
