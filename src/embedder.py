import numpy as np
from sentence_transformers import SentenceTransformer


class MedicalEmbedder:
    """HuggingFace Embedding 封裝

    模型：paraphrase-multilingual-MiniLM-L12-v2
    輸出維度：384，支援繁體中文 + 醫學英文混合
    """

    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        print(f'[Embedder] 載入模型：{model_name}')
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """回傳 (N, 384) 的 normalized embedding 矩陣"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """回傳單一 query 的 embedding，shape (384,)"""
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return np.array(embedding[0], dtype=np.float32)
