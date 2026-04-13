import os
import pickle

import faiss
import numpy as np


class MedicalRetriever:
    """FAISS 索引建立與查詢（Small-to-Big Retrieval）"""

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        # query chunks list（與 FAISS 索引順序對應）
        self._query_chunks: list[dict] = []
        # record_id -> context chunk 的查找表
        self._context_map: dict[str, dict] = {}

    def build_index(self, chunks: list[dict], embeddings: np.ndarray,
                    save_dir: str = 'data/faiss_index'):
        """建立 FAISS 索引並儲存

        chunks：所有 query-type chunks（與 embeddings 一一對應）
        embeddings：shape (N, 384)，已 normalized
        """
        os.makedirs(save_dir, exist_ok=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self._query_chunks = chunks

        index_path = os.path.join(save_dir, 'index.faiss')
        metadata_path = os.path.join(save_dir, 'metadata.pkl')

        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'query_chunks': self._query_chunks,
                'context_map': self._context_map,
            }, f)

        print(f'[Retriever] 索引已儲存：{index_path}')
        print(f'[Retriever] Query chunks 數量：{len(self._query_chunks)}，維度：{dim}')

    def register_context_chunks(self, context_chunks: list[dict]):
        """登記 context chunks，供 Small-to-Big 查找使用"""
        for chunk in context_chunks:
            self._context_map[chunk['record_id']] = chunk

    def load_index(self, index_path: str, metadata_path: str):
        """載入已建立的索引"""
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
        self._query_chunks = data['query_chunks']
        self._context_map = data.get('context_map', {})
        print(f'[Retriever] 索引載入完成，共 {len(self._query_chunks)} 個 query chunks')

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        查詢，回傳 top-k 相似病例。
        每筆結果：
        {
            "chunk_id": str,
            "record_id": str,
            "content": str,    # query chunk 內容
            "context": str,    # 對應的完整 context chunk
            "score": float,
            "metadata": dict
        }
        """
        if self.index is None:
            raise RuntimeError('索引尚未建立或載入，請先呼叫 build_index 或 load_index')

        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vec, min(top_k, len(self._query_chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._query_chunks[idx]
            record_id = chunk['record_id']
            context_chunk = self._context_map.get(record_id, {})
            results.append({
                'chunk_id': chunk['chunk_id'],
                'record_id': record_id,
                'content': chunk['content'],
                'context': context_chunk.get('content', ''),
                'score': float(score),
                'metadata': chunk.get('metadata', {}),
            })

        return results
