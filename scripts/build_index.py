"""
建立醫療病歷向量索引

執行方式：
    python scripts/build_index.py
    python scripts/build_index.py --records data/records.json --output data/faiss_index
"""

import argparse
import json
import os
import sys

# 讓 src 模組可被匯入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunker import MedicalChunker
from src.deidentifier import MedicalDeidentifier
from src.embedder import MedicalEmbedder
from src.retriever import MedicalRetriever


def main():
    parser = argparse.ArgumentParser(description='建立醫療病歷向量索引')
    parser.add_argument('--records', default='data/records.json', help='病歷 JSON 路徑')
    parser.add_argument('--output', default='data/faiss_index', help='索引輸出目錄')
    args = parser.parse_args()

    # 1. 載入病歷
    print(f'[1/6] 載入病歷：{args.records}')
    with open(args.records, 'r', encoding='utf-8') as f:
        records = json.load(f)
    if isinstance(records, dict):
        records = records.get('records', list(records.values()))
    print(f'      共 {len(records)} 筆病歷')

    # 2. 去識別化
    print('[2/6] 執行去識別化...')
    deidentifier = MedicalDeidentifier()
    deidentified = []
    for r in records:
        d = deidentifier.deidentify_record(r)
        # visit_date 會被 deidentifier 換成 [DOB]，先把年份存回來
        raw_date = r.get('visit_date', '')
        d['visit_year'] = raw_date[:4] if raw_date and len(raw_date) >= 4 else ''
        deidentified.append(d)

    # 3. Chunking
    print('[3/6] 執行章節切割...')
    chunker = MedicalChunker()
    all_chunks = []
    for record in deidentified:
        all_chunks.extend(chunker.chunk_record(record))

    query_chunks = [c for c in all_chunks if c['chunk_type'] == 'query']
    context_chunks = [c for c in all_chunks if c['chunk_type'] == 'context']
    print(f'      Query chunks：{len(query_chunks)}，Context chunks：{len(context_chunks)}')

    # 4. Embedding（只對 query chunks）
    print('[4/6] 計算 Embeddings...')
    embedder = MedicalEmbedder()
    texts = [c['content'] for c in query_chunks]
    embeddings = embedder.embed_texts(texts)
    print(f'      Embedding 矩陣：{embeddings.shape}')

    # 5. 建立 FAISS 索引
    print('[5/6] 建立 FAISS 索引...')
    retriever = MedicalRetriever()
    retriever.register_context_chunks(context_chunks)
    retriever.build_index(query_chunks, embeddings, save_dir=args.output)

    # 6. 統計
    print('[6/6] 完成！統計資訊：')
    index_path = os.path.join(args.output, 'index.faiss')
    index_size_kb = os.path.getsize(index_path) / 1024
    print(f'      病歷筆數：{len(records)}')
    print(f'      Query chunks 數量：{len(query_chunks)}')
    print(f'      Embedding 維度：{embeddings.shape[1]}')
    print(f'      索引檔案大小：{index_size_kb:.1f} KB')
    print(f'      索引位置：{os.path.abspath(args.output)}')
    print()
    print('      提示：可檢查 data/faiss_index/metadata.pkl 確認去識別化效果')


if __name__ == '__main__':
    main()
