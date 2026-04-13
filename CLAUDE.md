# CLAUDE.md — 給 Claude Code 的專案說明

本文件說明專案架構、設計決策與實作細節，供 Claude Code 在開發時參考。

---

## 專案目標

建立一套**全離線**醫療病歷 RAG 系統，讓醫療人員輸入症狀後，能查詢歷史相似病例，
取得診斷方向與治療建議。重點在於：**隱私安全**、**避免幻覺**、**語意完整保留**。

---

## 技術選型與理由

| 元件 | 選擇 | 理由 |
|------|------|------|
| Vector DB | FAISS | 純本地，無 server，適合練習 |
| Embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 多語言支援（繁中+醫學英文混合），輕量 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 多語言 cross-encoder，對 FAISS 候選重排 |
| LLM | `meta-llama/Llama-3.2-3B-Instruct` | 使用者指定，全離線 |
| Web UI | FastAPI + HTML/Jinja2 | 自訂版面彈性高，支援查詢紀錄等擴充功能 |
| Chunking | Section-based（章節切割） | 醫療文本不能在語意中間切斷 |

---

## 需要實作的檔案清單

以下是 Claude Code 需要建立的所有 Python 檔案：

---

### `requirements.txt`

```
torch>=2.0.0
transformers>=4.40.0
sentence-transformers>=2.7.0
faiss-cpu>=1.7.4          # 若有 GPU 可改 faiss-gpu
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
jinja2>=3.1.0
python-multipart>=0.0.9
accelerate>=0.27.0
huggingface-hub>=0.22.0
presidio-analyzer>=2.2.0
presidio-anonymizer>=2.2.0
spacy>=3.7.0
langdetect>=1.0.9
numpy>=1.24.0
tqdm>=4.66.0
```

---

### `src/deidentifier.py`

**功能**：PHI 去識別化模組（三層）

```python
# 實作要點：
# 1. 使用正規表示式偵測台灣特定格式的 PHI：
#    - 身分證字號：[A-Z][12]\d{8}
#    - 台灣手機號碼：09\d{2}-?\d{3}-?\d{3}
#    - 台灣地址：包含「市」「區」「路」「街」「號」的字串
#    - 中文姓名（2-4字，搭配上下文判斷，如「病患」「患者」後面的名字）
# 2. 替換策略：用佔位符取代，不是刪除
#    - 姓名 → [PATIENT_ID] 或 [姓名_A], [姓名_B]（同文件內保持一致）
#    - 身分證 → [ID_NUMBER]
#    - 電話 → [PHONE]
#    - 地址 → [ADDRESS]
#    - 生日 → [DOB]
#    - 緊急聯絡人 → [EMERGENCY_CONTACT]
# 3. 提供兩種 mode：
#    - deidentify_for_index(record: dict) -> dict  # 用於建索引，完整去識別
#    - scan_output(text: str) -> tuple[str, bool]  # 用於掃描輸出，偵測並警告

class MedicalDeidentifier:
    def deidentify_record(self, record: dict) -> dict:
        """去識別化整筆病歷，回傳去識別化後的 dict"""
        pass

    def scan_query(self, query: str) -> tuple[str, list[str]]:
        """掃描使用者查詢，回傳 (cleaned_query, detected_phi_types)"""
        pass

    def scan_output(self, text: str) -> tuple[str, bool]:
        """掃描 LLM 輸出，回傳 (text, has_phi_warning)"""
        pass
```

---

### `src/chunker.py`

**功能**：醫療文本章節切割（Section-based chunking）

```python
# 實作要點：
# 病歷 JSON 結構如下（去識別化後）：
# - chief_complaint: 主訴
# - hpi: 現病史
# - pmh: 過去病史
# - medications: 用藥
# - vitals: 生命徵象
# - labs: 檢驗
# - imaging: 影像
# - assessment: 診斷
# - plan: 治療計畫
# - outcome: 結果
# - follow_up: 追蹤

# Chunk 策略：
# 1. 主要 chunk（用於 Retrieval）：
#    將 chief_complaint + hpi + assessment 合併成一個 chunk
#    → 這是「症狀→診斷」的核心語意單元，最常被查詢
# 2. 詳細 chunk（用於 Generation context）：
#    完整病歷的 assessment + plan + outcome + follow_up
#    → 這是「診斷→治療→結果」的完整敘述
# 3. 每個 chunk 附 metadata：
#    record_id, department, chunk_type, visit_date（只保留年份避免重識別）

# Small-to-Big Retrieval 設計：
# - 用「主要 chunk」做向量相似度搜尋（小 chunk，語意精準）
# - 找到後，回傳對應的「詳細 chunk」給 LLM 生成（大 chunk，資訊完整）

class MedicalChunker:
    def chunk_record(self, record: dict) -> list[dict]:
        """
        回傳 list of chunks，每個 chunk 格式：
        {
            "chunk_id": str,
            "record_id": str,
            "chunk_type": "query" | "context",   # query=小chunk for retrieval, context=大chunk for generation
            "content": str,                        # 去識別化後的文字內容
            "metadata": dict
        }
        """
        pass
```

---

### `src/reranker.py`

**功能**：cross-encoder 重排序（第二層排序）

```python
# 實作要點：
# 1. 模型：cross-encoder/mmarco-mMiniLMv2-L12-H384-v1（多語言）
# 2. 輸入：(query, document) 配對 list
# 3. 輸出：依 rerank_score 降序，取 top-k
# 4. 在 pipeline 中：FAISS 先取 top-20，reranker 縮減為 top-5
# 5. 結果新增 rerank_score 欄位，保留原始 faiss score

class MedicalReranker:
    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """重排後回傳 top-k，每筆帶 rerank_score"""
        pass
```

---

### `src/embedder.py`

**功能**：HuggingFace Embedding 封裝

```python
# 實作要點：
# 1. 模型：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
#    - 支援繁體中文 + 醫學英文混合
#    - 輸出維度 384
# 2. 批次處理（batch_size=32）提高效率
# 3. 歸一化向量（normalize_embeddings=True），用於 cosine similarity

class MedicalEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        pass

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """回傳 (N, 384) 的 normalized embedding 矩陣"""
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """回傳單一 query 的 embedding，shape (384,)"""
        pass
```

---

### `src/retriever.py`

**功能**：FAISS 索引建立與查詢

```python
# 實作要點：
# 1. 索引類型：IndexFlatIP（內積，配合 normalized 向量等於 cosine similarity）
# 2. 儲存：index.faiss + metadata.pkl（pickle 儲存 chunk 的文字和 metadata）
# 3. 查詢時回傳：top-k chunks + 對應的 cosine similarity score
# 4. Small-to-Big 邏輯：
#    - 用 query chunk 找相似，得到 record_id
#    - 回傳對應的 context chunk（完整診斷+治療內容）給 LLM

class MedicalRetriever:
    def build_index(self, chunks: list[dict], embeddings: np.ndarray):
        """建立 FAISS 索引並儲存"""
        pass

    def load_index(self, index_path: str, metadata_path: str):
        """載入已建立的索引"""
        pass

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        查詢，回傳：
        [
            {
                "chunk_id": str,
                "record_id": str,
                "content": str,        # query chunk 內容（症狀+診斷摘要）
                "context": str,        # 對應的完整 context chunk
                "score": float,        # cosine similarity
                "metadata": dict
            },
            ...
        ]
        """
        pass
```

---

### `src/generator.py`

**功能**：Llama-3.2-3B-Instruct 文字生成

```python
# 實作要點：
# 1. 模型載入：使用 transformers AutoModelForCausalLM + AutoTokenizer
# 2. 量化（節省記憶體）：使用 BitsAndBytesConfig 4-bit quantization（如有 GPU）
#    或直接 float32 CPU 推理（慢但可用）
# 3. System prompt 設計（重要！）：
#    - 明確說明這是醫療資訊系統
#    - 要求回答必須基於提供的病例內容
#    - 要求標注不確定性
#    - 禁止輸出病患識別資訊
#    - 強制加上免責聲明

SYSTEM_PROMPT = """你是一個醫療資訊輔助系統。你的任務是根據提供的歷史病例資料，
幫助醫療人員了解相似病例的診斷方向和治療方式。

重要規則：
1. 只能根據提供的病例資料回答，不得憑空捏造醫療資訊
2. 不得輸出任何病患個人識別資訊（姓名、身分證、電話、地址）
3. 所有建議僅供參考，最終臨床決策需由執業醫師判斷
4. 如果提供的資料不足以回答問題，請明確說明「資料庫中無充足相似病例」
5. 回答時需標注資訊來源（「根據相似病例顯示...」）
6. 若涉及用藥劑量，必須強調需經醫師確認"""

# 4. Prompt 格式：
# System: SYSTEM_PROMPT
# User: 症狀描述 + 相關病例 context（去識別化後）
# Assistant: （生成）

class MedicalGenerator:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", use_4bit: bool = True):
        pass

    def generate(self, query: str, retrieved_contexts: list[str], max_new_tokens: int = 512) -> str:
        """根據 query 和 retrieved contexts 生成回答"""
        pass
```

---

### `src/hallucination_checker.py`

**功能**：驗證 LLM 回答是否有依據

```python
# 實作要點：
# 1. Faithfulness check（基於字符重疊，簡化版）：
#    - 計算回答中的關鍵醫療名詞是否出現在 retrieved context 中
#    - 醫療名詞提取：用正規表示式或簡單關鍵字列表（疾病名、藥物名）
#    - 分數 = 回答中出現在 context 的醫療詞彙 / 回答中所有醫療詞彙
#
# 2. Confidence level 判斷：
#    - retrieval_score >= 0.75 AND faithfulness >= 0.6 → HIGH（綠色）
#    - retrieval_score >= 0.6 OR faithfulness >= 0.5  → MEDIUM（黃色警示）
#    - 以下條件 → LOW（橙色警示，建議人工確認）：
#        * retrieval_score < 0.6
#        * 回答包含劑量數字
#        * 回答包含「建議立即」「必須」等強烈建議詞
#
# 3. 回傳結果：
#    {
#        "confidence": "HIGH" | "MEDIUM" | "LOW",
#        "retrieval_score": float,
#        "faithfulness_score": float,
#        "warnings": list[str],      # 具體警示訊息
#        "requires_review": bool
#    }

class HallucinationChecker:
    def check(self, query: str, answer: str, retrieved_contexts: list[str], retrieval_scores: list[float]) -> dict:
        pass
```

---

### `scripts/build_index.py`

**功能**：一次性建立向量索引

```python
# 執行流程：
# 1. 載入 data/records.json
# 2. 對每筆病歷執行 MedicalDeidentifier.deidentify_record()
# 3. 對每筆去識別化病歷執行 MedicalChunker.chunk_record()
# 4. 收集所有 query-type chunks 的文字
# 5. 用 MedicalEmbedder 批次 embed
# 6. 用 MedicalRetriever.build_index() 建立 FAISS 索引
# 7. 儲存至 data/faiss_index/
# 8. 印出統計：總 chunks 數、embedding 維度、索引大小

# 執行方式：python scripts/build_index.py
```

---

### `src/pipeline.py`

**功能**：整合所有模組

```python
# 整合流程：
# 1. 接收 user query（str）
# 2. deidentifier.scan_query(query) → 偵測輸入 PHI，若有則告警
# 3. embedder.embed_query(query) → query embedding
# 4. retriever.search(query_embedding, top_k=5) → 取得相似病例
# 5. 若最高 retrieval_score < 0.5 → 直接回傳「無充足相似病例」
# 6. generator.generate(query, contexts) → LLM 生成
# 7. hallucination_checker.check(...) → 信心評估
# 8. deidentifier.scan_output(answer) → 最後 PHI 掃描
# 9. 回傳完整結果 dict

class MedicalRAGPipeline:
    def query(self, user_input: str) -> dict:
        """
        回傳格式：
        {
            "answer": str,              # LLM 回答（已去識別化掃描）
            "retrieved_cases": list,    # 相似病例摘要（去識別化）
            "confidence": str,          # HIGH / MEDIUM / LOW
            "retrieval_score": float,   # 最高相似度分數
            "warnings": list[str],      # 任何警示訊息
            "requires_review": bool,    # 是否需要人工確認
            "phi_detected_in_query": bool  # 使用者輸入是否含 PHI
        }
        """
        pass
```

---

### `app.py`

**功能**：FastAPI 後端 + 查詢 Log

```python
# Routes：
# GET  /         → 渲染 templates/index.html（主查詢介面）
# POST /query    → 接收 {"query": str}，呼叫 pipeline，寫 log，回傳 JSON
# GET  /logs     → 讀取 data/query_log.jsonl，渲染 templates/logs.html

# /query 回傳格式：
# {
#   "confidence": "HIGH" | "MEDIUM" | "LOW",
#   "answer": str,                     # Markdown 文字
#   "retrieved_cases": [               # 已格式化，前端直接渲染
#     {"rank": int, "score": "77.9%", "meta": str, "summary": str, "context": str}
#   ],
#   "warnings": list[str]
# }

# 查詢 log 格式（data/query_log.jsonl，每行一筆 JSON）：
# {
#   "timestamp": "2026-04-05T10:30:00",
#   "query": str,
#   "confidence": str,
#   "retrieval_score": float,
#   "phi_detected": bool,
#   "warnings_count": int,
#   "requires_review": bool
# }

# 啟動：uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
```

### `templates/index.html`

**功能**：主查詢介面（純 HTML/CSS/JS）

```
# 版面：兩欄（1:2）
# 左欄：症狀描述 textarea、查詢按鈕、範例查詢清單（點擊填入）
# 右欄：信心等級（彩色圓點）、回答（marked.js 渲染 Markdown）、
#        相似病例 accordion、警示訊息區塊、免責聲明
#
# JS 行為：
# - fetch POST /query，動態更新右欄（無頁面重整）
# - Ctrl+Enter 送出
# - marked.js（CDN）渲染回答中的 Markdown
```

### `templates/logs.html`

**功能**：查詢紀錄頁（Jinja2 server-side render）

```
# 統計卡片：總次數、HIGH/MEDIUM/LOW 各自次數、PHI 偵測次數
# 表格欄位：時間、查詢內容、信心等級（badge）、相似度、警示數、PHI偵測、需確認
# 前端搜尋：搜尋框過濾查詢內容（純 JS，不重新請求後端）
```

---

## 重要設計決策記錄

### 去識別化：兩個階段的職責不同

- **Indexing 前**（`deidentify_record`）：靜態防護，確保 Vector DB 中不含 PHI
- **Query 時**（`scan_query`）：動態防護，攔截使用者不小心輸入的個資
- **Output 後**（`scan_output`）：最後防線，防止 LLM 從 context 重組 PHI

### Chunking：語意完整性優先

- 不使用固定字數切割
- `chief_complaint + hpi + assessment` 永遠在同一個 chunk
- `assessment + plan + outcome` 永遠在同一個 context chunk
- 診斷和症狀、診斷和治療計畫，**絕對不切開**

### Hallucination 處理：信心分級而非二元判斷

- 不是「有幻覺」或「沒幻覺」的二元分類
- 而是 HIGH / MEDIUM / LOW 的分級，搭配具體警示訊息
- LOW 信心不直接阻止回答，而是加上醒目警示，體現人在迴路精神

### FAISS 索引設計

- 只有 `query-type` chunks 進入 FAISS 索引（症狀+診斷描述，語意集中）
- `context-type` chunks 以 record_id 關聯，只在生成時取用
- 這是 Small-to-Big Retrieval 設計

---

## 常見開發問題

**Q: Llama-3.2 需要多少記憶體？**
A: 3B 模型：
- 4-bit 量化：約 2.5GB VRAM
- Float16：約 6GB VRAM
- CPU Float32：約 12GB RAM（速度很慢，每次生成約 200–400 秒）

**Q: Embedding 模型支援中文嗎？**
A: `paraphrase-multilingual-MiniLM-L12-v2` 支援 50+ 語言，包含繁體中文及中英混合。

**Q: 如果 FAISS 找不到相似案例怎麼辦？**
A: 在 `pipeline.py` 中，若 top-1 cosine similarity < 0.5，直接回傳預設訊息，不進行 LLM 生成，避免幻覺。

**Q: 如何測試去識別化效果？**
A: 可在 `scripts/build_index.py` 執行完後，檢查 `data/faiss_index/metadata.pkl` 中儲存的 chunk 文字，確認無原始 PHI。

**Q: PyTorch 裝了 CPU-only 版本怎麼辦？**
A: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
注意：torchvision/torchaudio 必須和 torch 同時在同一 index-url 下安裝，否則版本不相容會報 RuntimeError。

**Q: app.py 啟動時若索引不存在會怎樣？**
A: `POST /query` 在第一次呼叫時才初始化 pipeline。若 `data/faiss_index/index.faiss` 不存在，回傳 HTTP 503 + 說明訊息，前端顯示在回答區塊，不會拋出原始 FAISS 錯誤。

**Q: 查詢紀錄存在哪？格式是什麼？**
A: `data/query_log.jsonl`，每行一筆 JSON。可直接用文字編輯器或 `jq` 解析，也可透過 `http://localhost:7860/logs` 用瀏覽器查看。

---

## 注意事項

1. `data/records.json` 含有原始 PHI（模擬資料），不應直接進入索引，必須先過 `deidentifier`
2. `data/faiss_index/` 目錄加入 `.gitignore`（建立後本地使用，不上傳）
3. `HF_TOKEN` 請用環境變數，不要 hardcode 在程式碼中
4. FastAPI 以 `host="0.0.0.0"` 啟動僅供區域網路存取，醫療場景不應對外開放，勿加反向代理暴露到公網

---

## 實作細節補充（開發過程補記）

### records.json 資料結構
`medications` 欄位為 `list[dict]`，格式：
```json
[{"name": "Amlodipine", "dose": "5mg", "frequency": "QD"}]
```
`chunker.py` 在處理時需展開為字串，不能直接 `join`。

### generator.py attention_mask
Llama tokenizer 的 pad_token 與 eos_token 相同，生成時需明確傳入 `attention_mask`：
```python
tokenized = tokenizer.apply_chat_template(..., return_dict=True)
model.generate(input_ids, attention_mask=tokenized['attention_mask'], ...)
```

### app.py retrieved_cases 顯示
`pipeline.query()` 回傳的 `retrieved_cases` 包含 `summary`（症狀+診斷）與 `context`（治療計畫+結果）兩個欄位。`app.py` 的 `format_cases()` 會轉成前端易用的格式（score 已轉為百分比字串），`templates/index.html` 的 JS 直接渲染，分兩個區塊展示。

### Reranker 設計
- FAISS `retrieval_top_k`（預設 20）> reranker `top_k`（預設 5），資料量大時調高 `retrieval_top_k` 即可，不需改架構
- cross-encoder 使用 `content`（query chunk：症狀+診斷）而非 `context`（治療計畫）做排序，語意最集中
- 閾值判斷（`min_score_threshold`）在 reranker **之前**用 FAISS score 做，避免浪費 reranker 時間

### PyTorch GPU 安裝
需用官方 CUDA wheel index，且 torch / torchvision / torchaudio 三個套件必須同一指令一起裝：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
