# 醫療病歷 RAG 系統（練習專案）

> ⚠️ **重要聲明**：本專案為學習練習用途，所有病歷資料均為人工合成，不含真實病患資訊。

---

## 專案概述

本系統實作一套針對醫療病歷的 RAG（Retrieval-Augmented Generation）查詢架構，核心功能如下：

- 輸入症狀描述，查詢歷史病例中相似案例
- 取得對應的診斷方法、治療計畫與結果
- 內建 PHI（個人健康資訊）去識別化層
- 本地端完全離線運行（FAISS + HuggingFace 模型）
- FastAPI + 純 HTML/CSS/JS Web UI 介面
- 查詢紀錄功能（`/logs` 頁面）

---

## 系統架構

```
使用者查詢
    │
    ▼
[PHI 偵測層] ── 攔截輸入中的個資
    │
    ▼
[Embedding 查詢] ── HuggingFace 本地 Embedding 模型
    │
    ▼
[FAISS 向量檢索] ── 找出候選病例 chunks（top-20，Small-to-Big）
    │
    ▼
[Reranker] ── cross-encoder 重排，縮減為 top-5
    │
    ▼
[LLM 生成] ── Llama-3.2-3B-Instruct (HuggingFace)
    │
    ▼
[Hallucination Check] ── 驗證回答有無根據
    │
    ▼
[Output PHI 掃描] ── 確保回答不含識別資訊
    │
    ▼
FastAPI + HTML 呈現結果（附信心分數 + 來源依據 + 查詢紀錄）
```
| 頁面 | 網址 |
|------|------|
| 主查詢介面 | `http://localhost:7860` |
| 查詢紀錄 | `http://localhost:7860/logs` |
---

## 環境需求

- Python 3.10+
- CUDA GPU（建議 VRAM ≥ 3GB，4-bit 量化）或 CPU（速度極慢，約 200+ 秒/次）
  - 實測環境：NVIDIA RTX 3060 12GB + CUDA 12.6
- 磁碟空間：約 8GB（模型下載）
- RAM：建議 16GB+

---

## 啟動方式

有兩種方式，**互相獨立**，依需求選一種：

| | 本機直接跑 | Docker + Ollama |
|---|---|---|
| LLM | HuggingFace Llama-3.2（本地載入） | Ollama（獨立服務） |
| 需要 GPU | 建議有（4-bit 量化） | 不需要 |
| 關掉 terminal | 程式停止 | 繼續背景執行 |
| 安裝複雜度 | 需裝 PyTorch + CUDA | 需裝 Docker Desktop |
| requirements | `requirements.txt` | `requirements_ollama.txt` |

---

### 方式一：本機直接跑

使用 HuggingFace 直接載入 Llama-3.2，不需要 Ollama。

#### 前置需求
- Python 3.10+、CUDA GPU（建議 VRAM ≥ 3GB）
- HuggingFace 帳號 + [Llama-3.2 存取授權](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

#### 安裝與啟動

```powershell
# 1. 建立虛擬環境
python -m venv venv
venv\Scripts\activate

# 2. 安裝 PyTorch（需先裝，CUDA 12.6）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 3. 安裝其他套件
pip install -r requirements.txt

# 4. 設定 HuggingFace Token
$env:HF_TOKEN="your_token_here"

# 5. 建立向量索引（只需要一次）
python scripts/build_index.py

# 6. 啟動 app
python app.py
```

---

### 方式二：Docker + Ollama

不需要 GPU，關掉 terminal 後繼續背景執行，重開機自動重啟。

#### 前置需求
- 安裝並**開啟 Docker Desktop**（等系統匣鯨魚 icon 靜止才算就緒）

#### 第一次啟動

```powershell
# 1. 啟動所有服務（背景執行）
#    ollama-setup 會自動下載 llama3.2，需要一些時間
docker-compose up -d

# 2. 確認狀態（rag-app 和 ollama 應為 running，ollama-setup 為 Exit 0 正常）
docker-compose ps

# 3. 建立向量索引（只需要做一次）
docker-compose exec rag-app python scripts/build_index.py
```

#### 之後每次啟動

```powershell
docker-compose up -d
```

#### 停止

```powershell
docker-compose down
```

#### 查看 log

```powershell
docker-compose logs -f rag-app
```

#### 換模型

修改 `docker-compose.yml` 中的 `OLLAMA_MODEL`，再重啟：

```yaml
- OLLAMA_MODEL=qwen2.5
```

```powershell
docker-compose down && docker-compose up -d
```

---

## 目錄結構

```
medical_rag/
├── README.md
├── CLAUDE.md                  # Claude Code 專用說明
├── requirements.txt
├── app.py                     # FastAPI 主程式入口
├── templates/
│   ├── index.html             # 主查詢介面
│   └── logs.html              # 查詢紀錄頁
├── data/
│   ├── records.json           # 50 筆合成病歷（含 PHI）
│   ├── query_log.jsonl        # 查詢紀錄（啟動後自動建立）
│   └── faiss_index/           # 建立後自動生成
│       ├── index.faiss
│       └── metadata.pkl
├── scripts/
│   └── build_index.py         # 一次性索引建立腳本
└── src/
    ├── deidentifier.py        # PHI 去識別化模組
    ├── chunker.py             # 醫療文本章節切割
    ├── embedder.py            # HuggingFace Embedding 封裝
    ├── retriever.py           # FAISS 檢索邏輯
    ├── reranker.py            # cross-encoder 重排序
    ├── generator.py           # Llama-3.2 生成邏輯
    ├── hallucination_checker.py # 幻覺驗證層
    └── pipeline.py            # 整合所有模組的 RAG pipeline
```

---

## 使用方式

### Web UI

啟動後在輸入框描述症狀，例如：
- `「病患有胸痛、呼吸困難、冷汗，troponin 升高，心電圖 ST elevation，請問診斷和處置？」`
- `「病患突發頭痛、頸部僵硬、發燒、光敏感，腦脊髓液混濁，可能是什麼？」`

回答格式包含：
1. **信心等級**（🟢 HIGH / 🟡 MEDIUM / 🟠 LOW）
2. **主要回答**（LLM 生成，基於相似病例）
3. **相似病例來源**（可展開，含症狀診斷 + 治療計畫 + 結果）
4. **警示訊息**（低相似度、含劑量、強烈建議詞等情況）
5. **免責聲明**（僅供參考，非正式醫療建議）

---

## Web UI 設計

### 框架選擇：FastAPI + 純 HTML/CSS/JS

| 考量 | 說明 |
|------|------|
| 後端框架 | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| 前端 | 純 HTML/CSS/JS（Jinja2 模板），搭配 marked.js 渲染 Markdown |
| 選擇理由 | 版面完全自訂、可加入查詢紀錄等自定功能；比 Gradio 彈性高 |
| 部署方式 | 本地啟動（`host=0.0.0.0`），醫療場景不對外開放 |
| 入口檔案 | `app.py`（FastAPI app）；前端模板置於 `templates/` |

### UI 版面規劃

```
┌─────────────────────────────────────────────────────┐
│  醫療病歷 RAG 查詢系統                               │
│  ⚠️ 本系統僅供學習練習，不得用於臨床決策             │
├─────────────────────────────────────────────────────┤
│  【輸入區】                                          │
│  ┌───────────────────────────────────────────────┐  │
│  │ 症狀描述（Textbox，多行）                      │  │
│  └───────────────────────────────────────────────┘  │
│  [ 送出查詢 ]                                        │
│                                                     │
│  範例查詢（Examples，點擊自動填入）：                 │
│    • 胸痛 + ST elevation + troponin 升高            │
│    • 突發頭痛 + 頸部僵硬 + 發燒                     │
│    • 血糖 520 + 意識模糊 + 酮酸中毒                 │
├─────────────────────────────────────────────────────┤
│  【輸出區】                                          │
│                                                     │
│  信心等級：🟢 HIGH / 🟡 MEDIUM / 🟠 LOW             │
│  （依 retrieval score + faithfulness score 決定）    │
│                                                     │
│  主要回答（Markdown）                                │
│  ┌───────────────────────────────────────────────┐  │
│  │ LLM 生成回答，含來源標注                       │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ▶ 相似病例來源（Accordion，可展開）                 │
│    ├─ 病例 1：症狀 + 診斷摘要 / 治療計畫 + 結果     │
│    ├─ 病例 2：...                                   │
│    └─ 病例 3：...                                   │
│                                                     │
│  ⚠️ 警示訊息（若有）                                │
│                                                     │
│  免責聲明（固定顯示）                                │
└─────────────────────────────────────────────────────┘
```

### 各輸出區塊說明

| 區塊 | 對應 pipeline 欄位 | 說明 |
|------|-------------------|------|
| 信心等級 | `confidence` | HIGH/MEDIUM/LOW，顏色燈號提示 |
| 主要回答 | `answer` | Markdown 格式，含來源引用語句 |
| 相似病例來源 | `retrieved_cases` | 每筆含 `summary`（症狀+診斷）與 `context`（治療+結果） |
| 警示訊息 | `warnings` | 低相似度、含劑量、強烈建議詞等情況才顯示 |
| 免責聲明 | 靜態文字 | 永遠顯示，不可隱藏 |

### 異常狀態處理

| 情況 | UI 顯示 |
|------|---------|
| 索引尚未建立（`index.faiss` 不存在） | 信心等級欄顯示操作說明，提示執行 `build_index.py` |
| Retrieval score < 0.5 | 不進入 LLM，直接顯示「資料庫中無充足相似病例」 |
| 輸入含 PHI | 警示欄顯示偵測到的個資類型，query 仍可繼續（去識別後送出） |

### 查詢紀錄（`/logs`）

每次查詢結果會自動寫入 `data/query_log.jsonl`（每行一筆 JSON），紀錄欄位：

| 欄位 | 說明 |
|------|------|
| `timestamp` | 查詢時間（ISO 8601） |
| `query` | 原始查詢文字 |
| `confidence` | HIGH / MEDIUM / LOW |
| `retrieval_score` | 最高 cosine similarity |
| `phi_detected` | 是否偵測到輸入端 PHI |
| `warnings_count` | 警示訊息數量 |
| `requires_review` | 是否建議人工確認 |

`/logs` 頁面提供統計卡片（各信心等級次數、PHI 偵測次數）與可搜尋的查詢明細表。

---

## 安全設計說明

### PHI 去識別化（三層防護）

PHI（Protected Health Information）是指能識別病患身份的個人資訊。系統在三個時間點各自設有防護：

| 層級 | 觸發時機 | 處理方式 |
|------|----------|----------|
| 第一層 | 病歷建索引前（靜態） | 原始病歷的 PHI 全部替換為佔位符，確保 FAISS 向量索引中不含個資 |
| 第二層 | 使用者送出查詢時（動態） | 偵測輸入文字中的 PHI，替換後再送進 pipeline，並在 UI 顯示警示 |
| 第三層 | LLM 生成回答後 | 再掃描一次輸出，防止 LLM 從 context 重組出原始個資 |

#### 偵測的 PHI 類型與替換規則

| PHI 類型 | 偵測方式 | 替換結果 |
|----------|----------|----------|
| 台灣身分證 | 一個大寫字母 + 數字1或2 + 8位數字（如 A123456789） | `[ID_NUMBER]` |
| 手機號碼 | 09XX-XXX-XXX 格式 | `[PHONE]` |
| 市話 | 0X-XXXX-XXXX 格式 | `[PHONE]` |
| 生日 | 民國/西元年月日格式（如 85年3月12日、1996-03-12） | `[DOB]` |
| 台灣地址 | 含市/縣 + 區/鄉/鎮 + 路/街 + 號的完整地址 | `[ADDRESS]` |
| 緊急聯絡人姓名 | 「緊急聯絡人：」後方的 2-4 字中文姓名 | `[EMERGENCY_CONTACT]` |
| 病患姓名 | 「病患/患者/個案/病人」等關鍵字後方的 2-4 字中文姓名 | 建索引：`[姓名_A]`、`[姓名_B]`…；查詢/輸出掃描：`[PATIENT_NAME]` |

#### 姓名一致性處理

**建索引階段**（`deidentify_record`）：同一份病歷中出現多次的相同姓名，會被替換成**相同的佔位符**（如 `[姓名_A]`），確保去識別化後的文字邏輯仍然連貫，不會因為同一人有不同標籤而混亂。

**查詢 / 輸出掃描階段**（`scan_query`、`scan_output`）：屬於單次掃描，不需要跨句子的一致性，姓名統一替換為 `[PATIENT_NAME]`。

#### 取代而非刪除

系統採用**佔位符取代**而非直接刪除，原因是：
- 保留文字結構，LLM 仍能理解句子語意（「病患 [姓名_A] 主訴...」比「病患主訴...」資訊更完整）
- 佔位符可追蹤 PHI 出現的位置，便於稽核

### Reranker（兩階段檢索）

| 階段 | 方法 | 候選數 | 目的 |
|------|------|--------|------|
| 第一階段 | FAISS 向量搜尋（bi-encoder） | top-20 | 快速召回，確保不漏掉相關病例 |
| 第二階段 | cross-encoder 重排序 | top-5 | 精確排序，送給 LLM 的都是真正相關的 |

資料量增加時只需調高 `retrieval_top_k` 參數，不需修改架構。

### Hallucination 驗證機制

- **Faithfulness score**：回答內容是否能在 retrieved context 中找到依據
- **Confidence threshold**：Cosine similarity < 0.6 時，回答附加「資料庫中無充足相似病例」警示
- **強制引用**：LLM prompt 要求回答需標注所依據的病例類型（而非病患姓名）

### Human-in-the-Loop（練習版簡化）

本練習版在以下情況會在 UI 顯示橙色警示：
- Retrieval 相似度低（< 0.6）
- 問題涉及用藥劑量
- LLM 回答包含不確定語氣但仍給出建議

---

## 注意事項

1. 本系統**僅供學習練習**，不得用於真實臨床決策
2. 所有病歷均為人工合成，與任何真實病患無關
3. LLM 輸出具有不確定性，醫療決策需由執業醫師判斷
4. Llama-3.2-3B 為小型模型，醫療知識深度有限，正式場景應使用更大模型或醫療專用模型
