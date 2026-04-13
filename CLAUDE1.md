# CLAUDE.md — Ollama + Docker 升級任務

這份文件是針對**已完成的醫療病歷 RAG 系統**的升級任務說明。
原始專案的 CLAUDE.md 和 README.md 也在同一個資料夾，請一併參考。

升級目標只有兩件事：
1. 把 `src/generator.py` 的 LLM 從 Llama-3.2（HuggingFace 載入）換成 **Ollama**
2. 加入 **Docker Compose**，讓整個系統可以背景執行、關掉 terminal 也不會停

**其他所有檔案（pipeline、retriever、reranker、deidentifier、chunker、app.py、templates）完全不用動。**

---

## 任務一：修改 `src/generator.py`

### 改動前（原本）

原本是用 `transformers` 直接載入 Llama-3.2-3B 模型到記憶體：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
# 載入模型到 GPU/CPU，佔用大量記憶體
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", ...)
```

### 改動後（Ollama）

改成打 HTTP 請求給 Ollama，Ollama 負責管理模型：

```python
from openai import OpenAI

class MedicalGenerator:
    def __init__(self, model_name: str = "llama3.2", ollama_url: str = "http://ollama:11434"):
        self.client = OpenAI(
            base_url=f"{ollama_url}/v1",
            api_key="ollama"      # Ollama 不需要真實 key，但 OpenAI SDK 要求有值
        )
        self.model_name = model_name
        self.ollama_url = ollama_url

    def generate(self, query: str, retrieved_contexts: list[str], max_new_tokens: int = 512) -> str:
        context_text = "\n\n---\n\n".join(retrieved_contexts)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"症狀描述：{query}\n\n相似病例資料：\n{context_text}"}
            ],
            max_tokens=max_new_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content
```

### SYSTEM_PROMPT 保持不變

原本 `generator.py` 裡的 `SYSTEM_PROMPT` 文字完全不動，只是換呼叫方式。

### 注意：ollama_url 在 Docker 環境

- **Docker 環境**：`http://ollama:11434`（ollama 是 docker-compose 的 service 名稱）
- **本機直接跑**：`http://localhost:11434`

用環境變數控制，不要 hardcode：

```python
import os

class MedicalGenerator:
    def __init__(self, model_name: str = None, ollama_url: str = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2")
        ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.client = OpenAI(base_url=f"{ollama_url}/v1", api_key="ollama")
```

### requirements.txt 的修改

移除這些（不再需要）：
```
torch
transformers
accelerate
huggingface-hub
```

新增這個：
```
openai>=1.0.0
```

其他套件（faiss、sentence-transformers、fastapi、presidio 等）完全不動。

---

## 任務二：新增 Docker 相關檔案

### 需要新增的檔案

```
（新增）docker-compose.yml
（新增）Dockerfile
（新增）.dockerignore
（新增）scripts/ollama_setup.sh    # 自動拉取模型的腳本
```

---

### `docker-compose.yml`

三個 service：

```yaml
services:

  ollama:
    image: ollama/ollama
    container_name: medical_rag_ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama    # 模型快取，重啟不用重新下載
    restart: unless-stopped          # 當機自動重啟
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 5s
      retries: 5

  ollama-setup:
    image: ollama/ollama
    container_name: medical_rag_ollama_setup
    depends_on:
      ollama:
        condition: service_healthy    # 等 ollama 健康才執行
    volumes:
      - ollama_data:/root/.ollama
    entrypoint: >
      sh -c "
        ollama pull llama3.2 &&
        echo '模型下載完成'
      "
    environment:
      - OLLAMA_HOST=http://ollama:11434
    restart: "no"                    # 只執行一次，不重啟

  rag-app:
    build: .
    container_name: medical_rag_app
    ports:
      - "7860:7860"
    volumes:
      - ./data:/app/data             # 資料目錄掛載，索引和 log 持久化
    environment:
      - OLLAMA_URL=http://ollama:11434
      - OLLAMA_MODEL=llama3.2
      - HF_TOKEN=${HF_TOKEN}         # 從 .env 讀取
    depends_on:
      ollama:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860/"]
      interval: 15s
      timeout: 5s
      retries: 3

volumes:
  ollama_data:                       # 模型快取 volume，跨重啟保留
```

---

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安裝系統依賴（presidio 需要 spacy 的依賴）
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 先複製 requirements.txt 安裝依賴（利用 Docker layer cache）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 下載 spacy 中文模型（deidentifier 需要）
RUN python -m spacy download zh_core_web_sm || true

# 複製其他程式碼
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY templates/ ./templates/
COPY app.py .

# 建立資料目錄
RUN mkdir -p data/faiss_index

EXPOSE 7860

CMD ["python", "app.py"]
```

---

### `.dockerignore`

```
__pycache__/
*.pyc
*.pyo
.env
venv/
.git/
data/faiss_index/     # 索引在 container 內重建，不從外部複製
data/query_log.jsonl  # log 從 volume 掛載
*.md
```

---

### `scripts/ollama_setup.sh`

給本機（非 Docker）使用，手動拉取模型：

```bash
#!/bin/bash
# 確認 Ollama 已啟動
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "錯誤：Ollama 未啟動，請先執行 ollama serve"
    exit 1
fi

echo "拉取 llama3.2 模型..."
ollama pull llama3.2
echo "完成！"
```

---

## 啟動順序說明（給 Claude Code 參考，README 要寫清楚）

### Docker 啟動（推薦，關掉 terminal 也繼續跑）

```bash
# 第一次啟動（會下載模型，需要時間）
docker-compose up -d

# 查看狀態
docker-compose ps

# 查看 log
docker-compose logs -f rag-app

# 停止
docker-compose down
```

第一次 `ollama-setup` 這個 service 會自動下載 llama3.2 模型，之後重啟不會重新下載（存在 volume 裡）。

### 本機直接跑（開發測試用）

```bash
# 1. 啟動 Ollama（背景執行）
ollama serve &

# 2. 拉取模型（只需要一次）
bash scripts/ollama_setup.sh

# 3. 建立索引（只需要一次）
python scripts/build_index.py

# 4. 啟動 app
python app.py
```

---

## 注意事項

1. `ollama-setup` service 只是負責下載模型，下載完就結束，狀態會顯示 `Exit 0`，這是正常的
2. `rag-app` 第一次啟動時如果索引不存在，需要先執行 `build_index.py`
   - Docker 環境：`docker-compose exec rag-app python scripts/build_index.py`
   - 本機環境：`python scripts/build_index.py`
3. `data/` 目錄用 volume 掛載，索引和 log 在重啟後會保留
4. 模型換成別的（例如 `llama3.1` 或 `qwen2.5`）只需改 `OLLAMA_MODEL` 環境變數，不用改程式碼
5. `HF_TOKEN` 已經不需要了（不再直接載入 Llama），可以從 requirements 和環境變數中移除，但保留不影響運作
