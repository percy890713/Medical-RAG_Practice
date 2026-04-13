FROM python:3.11-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 先複製 requirements.txt 安裝依賴（利用 Docker layer cache）
COPY requirements_ollama.txt .
RUN pip install --no-cache-dir -r requirements_ollama.txt

# 下載 spacy 中文模型（deidentifier 需要）
RUN python -m spacy download zh_core_web_sm || true

# 複製程式碼
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY templates/ ./templates/
COPY app.py .

# 建立資料目錄
RUN mkdir -p data/faiss_index

EXPOSE 7860

CMD ["python", "app.py"]
