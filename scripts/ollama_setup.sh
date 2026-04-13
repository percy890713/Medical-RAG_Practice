#!/bin/bash
# 本機（非 Docker）環境手動拉取 Ollama 模型

MODEL=${OLLAMA_MODEL:-llama3.2}

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "錯誤：Ollama 未啟動，請先執行 ollama serve"
    exit 1
fi

echo "拉取 ${MODEL} 模型..."
ollama pull "${MODEL}"
echo "完成！"
