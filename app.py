import json
import os
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.pipeline import MedicalRAGPipeline

app = FastAPI(title='醫療病歷 RAG 系統')
templates = Jinja2Templates(directory='templates')

LOG_PATH = Path('data/query_log.jsonl')
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_EXAMPLES = [
    '患者主訴胸痛伴隨呼吸困難，有高血壓病史，心電圖顯示ST段上升',
    '病人出現持續發燒三天、咳嗽有痰，胸部X光顯示肺部浸潤',
    '糖尿病患者血糖控制不佳，HbA1c 9.5%，四肢出現麻木感',
    '患者頭痛眩暈反覆發作，伴隨噁心嘔吐，有偏頭痛家族史',
]

pipeline: MedicalRAGPipeline | None = None
_pipeline_error: str = ''


def load_pipeline():
    global pipeline, _pipeline_error
    if pipeline is not None:
        return
    index_path = Path('data/faiss_index/index.faiss')
    if not index_path.exists():
        _pipeline_error = (
            f'索引尚未建立，請先執行：python scripts/build_index.py'
            f'（找不到 {index_path}）'
        )
        return
    try:
        pipeline = MedicalRAGPipeline()
        _pipeline_error = ''
    except Exception as e:
        _pipeline_error = f'Pipeline 載入失敗：{e}'


def append_log(entry: dict):
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def format_cases(cases: list) -> list:
    out = []
    for c in cases:
        dept = c['metadata'].get('department', '')
        year = c['metadata'].get('visit_year', '')
        meta = ' | '.join(filter(None, [dept, str(year) if year else '']))
        rerank = c.get('rerank_score')
        out.append({
            'rank': c['rank'],
            'score': f"{c['score']:.1%}",
            'rerank_score': f"{rerank:.2f}" if rerank is not None else None,
            'meta': meta,
            'summary': c.get('summary', ''),
            'context': c.get('context', ''),
        })
    return out


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, name='index.html', context={'examples': _EXAMPLES}
    )


class QueryRequest(BaseModel):
    query: str


@app.post('/query')
async def query_endpoint(body: QueryRequest):
    query = body.query.strip()
    if not query:
        return JSONResponse({'error': '請輸入症狀描述'}, status_code=400)

    load_pipeline()
    if _pipeline_error:
        return JSONResponse({'error': _pipeline_error}, status_code=503)

    try:
        result = pipeline.query(query)
    except Exception as e:
        return JSONResponse({'error': f'系統錯誤：{e}'}, status_code=500)

    # 寫入查詢 log
    append_log({
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'query': query,
        'confidence': result.get('confidence', ''),
        'retrieval_score': round(float(result.get('retrieval_score', 0)), 4),
        'warnings_count': len(result.get('warnings', [])),
        'requires_review': bool(result.get('requires_review', False)),
    })

    warnings: list[str] = []
    if result.get('phi_detected_in_query'):
        warnings.append('查詢中偵測到個人識別資訊，已自動遮蔽。')
    warnings.extend(result.get('warnings', []))
    if result.get('requires_review'):
        warnings.append('本次回答建議由醫療專業人員進一步確認。')

    return {
        'confidence': result.get('confidence', ''),
        'answer': result.get('answer', ''),
        'retrieved_cases': format_cases(result.get('retrieved_cases', [])),
        'warnings': warnings,
        'retrieval_score': round(float(result.get('retrieval_score', 0)), 4),
        'faithfulness_score': round(float(result.get('faithfulness_score', 0)), 4),
    }


@app.get('/logs', response_class=HTMLResponse)
async def logs_page(request: Request):
    entries: list[dict] = []
    if LOG_PATH.exists():
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    entries.reverse()  # 最新在前
    return templates.TemplateResponse(
        request=request, name='logs.html', context={'entries': entries}
    )


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=7860, reload=False)
