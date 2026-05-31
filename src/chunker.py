class MedicalChunker:
    """醫療文本章節切割（Section-based chunking）

    Small-to-Big Retrieval：
    - query chunk：chief_complaint + hpi + assessment（用於向量搜尋）
    - context chunk：assessment + plan + outcome + follow_up（用於 LLM 生成）
    """

    def chunk_record(self, record: dict) -> list[dict]:
        """
        回傳 list of chunks，格式：
        {
            "chunk_id": str,
            "record_id": str,
            "chunk_type": "query" | "context",
            "content": str,
            "metadata": dict
        }
        """
        record_id = record.get('record_id', record.get('id', 'unknown'))
        department = record.get('department', '')
        visit_date = record.get('visit_date', '')
        # build_index 會預先存好 visit_year（避免 visit_date 被 deidentifier 換成 [DOB]）
        visit_year = record.get('visit_year') or (visit_date[:4] if len(visit_date) >= 4 else visit_date)

        base_metadata = {
            'record_id': record_id,
            'department': department,
            'visit_year': visit_year,
        }

        chunks = []

        # --- Query chunk：症狀→診斷核心語意單元 ---
        query_parts = []
        if record.get('chief_complaint'):
            query_parts.append(f'主訴：{record["chief_complaint"]}')
        if record.get('hpi'):
            query_parts.append(f'現病史：{record["hpi"]}')
        if record.get('assessment'):
            query_parts.append(f'診斷：{record["assessment"]}')

        if query_parts:
            chunks.append({
                'chunk_id': f'{record_id}_query',
                'record_id': record_id,
                'chunk_type': 'query',
                'content': '\n'.join(query_parts),
                'metadata': {**base_metadata, 'chunk_type': 'query'},
            })

        # --- Context chunk：診斷→治療→結果完整敘述 ---
        context_parts = []
        if record.get('assessment'):
            context_parts.append(f'診斷：{record["assessment"]}')
        if record.get('plan'):
            plan = record['plan']
            if isinstance(plan, list):
                plan = '；'.join(plan)
            context_parts.append(f'治療計畫：{plan}')
        if record.get('medications'):
            meds = record['medications']
            if isinstance(meds, list):
                meds_strs = []
                for m in meds:
                    if isinstance(m, dict):
                        parts = [m.get('name', ''), m.get('dose', ''), m.get('frequency', '')]
                        meds_strs.append(' '.join(p for p in parts if p))
                    else:
                        meds_strs.append(str(m))
                meds = '、'.join(meds_strs)
            context_parts.append(f'用藥：{meds}')
        if record.get('outcome'):
            context_parts.append(f'結果：{record["outcome"]}')
        if record.get('follow_up'):
            context_parts.append(f'追蹤：{record["follow_up"]}')

        if context_parts:
            chunks.append({
                'chunk_id': f'{record_id}_context',
                'record_id': record_id,
                'chunk_type': 'context',
                'content': '\n'.join(context_parts),
                'metadata': {**base_metadata, 'chunk_type': 'context'},
            })

        return chunks
