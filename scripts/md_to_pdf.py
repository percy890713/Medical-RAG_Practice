"""
將 系統技術說明.md 轉換為 PDF，支援繁體中文。
使用 fpdf2 + NotoSansTC 字型。

用法：python scripts/md_to_pdf.py
"""

import re
from pathlib import Path
from fpdf import FPDF

FONT_PATH = r'C:\Windows\Fonts\NotoSansTC-VF.ttf'
MD_PATH   = Path('系統技術說明.md')
OUT_PATH  = Path('系統技術說明.pdf')

# ── 顏色 ─────────────────────────────────────────────
C_BLACK   = (30,  30,  30)
C_HEADING = (26,  86, 151)   # 深藍
C_CODE_BG = (245, 245, 245)
C_CODE_FG = (60,  60,  60)
C_TABLE_H = (26,  86, 151)
C_TABLE_R = (255, 255, 255)
C_TABLE_A = (240, 245, 255)   # 交替行淺藍
C_RULE    = (180, 180, 180)
C_WARN_BG = (255, 243, 205)
C_LINK    = (26,  86, 151)


class MdPDF(FPDF):
    def __init__(self):
        super().__init__(unit='mm', format='A4')
        self.add_font('NotoTC', style='',  fname=FONT_PATH)
        self.add_font('NotoTC', style='B', fname=FONT_PATH)
        self.set_margins(20, 20, 20)
        self.set_auto_page_break(auto=True, margin=22)
        self.add_page()
        self._set_color(*C_BLACK)

    # ── 顏色輔助 ─────────────────────────────────────
    def _set_color(self, r, g, b):
        self.set_text_color(r, g, b)

    def _fill_rect(self, x, y, w, h, r, g, b):
        self.set_fill_color(r, g, b)
        self.rect(x, y, w, h, 'F')

    # ── 分隔線 ────────────────────────────────────────
    def _hr(self):
        self.ln(2)
        self.set_draw_color(*C_RULE)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)
        self.set_line_width(0.2)
        self.set_draw_color(0, 0, 0)

    # ── 標題 ─────────────────────────────────────────
    def h1(self, text):
        self._set_color(*C_HEADING)
        self.set_font('NotoTC', 'B', 20)
        self.ln(4)
        self.multi_cell(0, 10, text, align='L')
        self._hr()
        self._set_color(*C_BLACK)
        self.ln(1)

    def h2(self, text):
        self.ln(5)
        self._fill_rect(self.l_margin, self.get_y(), self.w - self.l_margin - self.r_margin, 8, *C_HEADING)
        self._set_color(255, 255, 255)
        self.set_font('NotoTC', 'B', 13)
        self.set_x(self.l_margin + 2)
        self.cell(0, 8, text, ln=True)
        self._set_color(*C_BLACK)
        self.ln(2)

    def h3(self, text):
        self.ln(4)
        self._set_color(*C_HEADING)
        self.set_font('NotoTC', 'B', 11)
        self.cell(3, 6, '', ln=False)   # indent
        self.multi_cell(0, 6, text)
        self._set_color(*C_BLACK)
        self.ln(1)

    def h4(self, text):
        self.ln(3)
        self._set_color(*C_HEADING)
        self.set_font('NotoTC', 'B', 10)
        self.multi_cell(0, 5.5, text)
        self._set_color(*C_BLACK)

    # ── 正文 ─────────────────────────────────────────
    def body(self, text, indent=0):
        self.set_font('NotoTC', '', 9.5)
        self._set_color(*C_BLACK)
        x = self.l_margin + indent
        self.set_x(x)
        self.multi_cell(self.w - x - self.r_margin, 5.5, text, align='L')

    # ── 引用區塊（> 開頭）────────────────────────────
    def blockquote(self, text):
        y = self.get_y()
        bw = self.w - self.l_margin - self.r_margin
        self._fill_rect(self.l_margin, y, bw, 12, *C_WARN_BG)
        self.set_draw_color(*C_HEADING)
        self.set_line_width(0.8)
        self.line(self.l_margin, y, self.l_margin, y + 12)
        self.set_line_width(0.2)
        self.set_draw_color(0, 0, 0)
        self._set_color(100, 70, 0)
        self.set_font('NotoTC', '', 9)
        self.set_x(self.l_margin + 4)
        clean = text.lstrip('> ').strip()
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)
        self.multi_cell(bw - 6, 5.5, clean)
        self._set_color(*C_BLACK)
        self.ln(2)

    # ── 程式碼區塊 ────────────────────────────────────
    def code_block(self, lines: list[str]):
        text = '\n'.join(lines)
        bw   = self.w - self.l_margin - self.r_margin
        # 計算高度（估計行數）
        line_h = 4.5
        n_lines = len(lines)
        h = max(n_lines * line_h + 4, 10)
        y = self.get_y()
        if y + h > self.h - self.b_margin - 5:
            self.add_page()
            y = self.get_y()
        self._fill_rect(self.l_margin, y, bw, h, *C_CODE_BG)
        self._set_color(*C_CODE_FG)
        self.set_font('NotoTC', '', 8)
        self.set_x(self.l_margin + 3)
        self.multi_cell(bw - 4, line_h, text)
        self._set_color(*C_BLACK)
        self.ln(3)

    # ── 清單項目 ──────────────────────────────────────
    def list_item(self, text, level=0):
        indent = 5 + level * 8
        bullet = '•' if level == 0 else '–'
        self.set_font('NotoTC', '', 9.5)
        self._set_color(*C_BLACK)
        x = self.l_margin + indent
        # bullet
        self.set_x(x - 4)
        self.cell(4, 5.5, bullet, ln=False)
        # text（去掉 bold markers，簡化處理）
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        clean = re.sub(r'`(.+?)`', r'\1', clean)
        self.set_x(x)
        self.multi_cell(self.w - x - self.r_margin, 5.5, clean)

    # ── 表格 ─────────────────────────────────────────
    def table(self, rows: list[list[str]]):
        if not rows:
            return
        usable = self.w - self.l_margin - self.r_margin
        n_cols = max(len(r) for r in rows)
        col_w  = usable / n_cols

        for ri, row in enumerate(rows):
            is_header = (ri == 0)
            row_h = 6.5

            # 填充顏色
            fill = C_TABLE_H if is_header else (C_TABLE_A if ri % 2 == 0 else C_TABLE_R)
            self._fill_rect(self.l_margin, self.get_y(), usable, row_h, *fill)

            txt_color = (255, 255, 255) if is_header else C_BLACK
            self._set_color(*txt_color)
            style = 'B' if is_header else ''
            self.set_font('NotoTC', style, 8.5)

            x0 = self.l_margin
            y0 = self.get_y()
            for ci, cell in enumerate(row):
                clean = re.sub(r'\*\*(.+?)\*\*', r'\1', cell.strip())
                clean = re.sub(r'`(.+?)`', r'\1', clean)
                self.set_xy(x0 + ci * col_w + 1, y0)
                self.cell(col_w - 2, row_h, clean[:40], ln=False, align='L')
            self.ln(row_h)

        self._set_color(*C_BLACK)
        self.ln(3)


# ── 解析並渲染 ────────────────────────────────────────────────────

def strip_inline(text: str) -> str:
    """去掉 inline markdown（bold/code/link）"""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'`(.+?)`',       r'\1', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    return text


def render(pdf: MdPDF, md_text: str):
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # --- 空行 ---
        if not line.strip():
            pdf.ln(2)
            i += 1
            continue

        # --- 分隔線 ---
        if re.match(r'^-{3,}$', line.strip()):
            pdf._hr()
            i += 1
            continue

        # --- 標題 ---
        if line.startswith('#### '):
            pdf.h4(strip_inline(line[5:].strip()))
            i += 1; continue
        if line.startswith('### '):
            pdf.h3(strip_inline(line[4:].strip()))
            i += 1; continue
        if line.startswith('## '):
            pdf.h2(strip_inline(line[3:].strip()))
            i += 1; continue
        if line.startswith('# '):
            pdf.h1(strip_inline(line[2:].strip()))
            i += 1; continue

        # --- 引用區塊 ---
        if line.startswith('>'):
            pdf.blockquote(line)
            i += 1; continue

        # --- 程式碼區塊 ---
        if line.startswith('```'):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1
            pdf.code_block(code_lines)
            i += 1; continue

        # --- 表格 ---
        if '|' in line and line.strip().startswith('|'):
            table_rows = []
            while i < len(lines) and '|' in lines[i] and lines[i].strip().startswith('|'):
                row_line = lines[i].strip().strip('|')
                # 跳過分隔行（|---|---|）
                if re.match(r'^[\|\-\: ]+$', row_line):
                    i += 1; continue
                cells = [c.strip() for c in row_line.split('|')]
                table_rows.append(cells)
                i += 1
            pdf.table(table_rows)
            continue

        # --- 清單 ---
        m = re.match(r'^(\s*)[-*] (.+)', line)
        if m:
            level  = len(m.group(1)) // 2
            text   = m.group(2)
            pdf.list_item(strip_inline(text), level=level)
            i += 1; continue

        # --- 數字清單 ---
        m = re.match(r'^(\s*)\d+\. (.+)', line)
        if m:
            level = len(m.group(1)) // 2
            text  = m.group(2)
            pdf.list_item(strip_inline(text), level=level)
            i += 1; continue

        # --- 一般段落 ---
        pdf.body(strip_inline(line))
        i += 1


def main():
    md_text = MD_PATH.read_text(encoding='utf-8')
    pdf = MdPDF()
    render(pdf, md_text)
    pdf.output(str(OUT_PATH))
    print(f'PDF 已產生：{OUT_PATH}')


if __name__ == '__main__':
    main()
