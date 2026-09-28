"""
财报解析与结构化切块模块 (Parse & Chunk)

该模块读取 10 家白酒上市公司的 2023 年报 PDF，完成：
1. 提取文字段落，保留段落语义完整性；
2. 提取表格并还原为标准的 Markdown 行列格式（确保表头、数值与说明对齐）；
3. 追踪章节层级（如“第二节 主要财务指标”、“第三节 管理层讨论与分析”）；
4. 为每一个切块打上元数据标签：[公司, 代码, 章节, 页码, 类型]，并持久化保存为 chunks.json。
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pdfplumber

# 针对 Windows 控制台编码加固
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ==========================================
# 可编辑参数配置区
# ==========================================
CURRENT_DIR = Path(__file__).resolve().parent
PDF_DIR: str = str(CURRENT_DIR / "pdf_reports")
OUTPUT_CHUNKS_PATH: str = str(CURRENT_DIR / "chunks.json")
MAX_PAGES_PER_REPORT: int = 75  # 重点解析前75页（已完全覆盖主要财务指标、管理层讨论、主营构成与产销研发）
TEXT_CHUNK_SIZE: int = 500      # 文本切块目标字数
TEXT_CHUNK_OVERLAP: int = 80    # 重叠滑动窗口

CHAPTER_PATTERN = re.compile(r"第[一二三四五六七八九十百]+节\s+([^\n\r]+)")


def format_table_as_markdown(table: List[List[Optional[str]]]) -> str:
    """
    将 pdfplumber 提取的二维列表表格还原为干净的 Markdown 格式文本。

    清洗换行符、空值及跨行噪声，保持行列对齐与数值完整。

    Args:
        table: 二维字符串列表。

    Returns:
        还原后的 Markdown 表格字符串。
    """
    if not table or len(table) < 2:
        return ""

    cleaned_rows = []
    max_cols = 0
    for row in table:
        cleaned_row = []
        for cell in row:
            if cell is None:
                val = ""
            else:
                val = str(cell).replace("\r", " ").replace("\n", " ").strip()
                val = val.replace("|", "/")
            cleaned_row.append(val)
        if any(cleaned_row):
            max_cols = max(max_cols, len(cleaned_row))
            cleaned_rows.append(cleaned_row)

    if len(cleaned_rows) < 2 or max_cols == 0:
        return ""

    # 统一补齐列数
    normalized_rows = []
    for row in cleaned_rows:
        row.extend([""] * (max_cols - len(row)))
        normalized_rows.append(row)

    header = "| " + " | ".join(normalized_rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * max_cols) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in normalized_rows[1:]]

    return "\n".join([header, separator] + body_lines)


def parse_single_pdf(
    pdf_path: str,
    max_pages: int = MAX_PAGES_PER_REPORT,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """
    解析单份财报 PDF，提取表格与文本，并附加完整的章节、页码元数据。

    Args:
        pdf_path: PDF 文件本地路径。
        max_pages: 最大解析页数。
        show_progress: 是否打印解析状态。

    Returns:
        切块列表，每个元素包含 chunk_id, company, code, chapter, page, type, content。
    """
    file_name = Path(pdf_path).stem
    # 从文件名提取代码与公司名称，格式：600519_贵州茅台_2023年报
    parts = file_name.split("_")
    code = parts[0] if len(parts) > 0 else "000000"
    company = parts[1] if len(parts) > 1 else "未知公司"
    year = "2023"

    if show_progress:
        print(f"[解析切块] 正在解析: {company} ({code}) ...")

    chunks: List[Dict[str, Any]] = []
    current_chapter = "前言/重要提示"

    with pdfplumber.open(pdf_path) as pdf:
        total_p = min(len(pdf.pages), max_pages)

        for p_idx in range(total_p):
            page_num = p_idx + 1
            page = pdf.pages[p_idx]

            # 1. 抽取页面文本
            page_text = page.extract_text() or ""

            # 检查是否有新的章节标题
            chap_match = CHAPTER_PATTERN.search(page_text)
            if chap_match:
                current_chapter = chap_match.group(0).strip().replace("\n", " ")

            # 2. 抽取并还原表格
            tables = page.extract_tables() or []
            table_idx = 0
            for raw_tbl in tables:
                md_tbl = format_table_as_markdown(raw_tbl)
                if md_tbl and len(md_tbl) > 30:
                    table_idx += 1
                    chunks.append({
                        "chunk_id": f"{code}_p{page_num}_tbl{table_idx}",
                        "company": company,
                        "code": code,
                        "year": year,
                        "chapter": current_chapter,
                        "page": page_num,
                        "type": "table",
                        "content": f"【{company} 2023年报 表格 | 出处: {current_chapter} 第{page_num}页】\n{md_tbl}",
                    })

            # 3. 抽取正文并做定长重叠切块
            # 清除表格重复文本与连续空白
            clean_text = page_text.replace("\r", " ").strip()
            # 过滤过短的无意义行
            paragraphs = [p.strip() for p in clean_text.split("\n") if len(p.strip()) > 8]
            combined_text = "\n".join(paragraphs)

            if len(combined_text) > 40:
                # 按滑动窗口切块
                start = 0
                sub_idx = 0
                while start < len(combined_text):
                    end = min(start + TEXT_CHUNK_SIZE, len(combined_text))
                    chunk_text = combined_text[start:end].strip()
                    if len(chunk_text) > 30:
                        sub_idx += 1
                        chunks.append({
                            "chunk_id": f"{code}_p{page_num}_txt{sub_idx}",
                            "company": company,
                            "code": code,
                            "year": year,
                            "chapter": current_chapter,
                            "page": page_num,
                            "type": "text",
                            "content": f"【{company} 2023年报 正文 | 出处: {current_chapter} 第{page_num}页】\n{chunk_text}",
                        })
                    start += TEXT_CHUNK_SIZE - TEXT_CHUNK_OVERLAP

    if show_progress:
        print(f"[解析切块] [OK] {company} 解析完成: 提取 {len(chunks)} 个切块 (前 {total_p} 页)。")
    return chunks


def batch_parse_all_reports(
    pdf_dir: str = PDF_DIR,
    output_path: str = OUTPUT_CHUNKS_PATH,
    max_pages: int = MAX_PAGES_PER_REPORT,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """
    遍历指定目录下所有财报 PDF，统一解析切块并导出为 JSON。

    Args:
        pdf_dir: PDF 存放目录。
        output_path: 目标 chunks.json 保存路径。
        max_pages: 每份财报最大解析页数。
        show_progress: 是否输出进度。

    Returns:
        包含所有切块的列表。
    """
    pdf_files = sorted([
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ])

    if show_progress:
        print(f"[解析切块] 开始批量切块解析，发现 {len(pdf_files)} 份财报文件...")

    all_chunks = []
    for pdf_p in pdf_files:
        chunks = parse_single_pdf(pdf_p, max_pages=max_pages, show_progress=show_progress)
        all_chunks.extend(chunks)

    # 导出到 JSON 文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    if show_progress:
        print(f"[解析切块] [DONE] 全部切块完成！累计生成 {len(all_chunks)} 个带元数据切块。")
        print(f"         切块已保存至: {output_path}")

    return all_chunks


def main() -> None:
    """直接执行该脚本完成解析切块。"""
    batch_parse_all_reports()


if __name__ == "__main__":
    main()
