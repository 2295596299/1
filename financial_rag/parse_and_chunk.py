"""逐页解析完整年报，保留表格、物理 PDF 页码和可核验的覆盖记录。"""

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader
from tqdm import tqdm

if __package__:
    from .download_reports import REPORTS_METADATA
else:
    from download_reports import REPORTS_METADATA

# ==========================================
# 可编辑参数配置区：IDE 直接运行本文件即可重建全部年报。
# ==========================================
CURRENT_DIR = Path(__file__).resolve().parent
PDF_DIR = CURRENT_DIR / "pdf_reports"
OUTPUT_CHUNKS_PATH = CURRENT_DIR / "chunks.json"
MAX_PAGES_PER_REPORT: int | None = None  # None 解析全文；整数仅用于显式调试。
TEXT_CHUNK_SIZE = 900
TEXT_CHUNK_OVERLAP = 120
TABLE_ROWS_PER_CHUNK = 16
SHOW_PROGRESS = True
PARSER_VERSION = "fulltext-v3-chapters"
TABLE_SETTINGS = {"snap_tolerance": 6, "join_tolerance": 6}

CHAPTER_PATTERN = re.compile(r"^第[一二三四五六七八九十百]+节\s*[^\n\r]{1,65}$", re.MULTILINE)


def format_table_as_markdown(table: list[list[str | None]]) -> str:
    """将原始表格转为保留空单元格及列位置的 Markdown 表格。

    Args:
        table: pdfplumber 提取的二维单元格列表，shape 为 (行数, 列数)。

    Returns:
        str，包含表头、分隔行和全部非空数据行；空表返回空字符串。
    """
    rows = [[re.sub(r"\s+", " ", str(cell or "")).strip().replace("|", "\\|")
             for cell in row] for row in table]
    rows = [row for row in rows if any(row)]
    if not rows:
        return ""
    width = max(map(len, rows))
    lines = ["| " + " | ".join(row + [""] * (width - len(row))) + " |" for row in rows]
    return "\n".join([lines[0], "| " + " | ".join(["---"] * width) + " |", *lines[1:]])


def detect_page_chapter(page_text: str, previous_chapter: str) -> tuple[str, str]:
    """识别本页真实章节，并返回本页标签及应延续到下一页的章节。

    目录条目按引导点或末尾页码过滤，不能用标题数量排除真实同页多节。
    同页出现多节时本页保留所有标题，后续页从本页最后一节延续。

    Args:
        page_text: 按行提取的完整页面正文，不包含人为添加的引用前缀。
        previous_chapter: 上一页结束时的章节，用于本页没有新标题的情况。

    Returns:
        tuple[str, str]，依次为本页展示章节、后续页延续章节；不推断块内精确分界。
    """
    titles = []
    for match in CHAPTER_PATTERN.finditer(page_text):
        title = " ".join(match.group().split())
        if re.search(r"[.．…·]{2}|\s\d+\s*$", title):
            continue
        if title not in titles:
            titles.append(title)
    return (" / ".join(titles), titles[-1]) if titles else (previous_chapter, previous_chapter)


def parse_single_pdf(
    pdf_path: str | Path,
    max_pages: int | None = MAX_PAGES_PER_REPORT,
    show_progress: bool = SHOW_PROGRESS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """处理一份年报的每一页，返回证据切块和实际解析覆盖记录。

    1. 正文保留短科目行；按目录条目的格式过滤，真实同页多节保留组合章节。
    2. 长表按行切分，重复前两行以保留多层表头，并保存表格上方标题及单位。
    3. 页错误和无可提取文字的页面单独记录，不把遍历到末页视为完整提取。

    Args:
        pdf_path: 待解析的本地 PDF 路径，文件名前缀应为股票代码。
        max_pages: None 表示全文；正整数仅调试，status 必为 partial。
        show_progress: 是否显示阶段提示和逐页 tqdm 进度条，默认开启。

    Returns:
        tuple，第一项为切块列表，第二项为页数、错误及来源 manifest 字典。
        chunk 的 page 为从 1 开始的物理 PDF 页码，不是纸面印刷页码。

    Raises:
        ValueError: max_pages 不是正整数或文件名无法匹配已登记的财报来源。
    """
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages 必须为正整数或 None")
    path = Path(pdf_path)
    code = path.stem.split("_")[0]
    meta = next((item for item in REPORTS_METADATA if item["code"] == code), None)
    if meta is None:
        raise ValueError(f"未登记的财报来源: {path.name}")
    started = time.perf_counter()
    if show_progress:
        print(f"[解析] 读取 {meta['name']} PDF，核对总页数后逐页提取...", flush=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source_url = "https://static.cninfo.com.cn/" + meta["adjunct_url"]
    manifest: dict[str, Any] = {
        "company": meta["name"], "code": code, "year": meta["year"],
        "pdf_filename": path.name, "source_url": source_url, "pdf_sha256": digest,
        "total_pages": 0, "processed_pages": 0, "pages_with_text": 0,
        "pages_with_tables": 0, "no_text_pages": [], "page_errors": [],
        "chunks": 0, "status": "failed", "page_numbering": "physical_pdf_1_based",
    }
    chunks: list[dict[str, Any]] = []
    chapter = "前言/重要提示"
    try:
        manifest["total_pages"] = len(PdfReader(path).pages)
        with pdfplumber.open(path) as pdf:
            if len(pdf.pages) != manifest["total_pages"]:
                raise ValueError("pypdf 与 pdfplumber 读取的总页数不一致")
            limit = len(pdf.pages) if max_pages is None else min(max_pages, len(pdf.pages))
            for page_index in tqdm(range(limit), desc=meta["name"], unit="页", disable=not show_progress):
                page = pdf.pages[page_index]
                page_num = page_index + 1
                try:
                    # 1. 保留每页全文，单独记录空文本，避免悄悄遗漏扫描页。
                    page_text = page.extract_text() or ""
                    clean_text = "\n".join(line.strip() for line in page_text.splitlines() if line.strip())
                    if clean_text:
                        manifest["pages_with_text"] += 1
                    else:
                        manifest["no_text_pages"].append(page_num)
                    page_chapter, chapter = detect_page_chapter(clean_text, chapter)
                    common = {
                        "company": meta["name"], "code": code, "year": meta["year"],
                        "chapter": page_chapter, "page": page_num, "source_url": source_url,
                        "pdf_filename": path.name,
                    }
                    citation = f"【{meta['name']} {meta['year']}年报 | {page_chapter} | PDF第{page_num}页】"

                    # 2. 正文不删除短行；滑窗保留相邻指标名及数值的上下文。
                    for part, start in enumerate(range(0, len(clean_text), TEXT_CHUNK_SIZE - TEXT_CHUNK_OVERLAP), 1):
                        chunks.append({
                            **common, "chunk_id": f"{code}_p{page_num}_txt{part}", "type": "text",
                            "content": citation + "\n" + clean_text[start:start + TEXT_CHUNK_SIZE],
                        })

                    # 3. 原文保留表头及单位；不猜测合并单元格或跨页表的口径。
                    # 填色单元格边缘有约 3–5 点偏移，合并近邻线避免生造空列。
                    tables = page.find_tables(TABLE_SETTINGS)
                    if tables:
                        manifest["pages_with_tables"] += 1
                    for table_index, table in enumerate(tables, 1):
                        rows = table.extract()
                        if not rows:
                            continue
                        top = max(0, table.bbox[1] - 150)
                        context = page.crop((0, top, page.width, max(top + 0.1, table.bbox[1]))).extract_text() or ""
                        context = "\n".join(context.splitlines()[-6:])
                        headers, body = rows[:2], rows[2:]
                        for part, start in enumerate(range(0, max(1, len(body)), TABLE_ROWS_PER_CHUNK), 1):
                            markdown = format_table_as_markdown(headers + body[start:start + TABLE_ROWS_PER_CHUNK])
                            if markdown:
                                chunks.append({
                                    **common, "chunk_id": f"{code}_p{page_num}_tbl{table_index}_{part}",
                                    "type": "table", "content": citation + "\n" + context + "\n" + markdown,
                                })
                    manifest["processed_pages"] += 1
                except Exception as exc:
                    manifest["page_errors"].append({"page": page_num, "error": f"{type(exc).__name__}: {exc}"})
                finally:
                    page.close()
            if manifest["page_errors"]:
                manifest["status"] = "failed"
            elif max_pages is not None:
                manifest["status"] = "partial"
            elif manifest["no_text_pages"]:
                manifest["status"] = "needs_ocr"
            else:
                manifest["status"] = "complete"
    except Exception as exc:
        manifest["page_errors"].append({"page": None, "error": f"{type(exc).__name__}: {exc}"})
    manifest["chunks"] = len(chunks)
    manifest["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    if show_progress:
        print(f"[解析] {meta['name']}: {manifest['status']}, {manifest['processed_pages']}/{manifest['total_pages']} 页, {len(chunks)} 切块", flush=True)
    return chunks, manifest


def batch_parse_all_reports(
    pdf_dir: str | Path = PDF_DIR,
    output_path: str | Path = OUTPUT_CHUNKS_PATH,
    max_pages: int | None = MAX_PAGES_PER_REPORT,
    show_progress: bool = SHOW_PROGRESS,
    force: bool = False,
) -> list[dict[str, Any]]:
    """遍历全部年报并输出切块及覆盖清单，复用来源和配置均一致的公司缓存。

    Args:
        pdf_dir: PDF 所在目录，必须包含 REPORTS_METADATA 登记的全部报告。
        output_path: 切块 JSON 路径；parse_manifest.json 写在相同目录。
        max_pages: None 表示全文；正整数输出明确标为 partial 的调试语料。
        show_progress: 是否显示公司、页面级进度和写入阶段提示。
        force: 是否忽略缓存，重新解析全部 PDF。

    Returns:
        list[dict]，全部公司的切块；只有完整解析或显式调试成功时返回。

    Raises:
        RuntimeError: 缺少报告、PDF 解析失败，或存在需 OCR 的空文本页。
    """
    files = sorted(Path(pdf_dir).glob("*.pdf"))
    expected = {item["code"] for item in REPORTS_METADATA}
    actual = {path.stem.split("_")[0] for path in files}
    if actual != expected or len(files) != len(expected):
        raise RuntimeError(f"报告数量或代码不符: expected={sorted(expected)}, actual={sorted(actual)}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output.parent / "outputs" / "parse_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = {"version": PARSER_VERSION, "max_pages": max_pages, "text_size": TEXT_CHUNK_SIZE,
              "text_overlap": TEXT_CHUNK_OVERLAP, "table_rows": TABLE_ROWS_PER_CHUNK,
              "table_settings": TABLE_SETTINGS}
    if show_progress:
        print(f"[解析] 开始核验并解析 {len(files)} 份报告；页数限制={max_pages or '全文'}", flush=True)
    started = time.perf_counter()
    all_chunks, reports = [], []
    for path in tqdm(files, desc="报告", unit="份", disable=not show_progress):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        cache_path = cache_dir / (path.stem.split("_")[0] + ".json")
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() and not force else {}
        if (cache.get("config") == config and cache.get("report", {}).get("pdf_sha256") == digest
                and cache.get("report", {}).get("status") in {"complete", "partial"}):
            chunks, report = cache["chunks"], cache["report"]
            if show_progress:
                print(f"[缓存] {report['company']}: 已核对 SHA256 和解析配置", flush=True)
        else:
            chunks, report = parse_single_pdf(path, max_pages=max_pages, show_progress=show_progress)
            cache_path.write_text(json.dumps({"config": config, "report": report, "chunks": chunks}, ensure_ascii=False), encoding="utf-8")
        all_chunks.extend(chunks)
        reports.append(report)
    status = "failed" if any(r["status"] in {"failed", "needs_ocr"} for r in reports) else (
        "partial" if any(r["status"] == "partial" for r in reports) else "complete")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "parser_config": config,
        "status": status, "reports": reports, "total_chunks": len(all_chunks),
        "total_pages": sum(r["total_pages"] for r in reports),
        "processed_pages": sum(r["processed_pages"] for r in reports),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "coverage_note": "complete 表示逐页无错误且有文本；不代表表格结构或财务口径已经人工核对。",
    }
    if show_progress:
        print(f"[写入] {len(all_chunks)} 切块及逐报告覆盖清单...", flush=True)
    output.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    (output.parent / "parse_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if status == "failed":
        raise RuntimeError("存在解析失败或空文本页，请检查 parse_manifest.json；不得作为完整语料验收")
    if show_progress:
        print(f"[完成] {status}: {manifest['processed_pages']}/{manifest['total_pages']} 页, {len(all_chunks)} 切块", flush=True)
    return all_chunks


def verify_corpus(output_path: str | Path, pdf_dir: str | Path, show_progress: bool = True) -> dict[str, Any]:
    """核对已生成语料的全文覆盖、切块数量、页码和当前 PDF 哈希。

    Args:
        output_path: 已生成 chunks.json 路径，其目录内必须有 parse_manifest.json。
        pdf_dir: 原始 PDF 所在目录，用于页数与 SHA256 独立复核。
        show_progress: 是否显示加载提示和报告核验进度。

    Returns:
        dict，核验通过后的 manifest，可用于验收展示。

    Raises:
        RuntimeError: 覆盖不全、来源改变或切块数量/页码不一致。
    """
    if show_progress:
        print("[核验] 加载切块和覆盖清单，复核原始 PDF...", flush=True)
    output = Path(output_path)
    chunks = json.loads(output.read_text(encoding="utf-8"))
    manifest = json.loads((output.parent / "parse_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or len(chunks) != manifest["total_chunks"]:
        raise RuntimeError("语料未完整解析或切块数量不符")
    if {r["code"] for r in manifest["reports"]} != {r["code"] for r in REPORTS_METADATA}:
        raise RuntimeError("语料未包含全部登记公司")
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise RuntimeError("chunk_id 不唯一")
    for report in tqdm(manifest["reports"], desc="核验", unit="份", disable=not show_progress):
        path = Path(pdf_dir) / report["pdf_filename"]
        company_chunks = [chunk for chunk in chunks if chunk["code"] == report["code"]]
        total = len(PdfReader(path).pages)
        if (report["status"] != "complete" or report["page_errors"] or report["no_text_pages"]
                or total != report["total_pages"] or total != report["processed_pages"]
                or total != report["pages_with_text"] or len(company_chunks) != report["chunks"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != report["pdf_sha256"]
                or {chunk["page"] for chunk in company_chunks} != set(range(1, total + 1))):
            raise RuntimeError(f"核验失败: {report['company']}")
    if show_progress:
        print(f"[核验通过] {len(manifest['reports'])} 份, {manifest['total_pages']} 页, {len(chunks)} 切块", flush=True)
    return manifest


def main() -> None:
    """执行 IDE、脚本或模块入口的全文解析或现有语料核验。

    Args:
        无；参数来自顶部可编辑配置和命令行。

    Returns:
        None，解析或核验完成后退出；异常导致非零退出码。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_CHUNKS_PATH)
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES_PER_REPORT, help="仅调试；截断语料标为 partial")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true", help="忽略公司缓存并重建")
    parser.add_argument("--verify-only", action="store_true", help="复核现有全文语料，不重复解析")
    args = parser.parse_args()
    show_progress = SHOW_PROGRESS and not args.no_progress
    if args.verify_only:
        verify_corpus(args.output, args.pdf_dir, show_progress)
    else:
        batch_parse_all_reports(args.pdf_dir, args.output, args.max_pages, show_progress, args.force)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    main()
