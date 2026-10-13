"""从当前评测与独立审阅生成可复核的一页作业结论。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from html import escape
import json
from pathlib import Path
import subprocess
import sys

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


# ==================== 可编辑参数区：IDE 直接运行本文件 ====================
SOURCE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SOURCE_DIR
PREVIEW_DIR = SOURCE_DIR / "outputs" / "submission_preview"
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
BOLD_FONT_PATH = Path("C:/Windows/Fonts/msyhbd.ttc")
PDFTOPPM = Path("C:/poppler/Library/bin/pdftoppm.EXE")
SHOW_PROGRESS = True
RENDER_PREVIEW = True
REPOSITORY_URL = "https://github.com/2295596299/1"

STATUS_LABELS = {
    "supported": "原文支持", "supported_at_reported_precision": "披露精度支持",
    "partial": "部分回答", "unanswered": "未完成回答",
    "scope_ambiguity_handled": "已说明口径歧义", "correct_refusal": "正确拒答",
    "incorrect": "存在事实错误",
}


def read_submission_data(source_dir: Path) -> dict:
    """读取同一批实际评测、全文记录与独立审阅，整理一页结论所需字段。

    金标只用于统计问题类别，不读取其答案作为模型成绩。响应文件只用于
    哈希一致性核验，防止重测期间把上一轮审阅配到新一轮回答上。

    Args:
        source_dir: 包含评测、审阅、全文解析清单与金标的目录。

    Returns:
        dict：本次指标、复核分类、动态失败案例、评测时间和输出段落。

    Raises:
        ValueError：审阅与响应哈希不一致，或逐题审阅尚不完整。
    """
    files = {
        name: json.loads((source_dir / filename).read_text(encoding="utf-8"))
        for name, filename in {
            "summary": "evaluation_summary.json", "review": "manual_review.json",
            "manifest": "parse_manifest.json", "gold": "gold_cases.json",
        }.items()
    }
    summary, review, manifest = files["summary"], files["review"], files["manifest"]
    response_hash = hashlib.sha256((source_dir / "evaluation_responses.json").read_bytes()).hexdigest()
    if review.get("responses_sha256") != response_hash:
        raise ValueError("当前审阅与实际回答不是同一批，请等待本轮逐题复核完成后再生成结论。")
    if len(review.get("records", [])) != summary["question_count"]:
        raise ValueError("独立审阅题数与评测题数不一致，暂不覆盖结论。")
    cases = {case["id"]: case for case in files["gold"]["cases"]}
    cross_ids = {key for key, case in cases.items() if len(case["target_companies"]) > 1}
    counts = Counter(record["answer_review_status"] for record in review["records"])
    count_text = "；".join(f"{STATUS_LABELS.get(key, key)} {value} 题" for key, value in counts.items()) + "。"
    examples = []
    for is_cross, category in ((False, "单公司"), (True, "全景/跨公司")):
        candidates = [record for record in review["records"]
                      if (record["id"] in cross_ids) == is_cross
                      and record["answer_review_status"] in {"partial", "unanswered", "incorrect"}]
        if candidates:
            chosen = max(candidates, key=lambda record: len(cases[record["id"]]["target_companies"]))
            detail = (chosen.get("limitation_or_failure") or chosen["finding"]).replace("ranking为空", "排序未给出")
            if len(detail) > 145:
                detail = detail[:144] + "…"
            examples.append(f"{category} {chosen['id']}：{STATUS_LABELS[chosen['answer_review_status']]}。{detail}")
        else:
            examples.append(f"{category}：本轮独立审阅未列出未答、部分回答或事实错误案例；这不代表后续问题不会失误。")
    recall, numeric = summary["recall"], summary["answer_numeric_facts"]
    companies = len(manifest["reports"])
    conclusion = "RAG 适合可追溯查证；跨公司需统一口径、逐家取证，仍会漏召回或误读。未实施方向 B，不比较微调的实测效果。"
    return {
        "title": "财报查证：先核对证据与口径",
        "subtitle": f"作业 3 · 方向 A | {summary['question_count']} 道真实问题，含 {len(cross_ids)} 道全景/跨公司题",
        "evaluated_at": summary["evaluated_at"], "conclusion": conclusion,
        "cards": [
            (f"{manifest['processed_pages']:,}/{manifest['total_pages']:,}", "PDF 物理页覆盖"),
            (f"{recall['matched']}/{recall['total']}", "金标证据召回"),
            (f"{numeric['verified']}/{numeric['total']}", numeric.get("metric_name", "严格结构化契约")),
        ],
        "sections": [
            ("01  全文与真实问答链", [
                f"{companies} 份 2023 年报，{manifest['total_chunks']:,} 个切块；BGE 中文向量 + BM25，经 RRF 融合，每家公司独立 Top-6 取证，再由本机 Codex CLI 真实生成。使用 CLI 默认模型，本轮未记录明确模型 ID。",
                "全文遍历不保证表格完全准确；原文引用与 PDF 物理页可回查，表头、单位、期间、合并/母公司口径仍须核对。",
            ]),
            ("02  独立逐题审阅分类", [
                count_text,
                "分类来自 Codex 独立逐题 PDF 审阅，未经课程老师或用户本人验收。自动字段核验要求数值、单位、期间、口径与引用共同满足规则；低分不能直接解释为所有数字错误，也不是整题正确率。",
            ]),
            ("03  本轮不足与复盘", examples),
            ("04  交付范围", [
                "演示运行于 http://127.0.0.1:8765/rag.html；公开 Vercel 页面尚未更新为此本机服务。新问题、原文证据及评测记录均可留档复核；本报告不预设正确率。",
            ]),
        ],
        "provenance": "依据：parse_manifest.json、evaluation_summary.json、manual_review.json；跨公司题数按独立金标题目范围统计。",
        "response_hash": response_hash,
    }


def write_submission(data: dict, output_dir: Path, preview_dir: Path,
                     render_preview: bool = True, show_progress: bool = True) -> dict:
    """将同一份结论数据输出为 Markdown、HTML 和一页中文 PDF，并渲染预览。

    Args:
        data: read_submission_data 返回的动态评测结论。
        output_dir: 三种最终文件的存放目录，沿用 one_page_conclusion 文件名。
        preview_dir: PNG 预览目录，应处于被 Git 忽略的 outputs 下。
        render_preview: 是否使用 Poppler 生成 PDF 页面 PNG。
        show_progress: 是否输出文件生成和渲染阶段提示。

    Returns:
        dict：包含最终文件路径、PDF 页数，以及可选预览 PNG 路径。

    Raises:
        ValueError：生成的 PDF 超过一页。
        FileNotFoundError：缺少配置的中文字体或 Poppler。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {kind: output_dir / f"one_page_conclusion.{kind}" for kind in ("md", "html", "pdf")}
    paths["submission_text"] = output_dir / "submission_text.txt"
    paths["submission_text"].write_text(
        f"方向：A\n代码仓库：{REPOSITORY_URL}\n一句话结论：{data['conclusion']}\n", encoding="utf-8"
    )
    if show_progress:
        print("[结论 2/3] 正在生成同源 Markdown、HTML 与中文 PDF…", flush=True)
    lines = [f"# {data['title']}", "", data["subtitle"], "", f"评测时间：{data['evaluated_at']}",
             "", f"**一句话结论：{data['conclusion']}**", "",
             "| 维度 | 本次记录 |", "| --- | --- |"]
    lines.extend(f"| {label} | {value} |" for value, label in data["cards"])
    for title, paragraphs in data["sections"]:
        lines.extend(["", f"## {title}", "", "\n\n".join(paragraphs)])
    lines.extend(["", data["provenance"], "", f"[代码仓库]({REPOSITORY_URL})", ""])
    paths["md"].write_text("\n".join(lines), encoding="utf-8")

    cards_html = "".join(f"<div class='metric'><strong>{escape(value)}</strong><span>{escape(label)}</span></div>" for value, label in data["cards"])
    sections_html = "".join(f"<section><h2>{escape(title)}</h2>" + "".join(f"<p>{escape(p)}</p>" for p in paragraphs) + "</section>" for title, paragraphs in data["sections"])
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{escape(data['title'])}</title>
<style>
@page {{ size: A4; margin: 14mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: 'Microsoft YaHei', sans-serif; color: #23334a; margin: 0; background: #edf2f7; }}
main {{ max-width: 760px; padding: 40px; margin: 20px auto; background: white; }}
h1 {{ font-size: 23px; margin: 0 0 8px; color: #123254; }}
.meta, footer {{ font-size: 12.7px; line-height: 1.6; color: #526277; }}
.conclusion {{ font-size: 15px; line-height: 1.65; padding: 14px 16px; background: #eaf2fb; border-left: 4px solid #347fc0; margin: 16px 0; }}
.metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 16px 0; }}
.metric {{ text-align: center; background: #f4f7fa; padding: 12px 4px; }}
.metric strong {{ display: block; font-size: 25px; color: #123254; }}
.metric span {{ font-size: 12.7px; }}
h2 {{ font-size: 15px; margin: 17px 0 7px; color: #17638c; }}
p {{ font-size: 13.5px; line-height: 1.65; margin: 6px 0; }}
footer {{ border-top: 1px solid #d6e0ec; margin-top: 18px; padding-top: 10px; }}
a {{ color: #17638c; }}
@media print {{ body {{ background: white; }} main {{ margin: 0; padding: 0; max-width: none; }} }}
</style></head><body><main>
<h1>{escape(data['title'])}</h1><div class="meta">{escape(data['subtitle'])}<br>评测时间：{escape(data['evaluated_at'])}</div>
<div class="conclusion">{escape(data['conclusion'])}</div><div class="metrics">{cards_html}</div>
{sections_html}<footer>{escape(data['provenance'])}<br><a href="{REPOSITORY_URL}">代码仓库</a></footer>
</main></body></html>"""
    paths["html"].write_text(html, encoding="utf-8")

    for name, font_path in (("Chinese", FONT_PATH), ("ChineseBold", BOLD_FONT_PATH)):
        if not font_path.is_file():
            raise FileNotFoundError(f"缺少中文字体：{font_path}")
        pdfmetrics.registerFont(TTFont(name, str(font_path), subfontIndex=0))
    body = ParagraphStyle("body", fontName="Chinese", fontSize=10.6, leading=15.8,
                          textColor=colors.HexColor("#23334a"), wordWrap="CJK", spaceAfter=5)
    title_style = ParagraphStyle("title", parent=body, fontName="ChineseBold", fontSize=20, leading=27, spaceAfter=6)
    meta = ParagraphStyle("meta", parent=body, fontSize=9.5, leading=14, textColor=colors.HexColor("#526277"))
    heading = ParagraphStyle("heading", parent=body, fontName="ChineseBold", fontSize=11.5, leading=17,
                             spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#17638c"))
    callout = ParagraphStyle("callout", parent=body, fontName="ChineseBold", fontSize=11, leading=16, spaceAfter=0)
    metric_number = ParagraphStyle("metricnumber", parent=body, fontName="ChineseBold", fontSize=20, leading=25, alignment=TA_CENTER)
    metric_label = ParagraphStyle("metriclabel", parent=meta, alignment=TA_CENTER, spaceAfter=0)
    page_width = A4[0]
    usable = page_width - 76
    story = [Paragraph(escape(data["title"]), title_style), Paragraph(escape(data["subtitle"]), meta),
             Paragraph("评测时间：" + escape(data["evaluated_at"]), meta), Spacer(1, 8)]
    box = Table([[Paragraph(escape(data["conclusion"]), callout)]], colWidths=[usable])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eaf2fb")),
                            ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#347fc0")),
                            ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                            ("TOPPADDING", (0, 0), (-1, -1), 11), ("BOTTOMPADDING", (0, 0), (-1, -1), 11)]))
    story.extend([box, Spacer(1, 12)])
    metrics = Table([[[Paragraph(escape(value), metric_number), Paragraph(escape(label), metric_label)]
                      for value, label in data["cards"]]], colWidths=[usable / 3] * 3)
    metrics.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f7fa")),
                                ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                                ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(metrics)
    for title, paragraphs in data["sections"]:
        story.append(Paragraph(escape(title), heading))
        story.extend(Paragraph(escape(paragraph), body) for paragraph in paragraphs)
    story.extend([Spacer(1, 9), Paragraph(escape(data["provenance"]), meta),
                  Paragraph(f'<link href="{REPOSITORY_URL}" color="#17638c">代码仓库</link>', meta)])
    document = SimpleDocTemplate(str(paths["pdf"]), pagesize=A4, leftMargin=38, rightMargin=38,
                                 topMargin=33, bottomMargin=30, title=data["title"], author="")
    document.build(story)
    pages = len(PdfReader(str(paths["pdf"])).pages)
    if pages != 1:
        raise ValueError(f"结论生成了 {pages} 页，请缩短内容后重新排版，不能直接交付。")
    result = {kind: str(path) for kind, path in paths.items()}
    result["pages"] = pages
    if render_preview:
        if not PDFTOPPM.is_file():
            raise FileNotFoundError(f"未找到 PDF 渲染器：{PDFTOPPM}")
        preview_dir.mkdir(parents=True, exist_ok=True)
        prefix = preview_dir / "one_page_conclusion"
        if show_progress:
            print("[结论 3/3] 正在渲染一页 PDF，供检查版式与中文字体…", flush=True)
        subprocess.run([str(PDFTOPPM), "-png", "-r", "130", "-singlefile", str(paths["pdf"]), str(prefix)],
                       check=True, capture_output=True, timeout=60, shell=False)
        result["preview"] = str(prefix.with_suffix(".png"))
    return result


def main() -> None:
    """从 IDE、脚本或模块入口生成当前批次的一页结论和预览。

    Args:
        无参数。可编辑顶部路径；命令行支持 --source-dir、--output-dir、--no-progress。

    Returns:
        None：写出三种结论文件并打印路径，不修改评测、金标或审阅结果。
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    progress = SHOW_PROGRESS and not args.no_progress
    if progress:
        print("[结论 1/3] 正在读取本轮评测、全文覆盖与独立审阅，核对批次一致性…", flush=True)
    data = read_submission_data(args.source_dir)
    outputs = write_submission(data, args.output_dir, PREVIEW_DIR, RENDER_PREVIEW, progress)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
