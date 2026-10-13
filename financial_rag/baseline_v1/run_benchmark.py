"""用独立 PDF 金标分别评价召回与实际生成答案，不以关键词出现判为答对。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sys

from tqdm import tqdm

# ==================== 可编辑参数区：IDE 直接运行此文件 ====================
CURRENT_DIR = Path(__file__).resolve().parent
GOLD_PATH = CURRENT_DIR / "gold_cases.json"
CHUNKS_PATH = CURRENT_DIR / "chunks.json"
OUTPUT_DIR = CURRENT_DIR
TOP_K = 6
SHOW_PROGRESS = True
# 填写已有真实模型响应的 JSON 路径可离线重评；None 则实际调用问答引擎。
RESPONSES_PATH: Path | None = None
RESUME = False  # True 时仅续跑同题目、同语料哈希的本次已完成响应。
# =======================================================================

UNIT_FACTORS = {"元": Decimal(1), "万元": Decimal(10000), "亿元": Decimal(100000000),
                "%": Decimal(1), "吨": Decimal(1), "家": Decimal(1), "个": Decimal(1)}


def normalized(text: object) -> str:
    """返回去除排版空白和标点后的证据文本，保留数字、小数点和汉字。

    Args:
        text: PDF/切块/模型返回的原文；不能据此判断语义正确性。

    Returns:
        str：用于验证引用是否确实来自已召回原文的紧凑文本。
    """
    return re.sub(r"[^\w\u4e00-\u9fff.%+\-−]", "", str(text))


def decimal_value(value: object, unit: str) -> Decimal | None:
    """将明确单位的数字转成精确十进制基准值，拒绝隐式猜单位。

    Args:
        value: 数字或十进制字符串，允许千分位和末尾百分号。
        unit: 元、万元、亿元、%、吨、家或个；金额统一为元。

    Returns:
        Decimal 或 None：可比较的基准值，格式或单位不支持时返回 None。
    """
    if unit not in UNIT_FACTORS:
        return None
    try:
        parsed = Decimal(str(value).replace(",", "").replace("，", "").strip().rstrip("%"))
        return parsed * UNIT_FACTORS[unit] if parsed.is_finite() else None
    except InvalidOperation:
        return None


def evidence_matches(fact: dict, chunk: dict) -> bool:
    """核对召回块是否含目标公司、已核实物理页、指标行锚点与数值。

    Args:
        fact: 金标事实，包含 company、acceptable_pages、value、evidence_anchor。
        chunk: 引擎召回块，必须保留 company、page、content 全文。

    Returns:
        bool：仅表示该事实的证据被召回，不代表模型答案正确。
    """
    content = normalized(chunk.get("content", ""))
    return (chunk.get("company") == fact["company"]
            and chunk.get("page") in fact.get("acceptable_pages", [fact["page"]])
            and normalized(fact["value"]) in content
            and normalized(fact["evidence_anchor"]) in content)


def score_case(case: dict, result: dict, chunk_index: dict[str, dict]) -> dict:
    """分别评价证据召回、数值回答、引用及需要人工核验的子问题。

    1. 召回评分只看独立金标证据是否进入实际 retrieved_chunks。
    2. 答案评分核对 claims 的数值、单位、期间、口径与引用，并检查
       该数值确实出现在最终回答，不能把找到证据当成回答正确。
    3. 定性结论、题干歧义和自由叙述含义不使用关键词自动判对。

    Args:
        case: gold_cases.json 中的一题，facts 为独立 PDF 金标。
        result: 对同一个原始 question 执行问答所得的实际模型响应。
        chunk_index: 全量真实切块映射，用于还原召回块及核验引用。

    Returns:
        dict：逐事实召回、答案检查项、人工复核队列及原始回答。
    """
    retrieved = []
    for item in result.get("retrieved_chunks", []):
        chunk_id = item if isinstance(item, str) else item.get("chunk_id", "")
        known = chunk_index.get(chunk_id)
        # 只承认真实索引内的切块，响应不能自行伪造证据正文。
        if known:
            retrieved.append({**known, "score": item.get("score") if isinstance(item, dict) else None})
    by_id = {chunk["chunk_id"]: chunk for chunk in retrieved}
    claims = result.get("claims", [])
    answer = str(result.get("answer", ""))
    citations = result.get("citations", [])
    citation_ids = {c if isinstance(c, str) else c.get("chunk_id", "") for c in citations}
    invalid_citations = []
    for citation in citations:
        cid = citation if isinstance(citation, str) else citation.get("chunk_id", "")
        actual = by_id.get(cid)
        if actual is None or (isinstance(citation, dict) and any(
                key in citation and citation[key] != actual.get(key) for key in ("company", "page"))):
            invalid_citations.append(citation)
    fact_rows = []
    for fact in case.get("facts", []):
        recalled = [c["chunk_id"] for c in retrieved if evidence_matches(fact, c)]
        valid_metrics = {normalized(m) for m in [fact["metric"], *fact.get("metric_aliases", [])]}
        valid_scopes = {normalized(s) for s in [fact["scope"], *fact.get("scope_aliases", [])]}
        candidates = [c for c in claims if c.get("company") == fact["company"]
                      and normalized(c.get("metric", "")) in valid_metrics]
        checks = []
        for claim in candidates:
            cited = by_id.get(claim.get("chunk_id", ""), {})
            claim_quote = re.sub(r"\s+", "", str(claim.get("quote", "")))
            gold_value = decimal_value(fact["value"], fact["unit"])
            claim_unit = str(claim.get("unit", "")).replace("人民币", "").replace("％", "%")
            claim_value = decimal_value(claim.get("value", ""), claim_unit)
            dimension_ok = ((fact["unit"] in {"元", "万元", "亿元"} and claim_unit in {"元", "万元", "亿元"})
                            or claim_unit == fact["unit"] or {claim_unit, fact["unit"]} <= {"家", "个"})
            period = str(fact["period"])
            claim_period = re.sub(r"\s+", "", str(claim.get("period", "")))
            claim_period = re.sub(r"[（(]本期[）)]$", "", claim_period)
            if re.fullmatch(r"\d{4}", period):
                periods = {period, f"{period}年", f"{period}年度", f"{period}全年",
                           f"{period}年1—12月", f"{period}年1-12月", f"{period}-01-01至{period}-12-31"}
            else:
                periods = {period, f"{period[:4]}年12月31日", f"{period[:4]}年末", f"{period[:4]}年期末"}
            period_ok = claim_period in periods
            details = {
                "value_unit": dimension_ok and gold_value == claim_value and claim_value is not None,
                "period": period_ok,
                "scope": normalized(claim.get("scope", "")) in valid_scopes,
                "citation": bool(cited) and claim.get("page") == cited.get("page")
                    and cited.get("company") == fact["company"]
                    and claim.get("chunk_id") in citation_ids and evidence_matches(fact, cited),
                "quote": bool(claim_quote) and claim_quote in re.sub(r"\s+", "", str(cited.get("content", "")))
                    and normalized(fact["value"]) in normalized(claim_quote),
                "in_answer": normalized(claim.get("value", "")) in normalized(answer)
                    and bool(str(claim.get("value", ""))),
            }
            checks.append({"claim": claim, "checks": details, "passed": all(details.values())})
        fact_rows.append({"gold": fact, "recalled": bool(recalled), "recalled_chunk_ids": recalled,
                          "answer_verified": any(c["passed"] for c in checks), "candidate_checks": checks})

    # 定性题召回也须有公司、页码与原文，不能仅凭公司名出现计为通过。
    evidence_rows = []
    for evidence in case.get("evidence", []):
        matches = [c["chunk_id"] for c in retrieved if c.get("company") == evidence["company"]
                   and c.get("page") == evidence["page"]
                   and re.sub(r"\s+", "", evidence["quote"]) in re.sub(r"\s+", "", c.get("content", ""))]
        evidence_rows.append({**evidence, "recalled_chunk_ids": matches, "recalled": bool(matches)})
    evidence_total = len(fact_rows) or len(evidence_rows)
    evidence_hits = sum(r["recalled"] for r in fact_rows or evidence_rows)
    verified = sum(r["answer_verified"] for r in fact_rows)
    manual = list(case.get("manual_review", []))
    ranking = None
    if case.get("expected_rank"):
        rank = result.get("ranking")
        if isinstance(rank, list) and rank and isinstance(rank[0], dict):
            rank = [r.get("company") for r in rank]
        ranking = {"expected": case["expected_rank"], "actual": rank,
                   "verified": rank == case["expected_rank"]}
        if not rank:
            manual.append("模型未返回可核对的结构化 ranking；不能仅凭答案出现公司名称判排序正确。")
        elif not ranking["verified"]:
            manual.append("结构化 ranking 与按独立金标数值计算的排序不一致。")
    unsupported_claims = []
    for claim in claims:
        cited = by_id.get(claim.get("chunk_id", ""), {})
        quote = re.sub(r"\s+", "", str(claim.get("quote", "")))
        if (not quote or quote not in re.sub(r"\s+", "", str(cited.get("content", "")))
                or claim.get("company") != cited.get("company")
                or claim.get("page") != cited.get("page")
                or claim.get("chunk_id") not in citation_ids):
            unsupported_claims.append(claim)
    status = "failed"
    if case["gold_status"] == "pending_review":
        status = "pending_gold_review"
    elif case["kind"] == "out_of_corpus":
        status = "refusal_needs_review" if result.get("status") in {"insufficient_evidence", "refused", "out_of_scope"} and not claims else "failed"
    elif not fact_rows:
        status = "needs_manual_review"
    elif verified == len(fact_rows) and not unsupported_claims:
        status = "numeric_verified_needs_review" if manual else "numeric_verified"
    elif verified:
        status = "partial"
    if invalid_citations or (ranking is not None and not ranking["verified"]):
        status = "partial" if verified else "failed"
    if result.get("error"):
        status = "execution_error"
    return {"id": case["id"], "question": case["question"], "kind": case["kind"],
            "gold_status": case["gold_status"], "target_companies": case["target_companies"],
            "recall": {"matched": evidence_hits, "total": evidence_total,
                       "rate": evidence_hits / evidence_total if evidence_total else None},
            "answer_evaluation": {"status": status, "verified_facts": verified,
                                  "total_facts": len(fact_rows), "unsupported_claims": unsupported_claims,
                                  "invalid_citations": invalid_citations,
                                  "ranking": ranking, "manual_review": manual,
                                  "manual_review_completed": False},
            "fact_checks": fact_rows, "qualitative_evidence_checks": evidence_rows,
            "generated_answer": answer, "engine_status": result.get("status"),
            "citations": citations, "claims": claims,
            "retrieved_chunks": [{k: v for k, v in c.items() if k != "content"} for c in retrieved],
            "engine_metadata": {k: result[k] for k in ("model", "provider", "usage", "error", "coverage", "generation", "retrieval", "model_run", "validation_issues", "corpus_sha256") if k in result}}


def evaluate_benchmark(gold_path: Path = GOLD_PATH, chunks_path: Path = CHUNKS_PATH,
                       output_dir: Path = OUTPUT_DIR, top_k: int = TOP_K,
                       show_progress: bool = SHOW_PROGRESS, responses_path: Path | None = RESPONSES_PATH,
                       resume: bool = RESUME) -> dict:
    """运行实际问答或重评已有真实响应，输出可逐项复核的独立评测。

    Args:
        gold_path: 独立 PDF 金标，只用于评分，不传入检索/生成模块。
        chunks_path: 全文切块 JSON，用于真实检索和出处核验。
        output_dir: 保存评测 JSON、Markdown 与摘要的目录。
        top_k: 每题检索参数；实际全景公司配额由引擎决定。
        show_progress: 默认显示读取、问答和逐题评分进度。
        responses_path: 可选真实响应 JSON，未提供时调用实际问答引擎。
        resume: 是否续跑输出目录中同题目、同语料的部分真实模型响应。

    Returns:
        dict：召回率、结构化数值事实通过率和人工复核数量。

    Raises:
        ValueError: 语料只有前75页或响应与题目不对应时拒绝输出完整成绩。
    """
    if show_progress:
        print("[评测 1/3] 读取独立 PDF 金标与全文切块，核验题目和出处。", flush=True)
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = raw_chunks.get("chunks", []) if isinstance(raw_chunks, dict) else raw_chunks
    chunk_index = {c["chunk_id"]: c for c in chunks}
    manifest = json.loads(chunks_path.with_name("parse_manifest.json").read_text(encoding="utf-8"))
    reports = manifest.get("reports", [])
    company_counts = Counter(c["company"] for c in chunks)
    gold_pdfs = {p["company"]: p for p in gold["pdf_inventory"]}
    if (manifest.get("status") != "complete" or manifest.get("total_chunks") != len(chunks)
            or len(chunk_index) != len(chunks)
            or {r["company"] for r in reports} != set(company_counts)
            or set(company_counts) != set(gold_pdfs)
            or manifest.get("total_pages") != sum(r["total_pages"] for r in reports)
            or manifest.get("processed_pages") != manifest.get("total_pages")):
        raise ValueError("全文解析清单与切块/金标报告集不一致，拒绝评测不完整语料。")
    for report in reports:
        company = report["company"]
        actual_pages = {c["page"] for c in chunks if c["company"] == company}
        if (report.get("status") != "complete" or report.get("page_errors")
                or report.get("no_text_pages")
                or not report["total_pages"] == report["processed_pages"] == report["pages_with_text"]
                or actual_pages != set(range(1, report["total_pages"] + 1))
                or report["chunks"] != company_counts[company]
                or report["total_pages"] != gold_pdfs[company]["page_count"]
                or report["pdf_sha256"] != gold_pdfs[company]["sha256"]):
            raise ValueError(f"{company} 的全文页覆盖、错误记录、切块计数或PDF哈希不一致。")
    cases = gold["cases"]
    questions = [c["question"] for c in cases]
    output_dir.mkdir(parents=True, exist_ok=True)
    if show_progress:
        print(f"[评测 2/3] 对 {len(cases)} 道问题执行真实问答；模型调用可能需要等待。", flush=True)
    if responses_path:
        raw = json.loads(responses_path.read_text(encoding="utf-8"))
        responses = raw.get("responses", []) if isinstance(raw, dict) else raw
        if len(responses) != len(cases) or any(r.get("question") != q for r, q in zip(responses, questions)):
            raise ValueError("重评响应必须按顺序保存每题原始 question，并与金标题目完全对应。")
        mode = "regrade_saved_responses"
    else:
        if __package__:
            from .search_engine import answer_questions
        else:
            from search_engine import answer_questions
        responses = []
        checkpoint = output_dir / "evaluation_responses.json"
        if resume and checkpoint.exists():
            responses = json.loads(checkpoint.read_text(encoding="utf-8"))
            corpus_hash = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
            if (len(responses) > len(questions)
                    or any(r.get("question") != questions[i] or r.get("corpus_sha256") != corpus_hash
                           for i, r in enumerate(responses))):
                raise ValueError("断点响应与当前题目或语料哈希不一致，拒绝复用。")
        for index in tqdm(range(len(responses), len(questions)), total=len(questions),
                          initial=len(responses), desc="真实问答", disable=not show_progress):
            question = questions[index]
            if show_progress:
                print(f"\n[Q{index + 1:02}/{len(questions)}] {question}", flush=True)
            batch = answer_questions([question], chunks_path=chunks_path, top_k=top_k, show_progress=show_progress)
            if len(batch) != 1:
                raise ValueError("问答结果数量与题目不一致，不能生成完整成绩。")
            responses.append({**batch[0], "question": question})
            # 每题完成即留档，长模型调用被中断时仍保有此前真实响应。
            (output_dir / "evaluation_responses.json").write_text(json.dumps(responses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        mode = "live_engine"
    (output_dir / "evaluation_responses.json").write_text(json.dumps(responses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if show_progress:
        print("[评测 3/3] 分别核对召回、答案、口径和引用，定性结论进入人工复核。", flush=True)
    records = [score_case(case, result, chunk_index) for case, result in
               tqdm(zip(cases, responses), total=len(cases), desc="逐题评分", disable=not show_progress)]
    facts_total = sum(r["answer_evaluation"]["total_facts"] for r in records)
    facts_verified = sum(r["answer_evaluation"]["verified_facts"] for r in records)
    recall_total = sum(r["recall"]["total"] for r in records)
    recall_matched = sum(r["recall"]["matched"] for r in records)
    summary = {"evaluated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
               "mode": mode, "question_count": len(cases), "chunk_count": len(chunks),
               "pdf_count": len(reports), "total_pdf_pages": manifest["total_pages"],
               "fulltext_manifest_verified": True,
               "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
               "chunks_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
               "recall": {"matched": recall_matched, "total": recall_total,
                          "micro_rate": recall_matched / recall_total if recall_total else None},
               "answer_numeric_facts": {"verified": facts_verified, "total": facts_total,
                                        "rate": facts_verified / facts_total if facts_total else None},
               "answer_status_counts": dict(Counter(r["answer_evaluation"]["status"] for r in records)),
               "fully_manually_reviewed_questions": 0,
               "limitations": ["数值通过率不是整题正确率；定性解释、自由叙述、梯队边界及拒答说明仍需人工逐条审阅。",
                               "金标页未列出的其他有效出处可能造成严格评分漏判，应补证据后重评而非改为关键词命中。",
                               "Q03题干分类不在原文，保留待核口径状态；不计入已核验数值事实分母。"]}
    (output_dir / "evaluation_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 财报真实问答独立评测", "", f"执行时间：{summary['evaluated_at']}", "",
             f"题目 {len(cases)} 道；切块 {len(chunks)} 个；证据召回 {recall_matched}/{recall_total}；"
             f"结构化数值事实核验 {facts_verified}/{facts_total}。", "",
             "以上是不同维度的成绩，不能换算成‘9/10答对’。定性、歧义、排序和拒答说明均保留人工复核。", "",
             "| 题号 | 证据召回 | 数值事实核验 | 答案状态 |", "| --- | --- | --- | --- |"]
    for record in records:
        recall, answer_eval = record["recall"], record["answer_evaluation"]
        lines.append(f"| {record['id']} | {recall['matched']}/{recall['total']} | "
                     f"{answer_eval['verified_facts']}/{answer_eval['total_facts']} | {answer_eval['status']} |")
    for record in records:
        lines.extend(["", f"## {record['id']} {record['question']}", "", "### 实际回答", "", record["generated_answer"], "", "### 独立金标与逐项判分", ""])
        for item in record["fact_checks"]:
            fact = item["gold"]
            lines.append(f"- {fact['company']} / {fact['metric']} / {fact['scope']} / {fact['period']}："
                         f"{fact['value']} {fact['unit']}，PDF第{fact['page']}页；召回={item['recalled']}，答案核验={item['answer_verified']}。")
            lines.append(f"  原文：{fact['quote']}")
            if not item["answer_verified"]:
                failures = [", ".join(k for k, v in c["checks"].items() if not v) for c in item["candidate_checks"]]
                lines.append(f"  未通过项：{'；'.join(failures) if failures else '缺少对应公司和指标的明确 claim。'}")
        for todo in record["answer_evaluation"]["manual_review"]:
            lines.append(f"- 待人工核验：{todo}")
        lines.extend(["", "### 实际召回出处", ""])
        lines.extend(f"- {c['company']}，PDF第{c['page']}页，`{c['chunk_id']}`。" for c in record["retrieved_chunks"])
    (output_dir / "evaluation_records.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if show_progress:
        print(f"评测已保存：召回 {recall_matched}/{recall_total}；数值事实 {facts_verified}/{facts_total}；人工复核尚未完成。", flush=True)
    return summary


def main() -> None:
    """以 IDE、脚本或模块入口执行完整评测。

    Args:
        无参数；从顶部可编辑参数及命令行读取路径、top_k 和进度开关。

    Returns:
        None；保存同一格式的评测记录与摘要。
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--responses", type=Path, default=RESPONSES_PATH)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--resume", action="store_true", default=RESUME)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    evaluate_benchmark(args.gold, args.chunks, args.output_dir, args.top_k,
                       SHOW_PROGRESS and not args.no_progress, args.responses, args.resume)


if __name__ == "__main__":
    main()
