"""全文 BGE 向量 + BM25 检索，按公司取证，再由大模型生成可核对的答案。"""

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from financial_rag.embeddings import LocalEmbedding, MODEL_REPO, MODEL_REVISION, MAX_TOKENS

# ==================== 可编辑参数区 ====================
CURRENT_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = CURRENT_DIR / "chunks.json"
INDEX_DIR = CURRENT_DIR / "vector_index"
DEFAULT_QUESTION = "贵州茅台2023年合并与母公司营业收入分别是多少？"
DEFAULT_TOP_K = 6
BATCH_SIZE = 16
SHOW_PROGRESS = True
PIPELINE_VERSION = "v2"
NUMBER_PATTERN = re.compile(r"(?<![\d.])(?P<value>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(?P<unit>亿元|万元|元|%|吨|家(?!\s*(?:公司|企业)))?(?![/／])")
CURRENCY_FACTORS = {"元": Decimal(1), "万元": Decimal(10000), "亿元": Decimal(100000000)}
_BENCHMARK_CACHE = {}
COMPANY_ALIASES = {"贵州茅台": ["茅台", "600519"], "五粮液": ["000858"], "泸州老窖": ["老窖", "000568"], "山西汾酒": ["汾酒", "600809"], "洋河股份": ["洋河", "002304"], "古井贡酒": ["古井", "000596"], "舍得酒业": ["舍得", "600702"], "今世缘": ["603369"], "迎驾贡酒": ["迎驾", "603198"], "口子窖": ["603589"]}


def tokenize_chinese(text: str) -> list[str]:
    """生成中文双字词元以及英文、数字词元，用于完整切块的 BM25 统计。

    Args:
        text: 问题或财报原文，不跨标点拼接双字词。

    Returns:
        list[str]: 用于倒排索引的词元列表。
    """
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())
    for part in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
        if len(part) == 1:
            tokens.append(part)
    return tokens


def build_vector_index(chunks_path: str | Path = CHUNKS_PATH, show_progress: bool = True) -> dict:
    """为全部切块建立本地 BGE 向量索引，按语料和模型摘要缓存批次。

    Args:
        chunks_path: 全文 chunks.json 路径。
        show_progress: 是否显示模型下载和向量批次进度。

    Returns:
        dict: 记录模型、语料摘要、向量维度、截断限制及数量的索引清单。

    Raises:
        RuntimeError: 语料为空或向量缓存无效。
    """
    raw = Path(chunks_path).read_bytes()
    chunks = json.loads(raw)
    if not chunks:
        raise RuntimeError("语料为空，请先执行全文解析。")
    signature = {"corpus_sha256": hashlib.sha256(raw).hexdigest(), "model": MODEL_REPO, "revision": MODEL_REVISION, "max_tokens": MAX_TOKENS, "pooling": "CLS_L2"}
    cache_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    cache = INDEX_DIR / "batches" / cache_key
    cache.mkdir(parents=True, exist_ok=True)
    if show_progress:
        print(f"[向量建库] {len(chunks)} 个全文切块；模型 {MODEL_REPO}。", flush=True)
    encoder = LocalEmbedding(show_progress=show_progress)
    matrices = []
    for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="本地 BGE 向量", disable=not show_progress):
        target = cache / f"{start:08d}.npy"
        texts = [c["content"] for c in chunks[start:start + BATCH_SIZE]]
        if target.exists():
            batch = np.load(target, allow_pickle=False)
        else:
            batch = encoder.encode(texts)
            np.save(target, batch, allow_pickle=False)
        if len(batch) != len(texts) or not np.isfinite(batch).all():
            raise RuntimeError("缓存向量不完整，请移除对应批次后重建。")
        matrices.append(batch)
    vectors = np.concatenate(matrices)
    manifest = {**signature, "chunk_count": len(chunks), "dimensions": int(vectors.shape[1]), "model_sha256": encoder.model_sha256, "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "vector_input_note": "每块语义编码最多512 tokens；BM25索引完整文本。"}
    np.save(INDEX_DIR / "vectors.npy", vectors, allow_pickle=False)
    (INDEX_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


class FinancialHybridRetriever:
    """BGE embedding 余弦相似度与 BM25 倒排检索，通过 RRF 融合排序。"""

    def __init__(self, chunks_path: str | Path = CHUNKS_PATH, show_progress: bool = True, lexical_only: bool = False):
        """加载全文向量索引并构建内存 BM25 倒排表。

        Args:
            chunks_path: 全文语料路径。
            show_progress: 是否显示索引加载进度。
            lexical_only: 仅用于离线诊断，禁止作为正式混合检索问答。

        Returns:
            None: 初始化可复用的检索器。

        Raises:
            RuntimeError: 索引缺失或与当前语料不一致。
        """
        raw = Path(chunks_path).read_bytes()
        self.chunks = json.loads(raw)
        self.corpus_sha256 = hashlib.sha256(raw).hexdigest()
        self.lexical_only = lexical_only
        self.vectors = None
        self.encoder = None
        if not self.chunks:
            raise RuntimeError("知识库为空。")
        if not lexical_only:
            if not (INDEX_DIR / "manifest.json").exists():
                raise RuntimeError("语义向量索引尚未建立，请执行 search_engine.py --build-index。")
            meta = json.loads((INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
            if meta["corpus_sha256"] != self.corpus_sha256 or meta["revision"] != MODEL_REVISION:
                raise RuntimeError("向量索引与全文语料或模型不一致，需要重新建库。")
            self.vectors = np.load(INDEX_DIR / "vectors.npy", mmap_mode="r", allow_pickle=False)
            if len(self.vectors) != len(self.chunks):
                raise RuntimeError("向量数与切块数不一致。")
            self.encoder = LocalEmbedding(show_progress=show_progress)
        self.postings = defaultdict(list)
        self.lengths = np.zeros(len(self.chunks), dtype=np.float32)
        self.companies = sorted({c["company"] for c in self.chunks})
        for i, chunk in enumerate(tqdm(self.chunks, desc="构建 BM25", disable=not show_progress)):
            counts = Counter(tokenize_chinese(chunk["content"]))
            self.lengths[i] = sum(counts.values())
            for token, count in counts.items():
                self.postings[token].append((i, count))
        self.average_length = max(float(self.lengths.mean()), 1)
        self._last_query = None
        self._last_scores = None

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K, company_filter: str | None = None) -> list[tuple[dict, float]]:
        """真实检索任意问题；公司过滤后在该公司内部独立融合两路排序。

        Args:
            query: 新问题文本，不匹配预置问答。
            top_k: 最多返回的切块数。
            company_filter: 指定公司时严格限制来源，避免全景题只召回头部公司。

        Returns:
            list[tuple[dict, float]]: 实际切块与 RRF 分数，并保留两路原始得分。
        """
        if query != self._last_query:
            lexical = np.zeros(len(self.chunks), dtype=np.float32)
            for token in set(tokenize_chinese(query)):
                entries = self.postings.get(token, [])
                if not entries:
                    continue
                idf = math.log(1 + (len(self.chunks) - len(entries) + .5) / (len(entries) + .5))
                for i, count in entries:
                    lexical[i] += idf * count * 2.5 / (count + 1.5 * (.25 + .75 * self.lengths[i] / self.average_length))
            semantic = np.zeros(len(self.chunks), dtype=np.float32)
            if self.encoder is not None:
                semantic = self.vectors @ self.encoder.encode([query], is_query=True)[0]
            self._last_query, self._last_scores = query, (lexical, semantic)
        lexical, semantic = self._last_scores
        eligible = np.array([i for i, c in enumerate(self.chunks) if company_filter is None or c["company"] == company_filter], dtype=int)
        if not len(eligible):
            return []
        fusion = np.zeros(len(self.chunks), dtype=np.float32)
        for scores in ([lexical] if self.lexical_only else [lexical, semantic]):
            order = eligible[np.argsort(-scores[eligible], kind="stable")[:max(60, top_k * 8)]]
            for rank, i in enumerate(order, 1):
                if scores[i] > 0:
                    fusion[i] += 1 / (60 + rank)
        selected = eligible[np.argsort(-fusion[eligible], kind="stable")]
        results = []
        for i in selected:
            if fusion[i] <= 0 or len(results) >= top_k:
                break
            chunk = {**self.chunks[i], "bm25_score": float(lexical[i]), "vector_score": float(semantic[i])}
            results.append((chunk, float(fusion[i])))
        return results

    def answer(self, query: str, top_k: int = DEFAULT_TOP_K, show_progress: bool = True) -> dict:
        """逐公司取证，调用真实模型，再检查出处和数值原文锚点。

        Args:
            query: 用户新问题，1 至 1000 字。
            top_k: 每家公司取证配额；全景题对每家公司独立检索。
            show_progress: 是否显示生成等待提示。

        Returns:
            dict: 答案、事实、引用、召回块、公司覆盖情况及模型审计信息。

        Raises:
            ValueError: 问题为空或过长。
            RuntimeError: 仅 BM25 诊断模式或模型调用失败。
        """
        if not query.strip() or len(query) > 1000:
            raise ValueError("请输入 1 至 1000 字的问题。")
        if self.lexical_only:
            raise RuntimeError("BM25 诊断模式不生成正式答案。")
        requested = [company for company in self.companies if any(name in query for name in [company, *COMPANY_ALIASES.get(company, [])])]
        if re.search(r"全行业|全景|10\s*家|十家|全部公司|各公司|各家|所有公司|梯队|行业.*(?:排名|排序)", query):
            requested = self.companies[:]
        results = []
        if requested:
            for company in requested:
                results.extend(self.retrieve(query, top_k, company))
        else:
            results = self.retrieve(query, top_k)
        result = synthesize_answer(query, results, requested, show_progress=show_progress)
        result["corpus_sha256"] = self.corpus_sha256
        return result


def numeric_tokens(text: str, full_value: bool = False) -> list[tuple[Decimal, str]]:
    """识别财报数字及明确单位，统一全角格式但保留正负号和百分比维度。

    Args:
        text: 原文、答案正文或 claim.value；只用于数值比较，不改写原文 quote。
        full_value: True 时整个字符串必须是单个数字及可选单位，否则返回空列表。

    Returns:
        list[tuple[Decimal, str]]，精确数值与紧邻单位；无单位用空字符串。
        “10家公司”等覆盖计数不识别为带“家”单位的业务事实。
    """
    normalized = unicodedata.normalize("NFKC", str(text)).replace("−", "-").strip()
    match = NUMBER_PATTERN.fullmatch(normalized) if full_value else None
    matches = ([match] if match else []) if full_value else NUMBER_PATTERN.finditer(normalized)
    return [(Decimal(item["value"].replace(",", "")), item["unit"] or "") for item in matches]


def synthesize_answer(query: str, retrieved_chunks: list[tuple[dict, float]], requested_companies: list[str] | None = None, show_progress: bool = True) -> dict:
    """把本次真实证据交给模型，验证引用身份、原文和数值锚点。

    检查明确数字及元/万元/亿元/%/吨/家单位，金额只允许精确等值换算，不接受自行舍入。
    家公司/家企业等覆盖计数不作为业务数量核对。检查不能证明科目、期间、叙述归属
    或完整财务口径正确，答案仍需独立金标与人工复核；被拒草稿仅供审计，不是有效答案。

    Args:
        query: 用户原问题。
        retrieved_chunks: 实际召回块与得分，保留完整原文和物理页码。
        requested_companies: 本次需覆盖的公司列表。
        show_progress: 是否显示模型生成进度提示。

    Returns:
        dict: 可展示的答案及供独立评测使用的结构化结果。
    """
    from financial_rag.model_client import generate_answer

    evidence = [{**chunk, "score": round(score, 6)} for chunk, score in retrieved_chunks]
    requested = requested_companies or []
    base = {"query": query, "mode": "llm", "pipeline_version": PIPELINE_VERSION, "retrieved_chunks": evidence}
    if not evidence:
        return {**base, "answer": "未检索到支持此问题的证据，暂不作答。", "status": "insufficient_evidence", "claims": [], "citations": [], "coverage": {"requested": requested, "supported": [], "cited": [], "missing": requested}}
    result = generate_answer(query, evidence, requested, show_progress=show_progress)
    by_id = {c["chunk_id"]: c for c in evidence}
    issues = []
    cited_ids = []
    for cid in result["citations"]:
        if not isinstance(cid, str) or cid not in by_id:
            issues.append(f"引用编号未在本次召回中：{cid}。")
        elif cid not in cited_ids:
            cited_ids.append(cid)
    for cid in re.findall(r"\[([^\[\]]+)\]", unicodedata.normalize("NFKC", result["answer"])):
        if cid not in by_id or cid not in cited_ids:
            issues.append(f"正文引用未召回或未列入引用清单：{cid}。")
    valid_claims = []
    valid_quantities = []
    for claim in result["claims"]:
        chunk = by_id.get(claim["chunk_id"])
        quote = str(claim["quote"])
        label = f"{claim['company']}/{claim['metric']}"
        if chunk is None:
            issues.append(f"{label}：事实引用 {claim['chunk_id']} 未召回。")
            continue
        if claim["company"] != chunk["company"]:
            issues.append(f"{label}：公司与引用所属公司 {chunk['company']} 不一致。")
            continue
        if str(claim["page"]) != str(chunk["page"]):
            issues.append(f"{label}：PDF页码 {claim['page']} 与引用页码 {chunk['page']} 不一致。")
            continue
        if not quote or quote not in chunk["content"]:
            issues.append(f"{label}：quote 不是对应切块中的连续原文。")
            continue
        parsed_value = numeric_tokens(claim["value"], full_value=True)
        if not parsed_value:
            issues.append(f"{label}：value 不是可核对的完整数字：{claim['value']}。")
            continue
        value, value_unit = parsed_value[0]
        unit = unicodedata.normalize("NFKC", claim["unit"]).strip().removeprefix("人民币").strip()
        declared = re.findall(r"单位\s*[:：]\s*(?:人民币)?\s*(亿元|万元|元|%|吨|家)(?![/／])", unicodedata.normalize("NFKC", chunk["content"]))
        declared_unit = declared[0] if len(set(declared)) == 1 else ""
        matches = []
        for line in quote.splitlines():
            # 同行“比例（%） 0.42”优先于整表“单位：元”；跨行或混合表头不猜测。
            row_units = re.findall(r"\(\s*(亿元|万元|元/股|元|%|吨|家)\s*\)", unicodedata.normalize("NFKC", line))
            row_unit = row_units[0] if len(set(row_units)) == 1 else ("" if row_units else declared_unit)
            matches.extend((number, explicit_unit or row_unit) for number, explicit_unit in numeric_tokens(line) if number == value)
        if value_unit and value_unit != unit:
            issues.append(f"{label}：value 中的单位 {value_unit} 与 claim.unit={unit} 冲突。")
            continue
        if not matches:
            issues.append(f"{label}：原文未出现保留正负号的数值 {claim['value']}。")
            continue
        if unit == "%" and not any(source_unit == "%" for _, source_unit in matches):
            issues.append(f"{label}：原文未证明该数值为百分比，不能将小数或普通数字改成%。")
            continue
        if unit != "%" and all(source_unit == "%" for _, source_unit in matches):
            issues.append(f"{label}：原文数值为百分比，与 claim.unit={unit} 冲突。")
            continue
        if not any(not source_unit or source_unit == unit for _, source_unit in matches):
            issues.append(f"{label}：原文明确单位 {sorted({source_unit for _, source_unit in matches})} 与 claim.unit={unit} 冲突。")
            continue
        valid_claims.append(claim)
        valid_quantities.append((value, unit))
        if claim["chunk_id"] not in cited_ids:
            cited_ids.append(claim["chunk_id"])
    # 正文仅做明确数字单位的等值检查，不能据此宣称语义或业务口径已全面验证。
    for value, unit in numeric_tokens(result["answer"]):
        if not unit:
            continue
        if not any(
            (unit == claim_unit and value == claim_value)
            or (unit in CURRENCY_FACTORS and claim_unit in CURRENCY_FACTORS
                and value * CURRENCY_FACTORS[unit] == claim_value * CURRENCY_FACTORS[claim_unit])
            for claim_value, claim_unit in valid_quantities
        ):
            issues.append(f"正文数值 {value}{unit} 无对应的已核对事实；仅允许精确等值单位换算，不接受自行计算或舍入。")
    citations = [by_id[cid] for cid in cited_ids]
    cited = sorted({c["company"] for c in citations})
    supported = sorted({claim["company"] for claim in valid_claims}) if result["claims"] else cited[:]
    missing = [name for name in requested if name not in supported]
    status = result["status"]
    answer = result["answer"]
    if issues:
        answer = "模型生成结果未通过出处或数值原文校验，暂不提供结论。请查看召回证据。"
        status = "insufficient_evidence"
        valid_claims = []
        citations = []
        supported = []
        cited = []
        missing = requested[:]
    elif status == "insufficient_evidence":
        supported = []
        missing = requested[:]
    elif missing or (status == "answered" and not citations):
        status = "partial"
    draft = {"validated": False, "purpose": "仅供审计被拒原因，不是有效答案", "answer": result["answer"], "status": result["status"], "claims": result["claims"], "citations": result["citations"], "ranking": result.get("ranking", [])} if issues else None
    return {**base, "answer": answer, "status": status, "claims": valid_claims, "citations": citations, "ranking": [] if issues else result.get("ranking", []), "coverage": {"requested": requested, "supported": supported, "cited": cited, "missing": missing}, "validation_issues": issues, "unvalidated_draft": draft, "model_run": result.get("_model_run", {})}


def answer_questions(questions: list[str], chunks_path: str | Path | None = None, top_k: int = DEFAULT_TOP_K, show_progress: bool = True) -> list[dict]:
    """使用与网页完全相同的检索和生成链回答一组新问题。

    Args:
        questions: 按顺序实际调用模型的问题列表。
        chunks_path: 全文语料路径，为 None 时使用默认值。
        top_k: 每家公司取证配额。
        show_progress: 是否显示索引和逐题进度。

    Returns:
        list[dict]: 按输入顺序返回真实回答，不预填答案或评分。
    """
    path = Path(chunks_path or CHUNKS_PATH).resolve()
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _BENCHMARK_CACHE:
        _BENCHMARK_CACHE.clear()
        _BENCHMARK_CACHE[key] = FinancialHybridRetriever(path, show_progress=show_progress)
    retriever = _BENCHMARK_CACHE[key]
    return [retriever.answer(query, top_k, show_progress) for query in tqdm(questions, desc="逐题问答", disable=not show_progress)]


def main() -> int:
    """直接执行全文向量建库或新问题问答，支持脚本和模块入口。

    Args:
        无。读取顶部参数及命令行选项。

    Returns:
        int: 完成返回 0，配置或运行失败返回 1。
    """
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--query", default=DEFAULT_QUESTION)
    parser.add_argument("--lexical-only", action="store_true", help="只诊断BM25召回，不生成答案")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    try:
        if args.build_index:
            result = build_vector_index(show_progress=SHOW_PROGRESS and not args.no_progress)
        elif args.lexical_only:
            retriever = FinancialHybridRetriever(show_progress=SHOW_PROGRESS and not args.no_progress, lexical_only=True)
            result = {"mode": "bm25_diagnostic_only", "retrieved_chunks": [chunk for chunk, _ in retriever.retrieve(args.query)]}
        else:
            result = answer_questions([args.query], show_progress=SHOW_PROGRESS and not args.no_progress)[0]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[未完成] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
