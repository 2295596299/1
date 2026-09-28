"""
向量 + BM25 混合检索与证据抽取引擎 (Search Engine)

该模块对 2,971 个带有元数据的财报切块建立：
1. BM25 词频精确检索索引（擅长精准匹配指标名、数字、特定科目）；
2. 向量/TF-IDF 语义检索索引（擅长匹配语义意图、业务描述）；
3. 动态融合重排机制（Hybrid Fusion），返回相关度最高的证据切块并输出带出处的答案。
"""

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 控制台编码安全配置
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
CHUNKS_PATH: str = str(CURRENT_DIR / "chunks.json")
BM25_K1: float = 1.5
BM25_B: float = 0.75
DEFAULT_TOP_K: int = 4
ALPHA_WEIGHT: float = 0.5  # 向量检索权重 (1-ALPHA 为 BM25 权重)

STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "多少", "什么", "请问", "哪些", "情况", "分析", "对比"
}


def tokenize_chinese(text: str) -> List[str]:
    """
    轻量且高效的中文分词器（结合字符、双字词与英文/数字单元）。

    不依赖额外第三方分词包，对财经专业术语（如“研发费用”、“营业收入”、“基酒”、“毛利率”）
    具备极高召回率。

    Args:
        text: 待切分的中文文本。

    Returns:
        切分后的词元列表。
    """
    clean_text = re.sub(r"[^\w\u4e00-\u9fff.]+", " ", text.lower())
    tokens: List[str] = []

    # 提取英文与连续数字（如 600519, 2023, 15.2%）
    tokens.extend(re.findall(r"[a-z0-9.]+", clean_text))

    # 提取中文单字与双字滑动窗口词元
    cn_chars = re.findall(r"[\u4e00-\u9fff]", clean_text)
    for i in range(len(cn_chars)):
        char = cn_chars[i]
        if char not in STOP_WORDS:
            tokens.append(char)
        if i < len(cn_chars) - 1:
            bi_gram = cn_chars[i] + cn_chars[i + 1]
            if bi_gram not in STOP_WORDS:
                tokens.append(bi_gram)

    return tokens


class BM25Index:
    """Okapi BM25 词频倒排检索器。"""

    def __init__(self, corpus: List[str], k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = []
        self.doc_freqs: Dict[str, int] = Counter()
        self.tokenized_corpus = []

        total_len = 0
        for doc in corpus:
            tokens = tokenize_chinese(doc)
            self.tokenized_corpus.append(tokens)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len
            for word in set(tokens):
                self.doc_freqs[word] += 1

        self.avg_dl = (total_len / self.corpus_size) if self.corpus_size > 0 else 1.0
        self.idf: Dict[str, float] = {}
        for word, freq in self.doc_freqs.items():
            # 标准 BM25 平滑 IDF 公式
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: str) -> np.ndarray:
        """计算查询语句与知识库中各文档的 BM25 分数。"""
        query_tokens = tokenize_chinese(query)
        scores = np.zeros(self.corpus_size, dtype=np.float32)

        for token in query_tokens:
            if token not in self.idf:
                continue
            token_idf = self.idf[token]
            for doc_idx, doc_tokens in enumerate(self.tokenized_corpus):
                # 统计词频
                f = doc_tokens.count(token)
                if f > 0:
                    denom = f + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[doc_idx] / self.avg_dl))
                    scores[doc_idx] += token_idf * ((f * (self.k1 + 1.0)) / denom)

        return scores


class FinancialHybridRetriever:
    """向量 + BM25 混合检索器。"""

    def __init__(self, chunks_path: str = CHUNKS_PATH):
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"未找到切块文件: {chunks_path}，请先运行 parse_and_chunk.py")

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks: List[Dict[str, Any]] = json.load(f)

        self.corpus = [c["content"] for c in self.chunks]
        print(f"[检索引擎] 正在为 {len(self.chunks)} 个切块构建 BM25 索引与向量空间...")

        # 1. 构建 BM25 索引
        self.bm25 = BM25Index(self.corpus)

        # 2. 构建 TF-IDF 密集向量空间
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize_chinese,
            token_pattern=None,
            sublinear_tf=True,
            norm="l2"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        print("[检索引擎] [OK] 混合双路索引构建完毕。")

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        company_filter: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        根据查询语句检索最相关的 top_k 个财报切块。

        综合 BM25 精确命中分数与向量余弦相似度分数，同时支持对特定公司进行定向提权。

        Args:
            query: 用户提问文本。
            top_k: 返回切块数量。
            company_filter: 可选的公司名称过滤。

        Returns:
            List of (chunk_dict, hybrid_score)。
        """
        # 1. 计算 BM25 得分并归一化
        bm25_raw = self.bm25.get_scores(query)
        bm25_max = np.max(bm25_raw)
        bm25_norm = (bm25_raw / bm25_max) if bm25_max > 0 else bm25_raw

        # 2. 计算向量余弦相似度并归一化
        q_vec = self.vectorizer.transform([query])
        vec_sim = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        vec_max = np.max(vec_sim)
        vec_norm = (vec_sim / vec_max) if vec_max > 0 else vec_sim

        # 3. 混合打分
        hybrid_scores = (1.0 - ALPHA_WEIGHT) * bm25_norm + ALPHA_WEIGHT * vec_norm

        # 4. 公司名称匹配提权（如果问题明确提及某公司，提升其相关切块权重）
        for idx, chunk in enumerate(self.chunks):
            comp = chunk.get("company", "")
            if comp and comp in query:
                hybrid_scores[idx] += 0.25

        # 5. 过滤与排序
        if company_filter:
            for idx, chunk in enumerate(self.chunks):
                if chunk.get("company") != company_filter:
                    hybrid_scores[idx] = -1.0

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(hybrid_scores[idx])
            if score > 0.05:
                results.append((self.chunks[idx], score))

        return results


def synthesize_answer(
    query: str,
    retrieved_chunks: List[Tuple[Dict[str, Any], float]]
) -> Dict[str, Any]:
    """
    根据检索到的证据切块生成结构化解答，并标注明确的财报出处（公司、章节、页码）。

    Args:
        query: 用户提问。
        retrieved_chunks: 检索召回的证据块与得分列表。

    Returns:
        包含 'answer'（综合解答）、'citations'（出处列表）和 'evidence'（证据文本）的字典。
    """
    if not retrieved_chunks:
        return {
            "answer": "未在 10 家上市公司的 2023 年报知识库中检索到高相关性的依据段落。",
            "citations": [],
            "evidence": []
        }

    citations = []
    evidence_blocks = []
    for chunk, score in retrieved_chunks:
        cite_str = f"【{chunk['company']}】《{chunk['year']}年报》{chunk['chapter']} 第{chunk['page']}页 ({chunk['type']})"
        citations.append({
            "company": chunk["company"],
            "year": chunk["year"],
            "chapter": chunk["chapter"],
            "page": chunk["page"],
            "type": chunk["type"],
            "chunk_id": chunk["chunk_id"],
            "score": round(score, 3),
            "label": cite_str
        })
        evidence_blocks.append(f"{cite_str}\n{chunk['content']}")

    # 提炼主要依据与核心结论
    primary_chunk = retrieved_chunks[0][0]
    first_cite = citations[0]["label"]

    answer_text = (
        f"根据知识库精准检索，相关事实依据出处如下：\n"
        f"• 核心出处：{first_cite}\n\n"
        f"【证据原文与表格摘要】：\n"
    )

    for i, (chunk, sc) in enumerate(retrieved_chunks[:3], 1):
        # 截取关键正文或表格展示
        snippet = chunk["content"]
        if len(snippet) > 350:
            snippet = snippet[:350] + "..."
        answer_text += f"\n[证据 {i}] ({chunk['company']} 第{chunk['page']}页 | 匹配度: {sc:.2f}):\n{snippet}\n"

    return {
        "query": query,
        "answer": answer_text,
        "citations": citations,
        "evidence": evidence_blocks,
    }


def main() -> None:
    """测试检索与回答流程。"""
    retriever = FinancialHybridRetriever()
    test_q = "贵州茅台2023年的研发费用是多少？主要用于哪些项目？"
    print(f"\n[测试提问]: {test_q}")
    results = retriever.retrieve(test_q, top_k=3)
    res = synthesize_answer(test_q, results)
    print("\n--- 生成答案 ---")
    print(res["answer"])


if __name__ == "__main__":
    main()
