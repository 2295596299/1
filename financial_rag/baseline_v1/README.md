# First real run：首次真实评测基线

本目录保留 2026-10-13 首轮完整的 14 题真实模型运行结果，不以之后的修复结果覆盖，也不逐题挑选最高分答案。

- 语料：10 份 2023 年报全文，1,857 页，6,696 个切块；BGE 向量与 BM25 混合检索后调用真实 Codex 模型。
- 运行：IDE/debugpy 直接运行 `run_benchmark.py`，14 题按顺序完成，约 6 分 08 秒；之后仅使用留存响应重评分，没有重新调用模型。
- 留档：`evaluation_responses.json` 是实际回答；`evaluation_records.json/md` 和 `evaluation_summary.json` 是自动评分；`manual_review.json` 是独立 PDF 逐条审阅。
- 可复核依据：同步保留 `gold_cases.json`、`parse_manifest.json` 及评分代码快照。各文件 SHA256、响应语料 SHA 和模型运行时间见 `archive_manifest.json`。

自动评分的 26/43 证据召回和 0/39 严格结构化通过，不能当作整题正确率。口径自然语言、指标别名和亿元舍入会造成严格匹配漏判，独立审阅另外说明有源支持、部分完成、拒答及失败。`manual_review.json` 仅对应本轮响应 SHA，不适用于未来版本。

修复通用检索、格式或引用校验缺陷后的第二轮必须重新执行全部 14 题，独立保存并重新审阅；不得把两轮中较好的单题拼成一次成绩。
