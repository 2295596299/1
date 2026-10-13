"""仅用 mock 模型返回检查答案校验边界，不加载索引、不执行模型推理。"""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from financial_rag.search_engine import synthesize_answer


class SynthesisValidationTests(unittest.TestCase):
    """校验真实生成链的输出检查函数，所有金额和公司引用均为独立测试夹具。"""

    def setUp(self):
        """建立互相一致的原文、事实和答案，供每个反例独立修改。

        Args:
            无；unittest 会在每个测试前调用。

        Returns:
            None，设置当前测试的 chunk、claim、payload 和 requested。
        """
        self.chunk = {
            "chunk_id": "600519_p87_tbl1", "company": "贵州茅台", "page": 87,
            "content": "2023年度 合并利润表 单位：元\n营业收入 1,000.00",
        }
        self.claim = {
            "company": self.chunk["company"], "page": self.chunk["page"],
            "chunk_id": self.chunk["chunk_id"], "metric": "营业收入",
            "value": "1,000.00", "unit": "元", "period": "2023年度",
            "scope": "合并", "quote": "营业收入 1,000.00",
        }
        self.payload = {
            "answer": "营业收入为1,000.00元 [600519_p87_tbl1]。", "status": "answered",
            "citations": [self.chunk["chunk_id"]], "claims": [self.claim], "ranking": [],
        }
        self.requested = [self.chunk["company"]]

    def run_case(self, query="贵州茅台2023年度合并营业收入是多少？", chunks=None):
        """使用受控模型返回执行校验函数，并确认没有绕过 mock 调用真实生成。

        Args:
            query: 当前测试问题，默认与营业收入夹具对应。
            chunks: 可选召回块列表；None 使用当前单个 chunk，空列表代表未召回。

        Returns:
            dict，synthesize_answer 的真实校验结果。
        """
        evidence = [self.chunk] if chunks is None else chunks
        with patch("financial_rag.model_client.generate_answer", return_value=deepcopy(self.payload)) as model:
            result = synthesize_answer(query, [(chunk, 1.0) for chunk in evidence], self.requested, show_progress=False)
            self.assertEqual(model.call_count, int(bool(evidence)))
        return result

    def assert_rejected(self, result):
        """确认错误事实被拒绝且不留下可展示的已验证事实或引用。

        Args:
            result: synthesize_answer 输出字典。

        Returns:
            None；拒绝状态不符合要求时产生 unittest 断言失败。
        """
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["claims"], [])
        self.assertEqual(result["citations"], [])

    def test_exact_comma_amount_and_quote_pass(self):
        result = self.run_case()
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["claims"][0]["value"], "1,000.00")

    def test_percent_number_with_separate_unit_passes(self):
        self.chunk["content"] = self.claim["quote"] = "营业收入增长率 12.34%"
        self.claim.update(metric="营业收入增长率", value="12.34", unit="%")
        self.payload["answer"] = "营业收入增长率12.34% [600519_p87_tbl1]。"
        self.assertEqual(self.run_case()["status"], "answered")

    def test_ascii_negative_sign_is_preserved(self):
        self.chunk["content"] = self.claim["quote"] = "财务费用 -1,000.00"
        self.claim.update(metric="财务费用", value="-1,000.00")
        self.payload["answer"] = "财务费用-1,000.00元 [600519_p87_tbl1]。"
        self.assertEqual(self.run_case()["status"], "answered")
        self.claim["value"] = "1,000.00"
        self.assert_rejected(self.run_case())

    def test_wrong_company_page_or_quote_is_rejected(self):
        original = deepcopy(self.claim)
        for field, wrong in (("company", "五粮液"), ("page", 88), ("quote", "营业收入 9,999.00")):
            with self.subTest(field=field):
                self.claim.update(original)
                self.claim[field] = wrong
                self.assert_rejected(self.run_case())

    def test_unretrieved_declared_and_inline_citations_are_rejected(self):
        self.payload["citations"].append("600519_p999_tbl1")
        self.assert_rejected(self.run_case())
        self.payload["citations"] = [self.chunk["chunk_id"]]
        for reference in ("[600519_p999_tbl1]", "[FAKE]", "［FAKE］"):
            with self.subTest(reference=reference):
                self.payload["answer"] = "营业收入1,000.00元 [600519_p87_tbl1]，另见" + reference
                self.assert_rejected(self.run_case())

    def test_missing_requested_company_is_partial(self):
        self.requested.append("五粮液")
        result = self.run_case()
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["coverage"]["missing"], ["五粮液"])

    def test_empty_recall_does_not_call_model(self):
        self.assert_rejected(self.run_case(chunks=[]))

    def test_raw_percent_value_is_accepted_without_changing_prompt_contract(self):
        self.chunk["content"] = self.claim["quote"] = "营业收入增长率 12.34%"
        self.claim.update(metric="营业收入增长率", value="12.34%", unit="%")
        self.payload["answer"] = "营业收入增长率12.34% [600519_p87_tbl1]。"
        self.assertEqual(self.run_case()["status"], "answered")

    def test_raw_unicode_minus_value_is_accepted(self):
        self.chunk["content"] = self.claim["quote"] = "财务费用 −1,000.00"
        self.claim.update(metric="财务费用", value="−1,000.00")
        self.payload["answer"] = "财务费用−1,000.00元 [600519_p87_tbl1]。"
        self.assertEqual(self.run_case()["status"], "answered")

    def test_unicode_minus_cannot_be_silently_dropped(self):
        self.chunk["content"] = self.claim["quote"] = "财务费用 −1,000.00"
        self.claim.update(metric="财务费用", value="1,000.00")
        self.payload["answer"] = "财务费用1,000.00元 [600519_p87_tbl1]。"
        self.assert_rejected(self.run_case())

    def test_explicit_currency_unit_contradiction_is_rejected(self):
        self.claim["quote"] = self.chunk["content"]
        self.claim["unit"] = "万元"
        self.payload["answer"] = "营业收入1,000.00万元 [600519_p87_tbl1]。"
        self.assert_rejected(self.run_case())

    def test_answer_amount_cannot_contradict_validated_claim(self):
        self.payload["answer"] = "营业收入9,999.00元 [600519_p87_tbl1]。"
        self.assert_rejected(self.run_case())

    def test_citing_cover_text_is_not_numeric_company_coverage(self):
        other = {"chunk_id": "000858_p1_txt1", "company": "五粮液", "page": 1, "content": "五粮液2023年度报告"}
        self.requested.append(other["company"])
        self.payload["citations"].append(other["chunk_id"])
        result = self.run_case("贵州茅台和五粮液2023年度营业收入分别是多少？", [self.chunk, other])
        self.assertEqual(result["status"], "partial")
        self.assertIn("五粮液", result["coverage"]["missing"])
        self.assertIn("五粮液", result["coverage"]["cited"])

    def test_full_width_sign_comma_and_percent_are_preserved(self):
        for raw, unit in (("－１，０００．００", "元"), ("１２．３４％", "％")):
            with self.subTest(value=raw):
                self.chunk["content"] = self.claim["quote"] = "测试指标 " + raw
                self.claim.update(metric="测试指标", value=raw, unit=unit)
                self.payload["answer"] = "测试指标" + raw + ("元" if unit == "元" else "") + " [600519_p87_tbl1]。"
                self.assertEqual(self.run_case()["status"], "answered")

    def test_percent_row_label_overrides_table_currency_header(self):
        self.chunk["content"] = "单位：元\n研发投入比例（%） | 0.42"
        self.claim.update(metric="研发投入比例", quote="研发投入比例（%） | 0.42", value="0.42", unit="%")
        self.payload["answer"] = "研发投入比例0.42% [600519_p87_tbl1]。"
        self.assertEqual(self.run_case()["status"], "answered")

    def test_plain_fraction_or_different_percent_is_not_equivalent(self):
        for quote, value in (("增长率 0.12", "0.12%"), ("增长率 12%", "0.12%")):
            with self.subTest(quote=quote):
                self.chunk["content"] = self.claim["quote"] = quote
                self.claim.update(metric="增长率", value=value, unit="%")
                self.payload["answer"] = "增长率0.12% [600519_p87_tbl1]。"
                self.assert_rejected(self.run_case())

    def test_exact_currency_conversion_is_accepted_but_rounding_is_not(self):
        self.payload["answer"] = "营业收入0.1万元 [600519_p87_tbl1]。"
        self.assertEqual(self.run_case()["status"], "answered")
        self.chunk["content"] = self.claim["quote"] = "营业收入 1,000.01"
        self.claim["value"] = "1,000.01"
        self.assert_rejected(self.run_case())

    def test_explicit_unit_after_value_cannot_be_changed(self):
        self.chunk["content"] = self.claim["quote"] = "营业收入 1,000.00万元"
        self.assert_rejected(self.run_case())

    def test_company_coverage_counts_are_not_business_quantity_claims(self):
        self.payload["answer"] = "这10家公司、9家 企业的材料中，本次收入为1,000.00元 [600519_p87_tbl1]。"
        self.assertEqual(self.run_case()["status"], "answered")
        self.payload["answer"] += "新增经销商10家。"
        self.assert_rejected(self.run_case())

    def test_refusal_keeps_cited_companies_without_supported_coverage(self):
        self.payload.update(answer="无法从资料确认该指标 [600519_p87_tbl1]。", status="insufficient_evidence", claims=[])
        result = self.run_case()
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["coverage"]["supported"], [])
        self.assertEqual(result["coverage"]["missing"], self.requested)
        self.assertEqual(result["coverage"]["cited"], self.requested)

    def test_rejected_draft_is_separate_and_keeps_model_audit(self):
        self.payload["answer"] = "营业收入9,999.00元 [600519_p87_tbl1]。"
        self.payload["_model_run"] = {"run_id": "mock-only"}
        result = self.run_case()
        self.assertEqual(result["pipeline_version"], "v2")
        self.assert_rejected(result)
        self.assertFalse(result["unvalidated_draft"]["validated"])
        self.assertEqual(result["unvalidated_draft"]["answer"], self.payload["answer"])
        self.assertEqual(result["model_run"]["run_id"], "mock-only")
        self.assertIn("9999.00", " ".join(result["validation_issues"]))


def main():
    """从 IDE、脚本或模块运行离线断言，并清楚显示通过项与待修项。

    Args:
        无；unittest 接收命令行提供的测试筛选参数。

    Returns:
        None，测试框架以运行结果决定退出码，不生成正式评测文件。
    """
    unittest.main(verbosity=2)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    main()
