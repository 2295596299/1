"""评分规则的离线回归：测试夹具不写入正式语料、金标或评测记录。"""

from copy import deepcopy
import unittest

if __package__:
    from .run_benchmark import score_case, validate_saved_responses
else:
    from run_benchmark import score_case, validate_saved_responses


class BenchmarkScoringTests(unittest.TestCase):
    def setUp(self):
        self.fact = {"company": "C", "metric": "营业收入", "value": "10", "unit": "元",
                     "period": "2023", "scope": "合并", "page": 1,
                     "evidence_anchor": "营业收入", "quote": "营业收入 10"}
        self.chunk = {"chunk_id": "real", "company": "C", "page": 1, "content": "营业收入 10"}
        self.case = {"id": "TEST", "question": "测试年度营业收入", "kind": "numeric",
                     "gold_status": "verified_pdf", "target_companies": ["C"], "facts": [self.fact]}
        self.claim = {**self.fact, "chunk_id": "real"}
        self.result = {"answer": "营业收入10元。", "claims": [self.claim], "citations": ["real"],
                       "retrieved_chunks": [self.chunk], "status": "answered"}

    def test_exact_fact_passes(self):
        scored = score_case(self.case, self.result, {"real": self.chunk})
        self.assertEqual(scored["answer_evaluation"]["status"], "numeric_verified")

    def test_flow_period_rejects_year_end(self):
        self.result["claims"][0]["period"] = "2023年末"
        scored = score_case(self.case, self.result, {"real": self.chunk})
        self.assertEqual(scored["answer_evaluation"]["verified_facts"], 0)

    def test_stock_period_rejects_whole_year(self):
        self.case["facts"][0]["period"] = "2023-12-31"
        self.result["claims"][0]["period"] = "2023"
        scored = score_case(self.case, self.result, {"real": self.chunk})
        self.assertEqual(scored["answer_evaluation"]["verified_facts"], 0)

    def test_quote_cannot_change_sign_or_percent(self):
        for quote in ("营业收入 -10", "营业收入 10%"):
            with self.subTest(quote=quote):
                result = deepcopy(self.result)
                result["claims"][0]["quote"] = quote
                scored = score_case(self.case, result, {"real": self.chunk})
                self.assertEqual(scored["answer_evaluation"]["verified_facts"], 0)

    def test_extra_fake_citation_prevents_full_answer_pass(self):
        self.result["citations"].append("FAKE")
        scored = score_case(self.case, self.result, {"real": self.chunk})
        self.assertEqual(scored["answer_evaluation"]["verified_facts"], 1)
        self.assertEqual(scored["answer_evaluation"]["status"], "partial")
        self.assertEqual(scored["answer_evaluation"]["invalid_citations"], ["FAKE"])

    def test_wrong_citation_metadata_prevents_full_answer_pass(self):
        self.result["citations"] = [{"chunk_id": "real", "company": "WRONG", "page": 1}]
        scored = score_case(self.case, self.result, {"real": self.chunk})
        self.assertEqual(scored["answer_evaluation"]["status"], "partial")

    def test_wrong_ranking_is_separate_from_numeric_facts(self):
        self.case["expected_rank"] = ["C", "D"]
        self.result["ranking"] = ["D", "C"]
        scored = score_case(self.case, self.result, {"real": self.chunk})
        self.assertEqual(scored["answer_evaluation"]["verified_facts"], 1)
        self.assertFalse(scored["answer_evaluation"]["ranking"]["verified"])
        self.assertEqual(scored["answer_evaluation"]["status"], "partial")

    def test_keywords_do_not_count_as_answer(self):
        self.result["answer"] = "检索到了营业收入。"
        self.result["claims"] = []
        scored = score_case(self.case, self.result, {"real": self.chunk})
        self.assertEqual(scored["recall"]["matched"], 1)
        self.assertEqual(scored["answer_evaluation"]["verified_facts"], 0)

    def test_wrong_unit_and_scope_fail(self):
        for field, value in (("unit", "万元"), ("scope", "母公司")):
            with self.subTest(field=field):
                result = deepcopy(self.result)
                result["claims"][0][field] = value
                scored = score_case(self.case, result, {"real": self.chunk})
                self.assertEqual(scored["answer_evaluation"]["verified_facts"], 0)

    def test_resume_rejects_old_or_mixed_versions(self):
        old = {"question": "q1", "corpus_sha256": "same_corpus"}
        current = {"question": "q2", "corpus_sha256": "same_corpus", "pipeline_version": "v2"}
        for responses in ([old], [old, current]):
            with self.subTest(responses=responses), self.assertRaises(ValueError):
                validate_saved_responses(responses, ["q1", "q2"], "same_corpus", "v2", for_resume=True)

    def test_regrade_preserves_historical_version(self):
        old = {"question": "q1", "corpus_sha256": "same_corpus"}
        self.assertEqual(validate_saved_responses([old], ["q1"], "same_corpus", "v2"), "v1")

    def test_resume_accepts_same_version_prefix(self):
        current = {"question": "q1", "corpus_sha256": "same_corpus", "pipeline_version": "v2"}
        self.assertEqual(validate_saved_responses([current], ["q1", "q2"], "same_corpus", "v2", for_resume=True), "v2")


if __name__ == "__main__":
    unittest.main()
