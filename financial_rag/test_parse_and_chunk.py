"""检查目录条目与真实同页多节的区分，不改动财报或正式语料。"""

import unittest

if __package__:
    from .parse_and_chunk import detect_page_chapter
else:
    from parse_and_chunk import detect_page_chapter


class ChapterDetectionTests(unittest.TestCase):
    def test_directory_entries_do_not_change_current_chapter(self):
        page = "目录\n第一节 释义 ........ 4\n第二节 公司简介 4\n第九节 债券相关情况 …… 55\n第十节 财务报告 55"
        self.assertEqual(detect_page_chapter(page, "前言/重要提示"), ("前言/重要提示", "前言/重要提示"))

    def test_real_same_page_sections_preserve_both_and_carry_last(self):
        page = "第九节 债券相关情况\n不适用\n第十节 财务报告\n一、审计报告"
        current, carry = detect_page_chapter(page, "第八节 优先股相关情况")
        self.assertEqual(current, "第九节 债券相关情况 / 第十节 财务报告")
        self.assertEqual(carry, "第十节 财务报告")
        self.assertEqual(detect_page_chapter("合并资产负债表\n单位：元", carry), (carry, carry))


def main():
    """运行两个独立的章节识别回归，不重解析财报。

    Args:
        无；测试夹具写在测试类中。

    Returns:
        None，unittest 根据断言结果设置进程退出码。
    """
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
