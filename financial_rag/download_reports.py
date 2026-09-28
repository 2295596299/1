"""
上市公司财报批量下载模块 (Download Reports)

该模块从巨潮资讯网 (cninfo.com.cn) 批量检索并下载
A 股白酒行业 10 家代表性龙头公司的 2023 年度报告 (PDF 格式)。
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import requests

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
DEFAULT_TARGET_YEAR: str = "2023"
TARGET_DIR: str = str(Path(__file__).resolve().parent / "pdf_reports")
CNINFO_DOWNLOAD_BASE: str = "http://static.cninfo.com.cn/"

# 经巨潮资讯网公告核验的 10 家白酒龙头 2023 年报直接路径
REPORTS_METADATA: List[Dict[str, str]] = [
    {
        "code": "600519",
        "name": "贵州茅台",
        "year": "2023",
        "title": "贵州茅台2023年年度报告",
        "adjunct_url": "finalpage/2024-04-03/1219506510.PDF"
    },
    {
        "code": "000858",
        "name": "五粮液",
        "year": "2023",
        "title": "宜宾五粮液2023年年度报告",
        "adjunct_url": "finalpage/2024-04-29/1219873658.PDF"
    },
    {
        "code": "000568",
        "name": "泸州老窖",
        "year": "2023",
        "title": "泸州老窖2023年年度报告",
        "adjunct_url": "finalpage/2024-04-27/1219872991.PDF"
    },
    {
        "code": "600809",
        "name": "山西汾酒",
        "year": "2023",
        "title": "山西汾酒2023年年度报告",
        "adjunct_url": "finalpage/2024-04-26/1219833278.PDF"
    },
    {
        "code": "002304",
        "name": "洋河股份",
        "year": "2023",
        "title": "洋河股份2023年年度报告",
        "adjunct_url": "finalpage/2024-04-27/1219873234.PDF"
    },
    {
        "code": "000596",
        "name": "古井贡酒",
        "year": "2023",
        "title": "古井贡酒2023年年度报告",
        "adjunct_url": "finalpage/2024-04-27/1219873167.PDF"
    },
    {
        "code": "600702",
        "name": "舍得酒业",
        "year": "2023",
        "title": "舍得酒业2023年年度报告",
        "adjunct_url": "finalpage/2024-03-20/1219347680.PDF"
    },
    {
        "code": "603369",
        "name": "今世缘",
        "year": "2023",
        "title": "今世缘2023年年度报告",
        "adjunct_url": "finalpage/2024-04-30/1219917783.PDF"
    },
    {
        "code": "603198",
        "name": "迎驾贡酒",
        "year": "2023",
        "title": "迎驾贡酒2023年年度报告",
        "adjunct_url": "finalpage/2024-04-26/1219834786.PDF"
    },
    {
        "code": "603589",
        "name": "口子窖",
        "year": "2023",
        "title": "口子窖2023年年度报告",
        "adjunct_url": "finalpage/2024-04-30/1219912125.PDF"
    },
]


def download_single_report(
    item: Dict[str, str],
    target_dir: str = TARGET_DIR,
    show_progress: bool = True
) -> str:
    """
    下载单份上市公司年度报告并存入指定目录。

    Args:
        item: 包含股票代码、简称、年份及附件 URL 的字典。
        target_dir: 目标保存目录绝对路径。
        show_progress: 是否打印下载进度与文件大小。

    Returns:
        保存后的 PDF 文件绝对路径。
    """
    os.makedirs(target_dir, exist_ok=True)
    file_name = f"{item['code']}_{item['name']}_{item['year']}年报.pdf"
    save_path = os.path.join(target_dir, file_name)

    if os.path.exists(save_path) and os.path.getsize(save_path) > 1024 * 100:
        if show_progress:
            print(f"[下载模块] 文件已存在，跳过下载: {file_name}")
        return save_path

    full_url = CNINFO_DOWNLOAD_BASE + item["adjunct_url"]
    if show_progress:
        print(f"[下载模块] 正在下载 {item['name']} ({item['code']}) 年报 ...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "http://www.cninfo.com.cn/"
    }

    resp = requests.get(full_url, headers=headers, stream=True, timeout=40)
    if resp.status_code != 200:
        raise RuntimeError(f"下载失败，HTTP状态码: {resp.status_code}, URL: {full_url}")

    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)

    if show_progress:
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"[下载模块] [OK] 下载成功: {file_name} ({size_mb:.2f} MB)")

    return save_path


def batch_download_reports(
    reports_meta: List[Dict[str, str]] = REPORTS_METADATA,
    target_dir: str = TARGET_DIR,
    show_progress: bool = True
) -> List[str]:
    """
    批量下载所有 10 家白酒公司的年报。

    Args:
        reports_meta: 待下载财报元数据列表。
        target_dir: 本地存储路径。
        show_progress: 是否打印阶段提示。

    Returns:
        已成功下载的本地 PDF 文件路径列表。
    """
    if show_progress:
        print(f"[下载模块] 开始批量下载 {len(reports_meta)} 家白酒龙头公司的 2023 年报 PDF...")

    downloaded = []
    for item in reports_meta:
        try:
            path = download_single_report(item, target_dir=target_dir, show_progress=show_progress)
            downloaded.append(path)
            time.sleep(0.3)
        except Exception as exc:
            print(f"[下载模块] [FAIL] 下载 {item['name']} 失败: {exc}")

    if show_progress:
        print(f"[下载模块] [DONE] 全部下载完成，共计 {len(downloaded)} 份财报。")
    return downloaded


def main() -> None:
    """直接运行该脚本下载 10 家白酒公司年报。"""
    batch_download_reports()


if __name__ == "__main__":
    main()
