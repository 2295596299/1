# 个人主页与白酒财报知识库

仓库包含个人展示页 `index.html`，以及第三周作业的财报检索问答页 `rag.html`。代码仓库：[GitHub](https://github.com/2295596299/1)。

当前问答方案在本机运行：新问题经全文检索后，把本次证据交给已登录的 Codex CLI 生成回答。页面没有预置答案或按问题匹配答案的题库；示例按钮与自由提问使用同一流程。此改动不代表公开 Vercel 网站已经更新，也不代表静态站点已经具备本机问答能力。

## 优先使用 IDE Run / Debug

在 IDE 中将 `D:/homepage-1` 打开为项目，工作目录也设为该目录。直接运行下表中的 `.py` 文件，各文件顶部均有可编辑参数区，底部 `main()` 串起完整流程。本机 `.vscode/launch.json` 已提供从“财报 1”到“财报 6”的 Run / Debug 配置，依次对应下表；IDE 配置不提交仓库。

| 步骤 | 直接运行文件 | 解释器 | 参数 |
| --- | --- | --- | --- |
| 首次获取公开年报 | `financial_rag/download_reports.py` | `C:/Users/alpha_intern2/miniconda3/python.exe` | 无 |
| 重新解析全部年报 | `financial_rag/parse_and_chunk.py` | `C:/Users/alpha_intern2/miniconda3/python.exe` | 无 |
| 构建语义向量索引 | `financial_rag/search_engine.py` | `C:/Users/alpha_intern2/miniconda3/envs/py311_env/python.exe` | `--build-index` |
| 启动问答网页 | `financial_rag/web_api.py` | `C:/Users/alpha_intern2/miniconda3/envs/py311_env/python.exe` | 无 |
| 运行或续跑真实评测 | `financial_rag/run_benchmark.py` | `C:/Users/alpha_intern2/miniconda3/envs/py311_env/python.exe` | `--resume` |
| 生成一页结论（评测及独立审阅完成后） | `financial_rag/build_submission.py` | `C:/Users/alpha_intern2/miniconda3/python.exe` | 无 |

已有年报、全文切块和匹配的向量索引时，可直接启动网页。在浏览器打开 [本机问答页](http://127.0.0.1:8765/rag.html)，等待状态显示就绪，再输入新问题。网页和终端都会显示阶段提示；解析、建库与评测默认开启进度反馈，相应脚本可传 `--no-progress` 关闭。停止服务使用 IDE Stop 或终端 `Ctrl+C`。

依赖清单按实际使用的两个环境拆分，并记录本机已安装版本：`requirements-parser.txt` 对应 base 的 Python 3.13.12，包含 `requests`、`pdfplumber`、`pypdf`、`tqdm`、`reportlab`，用于解析与生成 PDF；`requirements.txt` 对应 `py311_env` 的 Python 3.11.15，包含 `requests`、`numpy`、`onnxruntime`、`tokenizers`、`tqdm`。模型客户端与网页服务本身使用标准库，服务调用的检索引擎仍依赖后者。Codex CLI 由现有安装提供，不属于 pip 依赖。结论预览使用本机 `C:/poppler/Library/bin/pdftoppm.EXE` 与 Windows 微软雅黑字体；换机时需调整 `build_submission.py` 顶部路径。

当前两个环境已具备这些依赖，无需重复安装。更换机器或新建环境时，可使用以下对应命令准备依赖，再调整 IDE 中的解释器与工作目录：

```powershell
& 'C:/Users/alpha_intern2/miniconda3/python.exe' -m pip install -r requirements-parser.txt
& 'C:/Users/alpha_intern2/miniconda3/envs/py311_env/python.exe' -m pip install -r requirements.txt
```

## 命令行入口

以下 PowerShell 命令均从 `D:/homepage-1` 执行。它们是 IDE 直接运行之外的等价入口。

```powershell
& 'C:/Users/alpha_intern2/miniconda3/python.exe' financial_rag/parse_and_chunk.py
& 'C:/Users/alpha_intern2/miniconda3/envs/py311_env/python.exe' financial_rag/search_engine.py --build-index
& 'C:/Users/alpha_intern2/miniconda3/envs/py311_env/python.exe' financial_rag/web_api.py
```

模块入口同样支持：

```powershell
& 'C:/Users/alpha_intern2/miniconda3/envs/py311_env/python.exe' -m financial_rag.web_api
& 'C:/Users/alpha_intern2/miniconda3/envs/py311_env/python.exe' -m financial_rag.run_benchmark --resume
```

只检查本机准备状态，不生成回答：

```powershell
& 'C:/Users/alpha_intern2/miniconda3/envs/py311_env/python.exe' financial_rag/web_api.py --check
& 'C:/Users/alpha_intern2/miniconda3/envs/py311_env/python.exe' financial_rag/model_client.py --status
```

`model_client.py --self-test` 会真实调用一次模型，但输入是明确虚构的两行测试数据。它只检查输出结构和工具事件，不属于财报评测。

## 全文语料与可追溯出处

数据来自 10 家白酒上市公司的 2023 年度报告：贵州茅台、五粮液、泸州老窖、山西汾酒、洋河股份、古井贡酒、舍得酒业、今世缘、迎驾贡酒、口子窖。公开 PDF 来源列于 `financial_rag/download_reports.py`。

本次全文重建记录为 **1,857 / 1,857 页、6,696 个切块**。以 `financial_rag/parse_manifest.json` 为准：逐公司保存文件摘要、PDF 总页数、处理页数、空文本页和页错误。`MAX_PAGES_PER_REPORT = None` 表示全文；设置页数上限只用于调试，不能据此宣称完成全文解析。

切块包含公司、年度、章节、类型、`chunk_id`、来源链接和物理 PDF 页码。正文采用滑动切块，表格分段保留表头与上下文。网页可点击答案中的 `[chunk_id]` 定位并展开实际切块，再打开来源 PDF 的对应物理页；物理页码可能与报告印刷页码不同。

遍历全部页只证明解析覆盖范围。合并单元格、跨页表头、数值列对应和表格单位仍可能存在解析误差，精确财务结论应回看原 PDF。

## 真实检索与模型生成

1. **语义检索**：本地 `Xenova/bge-small-zh-v1.5` 量化 ONNX，固定 revision `fcecc3c5fef6becfa2b2bdda15c1c938857be534`，首次下载模型约 24 MB。使用 CLS pooling 与 L2 归一化；每个切块的向量输入最多 512 tokens，不能把长块截断编码说成全文语义编码。
2. **关键词检索**：BM25 索引切块的完整 `content`，与向量结果通过 RRF 融合。索引清单记录语料哈希和模型版本，语料改变后必须重新建库。
3. **跨公司取证**：识别目标公司后，每家公司默认独立取 Top-6。全景题按公司分别取证；未召回足够信息的公司明确列为缺失，不能用头部公司的局部结果代表全部公司。
4. **真实生成**：`model_client.py` 通过 stdin 将问题和本次证据交给已登录的 Codex CLI，要求返回答案、引用、数值事实与可比公司排序。调用忽略用户项目配置，未指定模型名，使用 CLI 默认模型；审计中未取得具体模型 ID 时明确记为 `cli_default`，不需要 API key。
5. **输出校验**：核对引用清单及正文 `[chunk_id]` 是否属于本次召回，以及 claim 的公司、物理页、连续原文与数值锚点。数值比较保留正负号和百分比维度，检查原文明确标注的单位；正文带明确单位的数字须对应已核对 claim，金额仅接受精确等值换算，不接受自行计算或舍入。此处只是有限的格式、出处及数值一致性检查，不能保证句意、期间、表格列对应和会计口径准确。模型或连接失败会明确报错，不回退到预置答案。

公司覆盖同时保留 `coverage.cited`（引用涉及的公司）与 `coverage.supported`（数值题中至少有一条 claim 通过上述检查的公司）；引用到某公司不等于已回答该公司的全部问题。无数值 claim 的回答以引用公司作为覆盖提示，也不代表定性结论已被语义验证。输出校验失败时不展示草稿结论，清空有效事实、引用、排序及覆盖；原草稿存入明确标记 `validated: false` 的 `unvalidated_draft`，仅供审计被拒原因，不计为有效答案。

Codex 在临时目录、只读沙箱下运行，并关闭可关闭的 shell、插件、浏览器等能力；检测到工具执行事件时拒绝使用该次回答。这里不把这些设置描述为完全无工具隔离。`financial_rag/outputs/model_runs/` 只保留请求哈希、用量、状态等匿名审计摘要，不默认保存原始执行日志或账号信息。

## 评测方法与提交材料

`financial_rag/gold_cases.json` 是从原 PDF 独立核对的金标，只交给评分器，不传入检索或生成。当前包含原有问题与新增全文深页、口径辨析、跨公司及知识库外问题；每题应保留实际回答，而非评测前预填结论。

- **召回表现**：目标公司、物理页、科目行及数值证据是否进入本次召回。
- **严格结构化契约**：逐事实核对数值、单位、期间、合并或母公司口径、引用与原文；排序单独核对。该自动指标不等于财务事实正确率，低分也不能直接解释为所有数字错误。
- **整题结论**：数值事实通过率不能换算成整题正确率。定性解释、歧义题、梯队划分、自由叙述和拒答需独立查看原 PDF，再记录整题分类；当前审阅者为 Codex，未经过课程老师或用户本人验收。

实际问答保存到 `evaluation_responses.json`；逐题核验保存到 `evaluation_records.json` 与 `evaluation_records.md`；汇总保存到 `evaluation_summary.json`。`--resume` 只复用同题目、同语料哈希、同 `pipeline_version` 的已完成响应，旧版不能续入新版。`--responses` 可重新评分同一版本的完整历史真实响应，摘要会注明历史版本与当前版本；不能用虚构测试响应冒充真实评测。评测成绩以实际生成的文件为准，本说明不预先承诺通过率。

最终材料位于 `D:/homepage-1/financial_rag/`：

| 文件 | 内容 |
| --- | --- |
| [parse_manifest.json](financial_rag/parse_manifest.json) | 全文解析覆盖与逐报告记录 |
| [evaluation_responses.json](financial_rag/evaluation_responses.json) | 本轮真实模型响应与召回证据 |
| [evaluation_records.md](financial_rag/evaluation_records.md)、[evaluation_records.json](financial_rag/evaluation_records.json) | 逐题召回及严格结构化契约核验 |
| [evaluation_summary.json](financial_rag/evaluation_summary.json) | 本轮自动评测汇总 |
| [manual_review.md](financial_rag/manual_review.md)、[manual_review.json](financial_rag/manual_review.json) | Codex 独立逐题 PDF 审阅分类与理由，含对应响应文件哈希及全景题逐公司对照 |
| [one_page_conclusion.pdf](financial_rag/one_page_conclusion.pdf)、[HTML](financial_rag/one_page_conclusion.html)、[Markdown](financial_rag/one_page_conclusion.md) | 从同批实际评测与审阅生成的一页结论；此 PDF 是允许提交仓库的交付文件 |
| [submission_text.txt](financial_rag/submission_text.txt) | 可粘贴的方向、仓库链接及一句话结论 |
| [rag_web_screenshot.png](financial_rag/rag_web_screenshot.png)、[source_page_verification.png](financial_rag/source_page_verification.png) | 最终语料的真实网页操作截图与来源 PDF 对照截图 |
| [web_verification.md](financial_rag/web_verification.md)、[web_verification.json](financial_rag/web_verification.json) | 本机网页验收记录及语料、响应依据，不代表公开 Vercel 已更新 |

先完成最终语料的真实评测，再逐题独立审阅原 PDF 并更新 `manual_review.json`，最后运行 IDE“财报 6：生成一页结论”。生成器核对审阅的响应哈希与题数；不匹配时拒绝覆盖结论。重建语料或重跑评测期间，不应把留存的旧结论当作最终结果。下列两个生成命令等价，任选一个，且仅在上述评测与审阅完成后执行：

```powershell
& 'C:/Users/alpha_intern2/miniconda3/python.exe' financial_rag/build_submission.py
& 'C:/Users/alpha_intern2/miniconda3/python.exe' -m financial_rag.build_submission
```

生成器默认显示阶段进度，同时渲染预览到 `financial_rag/outputs/submission_preview/one_page_conclusion.png`，生成后应目视检查一页排版。预览目录已忽略；批处理可加 `--no-progress`。历史轮次位于 `financial_rag/baseline_v1/`、`financial_rag/baseline_v2/`，仅供复盘，不代替最终材料。

提交时应准备：源码、全文解析清单、真实自由提问演示、引用原 PDF 核对截图、独立评测记录，以及与评测结果一致的结论材料。公开网页链接是否可问答，需另外验证实际部署，不能用本机截图证明公开服务已上线。

## 本机运行与公开仓库边界

服务默认只监听 `127.0.0.1:8765`，检查本机 Host 和同源请求，并仅开放明确列出的页面与接口。现有 Vercel 静态托管不会运行此 Python 服务，也无法使用这台电脑上的 Codex 登录。不要把已登录的本机服务转发为公共问答端点。

源码和提交材料不写入 API key、密码、登录凭据或个人模型账户信息。本方案不要求填写 `.env`；`.env.example` 仅作说明，真实 `.env` 已忽略。不要提交 `auth.json`、原始登录输出、原始模型执行日志、原始财报 PDF、下载的模型、向量缓存或个人 IDE 配置。仓库 `.gitignore` 已覆盖 `.env`、原始 PDF、`financial_rag/models/`、`financial_rag/vector_index/`、`financial_rag/outputs/` 与 `.vscode/`，仅为交付物 `financial_rag/one_page_conclusion.pdf` 设置 PDF 例外；提交前仍应检查实际暂存文件。
