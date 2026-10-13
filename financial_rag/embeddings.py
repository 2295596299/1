"""使用本地 BGE 中文预训练模型计算真实稠密向量，无需 embedding API key。"""

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
from tokenizers import Tokenizer
from tqdm import tqdm

# ==================== 可编辑参数区 ====================
MODEL_REPO = "Xenova/bge-small-zh-v1.5"
MODEL_REVISION = "fcecc3c5fef6becfa2b2bdda15c1c938857be534"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "bge-small-zh-v1.5"
MAX_TOKENS = 512
CPU_THREADS = 4


class LocalEmbedding:
    """本地 ONNX BGE embedding；遵循模型文档的 CLS pooling 和 L2 归一化。"""

    def __init__(self, show_progress: bool = True):
        """准备固定版本的模型资产并启动 CPU 推理会话。

        Args:
            show_progress: 是否展示首次下载的阶段提示与字节进度。

        Returns:
            None: 初始化 tokenizer 和 ONNX 会话，模型存于本地 models 目录。
        """
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for remote, filename in (("onnx/model_quantized.onnx", "model_quantized.onnx"), ("tokenizer.json", "tokenizer.json")):
            target = MODEL_DIR / filename
            if target.exists():
                continue
            if show_progress:
                print(f"[模型下载] {MODEL_REPO}/{remote}", flush=True)
            url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{remote}"
            partial = target.with_suffix(target.suffix + ".part")
            with requests.get(url, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                with partial.open("wb") as stream, tqdm(total=int(response.headers.get("Content-Length", 0)) or None, unit="B", unit_scale=True, disable=not show_progress) as progress:
                    for block in response.iter_content(1024 * 256):
                        stream.write(block)
                        progress.update(len(block))
            partial.replace(target)
        self.tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = CPU_THREADS
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(MODEL_DIR / "model_quantized.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        self.model_sha256 = hashlib.sha256((MODEL_DIR / "model_quantized.onnx").read_bytes()).hexdigest()
        self.truncated_count = 0

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """计算文本批次的单位向量，长块的向量输入最多 512 tokens。

        BM25 仍索引完整切块；向量截断数另行记录，不把截断视为全文语义编码。

        Args:
            texts: 待编码的非空文本列表。
            is_query: 查询加模型推荐中文检索指令，语料不加。

        Returns:
            np.ndarray: float32 的 (N, D) 归一化 CLS 向量。
        """
        if is_query:
            texts = ["为这个句子生成表示以用于检索相关文章：" + text for text in texts]
        encoded = self.tokenizer.encode_batch(texts)
        self.truncated_count += sum(bool(item.overflowing) for item in encoded)
        arrays = {
            "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
            "attention_mask": np.asarray([item.attention_mask for item in encoded], dtype=np.int64),
            "token_type_ids": np.asarray([item.type_ids for item in encoded], dtype=np.int64),
        }
        outputs = self.session.run(None, {entry.name: arrays[entry.name] for entry in self.session.get_inputs()})
        vectors = outputs[0][:, 0, :].astype(np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
