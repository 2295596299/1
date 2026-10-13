"""在本机展示财报问答页面，并将新问题交给真实检索与模型生成流程。"""

import argparse
import hashlib
import json
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ==================== 可编辑参数区：IDE 直接运行本文件 ====================
PORT = 8765
SHOW_PROGRESS = True
CURRENT_DIR = Path(__file__).resolve().parent
SITE_DIR = CURRENT_DIR.parent
MAX_REQUEST_BYTES = 8192
MAX_QUESTION_CHARS = 2000
STATIC_FILES = {"/": "rag.html", "/rag.html": "rag.html", "/index.html": "index.html"}

_ask_lock = threading.Lock()
_status_lock = threading.Lock()
_model_status_cache = (0.0, {})
_retriever = None


def get_status() -> dict:
    """检查全文覆盖、向量索引摘要与本机模型可用性，不下载模型或执行推理。

    模型登录状态最多缓存 15 秒；语料与索引摘要每次重新核对。

    Args:
        无；读取本项目的 chunks.json、parse_manifest.json 和索引清单。

    Returns:
        dict，包含 ready、chunkCount、manifest、model、provider、message；不含凭据。
    """
    global _model_status_cache
    with _status_lock:
        checked_at, model_state = _model_status_cache
        if time.monotonic() - checked_at >= 15 or not model_state:
            try:
                from financial_rag.model_client import get_model_status
                model_state = get_model_status()
            except (ImportError, OSError, RuntimeError):
                model_state = {"ready": False, "model": "", "provider": "", "message": "模型客户端尚未就绪。"}
            _model_status_cache = (time.monotonic(), model_state)
    state = {
        "ready": False, "chunkCount": 0, "manifest": {},
        "model": model_state.get("model", ""), "provider": model_state.get("provider", ""),
        "message": "全文语料或向量索引尚未就绪。",
    }
    try:
        manifest = json.loads((CURRENT_DIR / "parse_manifest.json").read_text(encoding="utf-8"))
        state["manifest"] = manifest
        state["chunkCount"] = manifest.get("total_chunks", 0)
        raw = (CURRENT_DIR / "chunks.json").read_bytes()
        index = json.loads((CURRENT_DIR / "vector_index" / "manifest.json").read_text(encoding="utf-8"))
        valid_index = (
            manifest.get("status") == "complete"
            and manifest.get("processed_pages") == manifest.get("total_pages")
            and index.get("corpus_sha256") == hashlib.sha256(raw).hexdigest()
            and index.get("chunk_count") == state["chunkCount"] > 0
            and (CURRENT_DIR / "vector_index" / "vectors.npy").is_file()
        )
        if valid_index:
            state["ready"] = bool(model_state.get("ready"))
            state["message"] = "全文索引与模型已就绪，可以输入新问题。" if state["ready"] else model_state.get("message", "模型尚未就绪。")
        else:
            state["message"] = "索引与全文语料不一致或尚未构建完成，请重新建库。"
    except (OSError, ValueError, TypeError):
        pass
    return state


class LocalRagHandler(SimpleHTTPRequestHandler):
    """只提供明确白名单页面和本机问答 API，不公开项目目录。"""

    def respond(self, status: int, payload: dict | bytes, content_type: str = "application/json; charset=utf-8") -> None:
        """发送固定长度的 JSON 或静态页面响应，并阻止浏览器跨源读取。

        Args:
            status: HTTP 状态码。
            payload: JSON 字典或已从白名单文件读取的 bytes。
            content_type: 明确的响应 MIME 类型，默认 UTF-8 JSON。

        Returns:
            None，响应直接写入当前 HTTP 连接；HEAD 请求只发送头部。
        """
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if isinstance(payload, dict) else payload
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def local_request(self) -> bool:
        """校验 Host 与可选 Origin，防止非本机域名或跨站页面调用登录态模型。

        Args:
            无；使用当前请求头和服务实际监听端口。

        Returns:
            bool，合法本机同源请求为 True；否则已发送 403 并返回 False。
        """
        try:
            hosts = self.headers.get_all("Host", [])
            if len(hosts) != 1:
                raise ValueError("Host")
            host = urlsplit("//" + hosts[0])
            if (host.hostname not in {"127.0.0.1", "localhost"}
                    or host.port != self.server.server_port or host.username is not None
                    or host.password is not None or host.path or host.query or host.fragment):
                raise ValueError("Host")
            origins = self.headers.get_all("Origin", [])
            if len(origins) > 1:
                raise ValueError("Origin")
            if origins:
                origin = urlsplit(origins[0])
                if (origin.scheme != "http" or origin.hostname != host.hostname
                        or origin.port != self.server.server_port or origin.username is not None
                        or origin.password is not None or origin.path or origin.query or origin.fragment):
                    raise ValueError("Origin")
            return True
        except ValueError:
            self.respond(403, {"error": "仅接受本机页面的同源请求。"})
            return False

    def do_GET(self) -> None:
        """返回状态或白名单 HTML 文件，其他路径统一拒绝。

        Args:
            无；从当前请求路径分发。

        Returns:
            None，发送状态 JSON、HTML 或 404；不调用父类目录处理逻辑。
        """
        if not self.local_request():
            return
        route = urlsplit(self.path).path
        if route == "/api/status":
            self.respond(200, get_status())
        elif route in STATIC_FILES:
            path = SITE_DIR / STATIC_FILES[route]
            if path.is_file():
                self.respond(200, path.read_bytes(), "text/html; charset=utf-8")
            else:
                self.respond(404, {"error": "页面文件不存在。"})
        else:
            self.respond(404, {"error": "未开放此路径。"})

    do_HEAD = do_GET

    def do_POST(self) -> None:
        """校验新问题后串行调用同一检索器，直接返回真实问答结果。

        Args:
            无；请求体必须为含 query 字符串的 UTF-8 JSON 对象。

        Returns:
            None，发送问答 JSON；输入错误、忙碌或未就绪时分别返回对应状态码。
        """
        global _retriever
        if not self.local_request():
            return
        if urlsplit(self.path).path != "/api/ask":
            self.respond(404, {"error": "未开放此路径。"})
            return
        if self.headers.get_content_type() != "application/json":
            self.respond(415, {"error": "Content-Type 必须为 application/json。"})
            return
        try:
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
                raise ValueError("Content-Length")
            length = int(lengths[0])
            if length <= 0:
                raise ValueError("Content-Length")
            if length > MAX_REQUEST_BYTES:
                self.respond(413, {"error": "问题请求过大。"})
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            query = payload.get("query") if isinstance(payload, dict) else None
            if not isinstance(query, str) or not 1 <= len(query.strip()) <= MAX_QUESTION_CHARS:
                raise ValueError("query")
            query = query.strip()
        except (ValueError, UnicodeDecodeError):
            self.respond(400, {"error": "请提交含有效 query 字符串的 JSON 问题。"})
            return
        if not _ask_lock.acquire(blocking=False):
            self.respond(429, {"error": "正在处理上一条问题，请稍后重试。"})
            return
        try:
            state = get_status()
            if not state["ready"]:
                self.respond(503, {"error": state["message"]})
                return
            corpus_hash = hashlib.sha256((CURRENT_DIR / "chunks.json").read_bytes()).hexdigest()
            if _retriever is None or _retriever.corpus_sha256 != corpus_hash:
                from financial_rag.search_engine import FinancialHybridRetriever
                _retriever = FinancialHybridRetriever(show_progress=SHOW_PROGRESS)
            result = _retriever.answer(query, top_k=6, show_progress=SHOW_PROGRESS)
            self.respond(200, result)
        except Exception as exc:
            if SHOW_PROGRESS:
                print(f"[问答失败] {type(exc).__name__}", file=sys.stderr, flush=True)
            self.respond(500, {"error": "本次问答未完成，请检查本机服务状态后重试。"})
        finally:
            _ask_lock.release()


def main() -> None:
    """从 IDE、脚本或模块启动本机服务，或只检查准备状态。

    Args:
        无；PORT、SHOW_PROGRESS 可在顶部编辑，命令行支持 --check、--no-progress。

    Returns:
        None，服务运行至 Ctrl+C；--check 打印状态后退出。
    """
    global SHOW_PROGRESS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="仅检查语料、索引和模型状态")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    SHOW_PROGRESS = SHOW_PROGRESS and not args.no_progress
    if args.check:
        print(json.dumps(get_status(), ensure_ascii=False, indent=2))
        return
    with ThreadingHTTPServer(("127.0.0.1", PORT), LocalRagHandler) as server:
        if SHOW_PROGRESS:
            print(f"[服务] 本机页面 http://127.0.0.1:{PORT}/rag.html ，按 Ctrl+C 停止。", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    main()
