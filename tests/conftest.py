import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Criteria, Settings  # noqa: E402


@pytest.fixture
def criteria() -> Criteria:
    return Criteria.load(ROOT / "criteria.example.yaml")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_path=tmp_path / "leads.db",
        fetch_pages=False,
        search_delay=0,
        llm_max_retries=2,
        anthropic_api_key="test-key",
        openai_api_key="test-key",
    )


class MockServer:
    """Local HTTP server. Queue responses per path; records every request."""

    def __init__(self):
        self.requests: list[dict] = []
        self.responses: dict[str, list[tuple[int, dict, dict]]] = {}
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                server.requests.append({
                    "method": self.command,
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "json": json.loads(raw) if raw else None,
                })
                queue = server.responses.get(self.path.split("?")[0], [])
                status, body, headers = queue.pop(0) if queue else (404, {"error": "no mock"}, {})
                payload = body.encode() if isinstance(body, str) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", headers.pop("Content-Type", "application/json"))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST = _reply

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def add(self, path: str, status: int = 200, body=None, headers: dict | None = None):
        self.responses.setdefault(path, []).append((status, body if body is not None else {}, headers or {}))


@pytest.fixture
def mock_server():
    server = MockServer()
    yield server
    server.httpd.shutdown()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    import analyzer, main, notifier, searcher
    for mod in (analyzer, main, notifier, searcher):
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
