"""HTTP 侧车测试。

起一个真实的本地 HTTP 服务，走完整协议验证。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.http.server import MemFadeHTTPService, MemFadeHandler  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402
from memfade import MemFadeConfig  # noqa: E402
from memfade.engine import MemoryEngine  # noqa: E402


class TestHTTPSidecar(unittest.TestCase):
    def setUp(self):
        handle, self.db = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        os.remove(self.db)
        config = MemFadeConfig.from_dict(
            {"storage": {"db_path": self.db, "enable_vector": False}, "extraction": {"min_importance": 0.2}}
        )
        self.engine = MemoryEngine(config)
        self.service = MemFadeHTTPService(self.engine)
        self.service.start_loop()
        handler = type("H", (MemFadeHandler,), {"service": self.service, "token": "t0ken"})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.service.stop_loop()
        self.engine.close()
        for suffix in ("", "-wal", "-shm"):
            path = self.db + suffix
            if os.path.exists(path):
                os.remove(path)

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-MemFade-Token": "t0ken"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            headers={"X-MemFade-Token": "t0ken"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_health(self):
        result = self._get("/health")
        self.assertTrue(result["ok"])

    def test_unauthorized(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/health")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=15)
        self.assertEqual(ctx.exception.code, 401)

    def test_remember_and_context(self):
        scope = {"platform": "qq", "chat_type": "private", "chat_id": "u1"}
        result = self._post(
            "/remember",
            {"scope": scope, "content": "许杏玖喜欢吃辣的食物", "kind": "semantic", "importance": 0.9},
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["written"], 1)
        ctx = self._post("/context", {"scope": scope, "query": "今天吃什么"})
        self.assertIn("许杏玖喜欢吃辣的食物", ctx["text"])
        self.assertEqual(len(ctx["facts"]), 1)

    def test_scope_isolation_over_http(self):
        group = {"platform": "qq", "chat_type": "group", "chat_id": "g1"}
        private = {"platform": "qq", "chat_type": "private", "chat_id": "u1"}
        self._post("/remember", {"scope": private, "content": "私聊的机密内容", "importance": 0.9})
        ctx = self._post("/context", {"scope": group, "query": "机密内容"})
        self.assertNotIn("私聊的机密内容", ctx["text"])

    def test_stats(self):
        result = self._get("/stats")
        self.assertTrue(result["ok"])
        self.assertIn("by_status", result["stats"])

    def test_consolidate(self):
        result = self._post("/consolidate", {})
        self.assertTrue(result["ok"])
        self.assertIn("scanned", result["report"])

    def test_long_content_is_split(self):
        """侧车写进来的长内容也要拆成「一件事一条」。

        侧车是给别的框架用的——写进来的东西多半没人再管，
        一段 200 字塞进去，注入时只能截断。
        """

        scope = {"platform": "qq", "chat_type": "private", "chat_id": "u1"}
        long_text = (
            "季怡在做小红书账号，发AI绘画内容，目标是靠流量和接单赚钱，"
            "10月起零关注零粉丝，只有几个赞，曾被卖家临时加价拉黑二十块"
        )
        result = self._post(
            "/remember", {"scope": scope, "content": long_text, "importance": 0.9}
        )
        self.assertGreater(result["written"], 1, "长内容应该被拆开")
        for mid in result["ids"]:
            mem = self.engine.store.get(mid)
            self.assertLessEqual(len(mem.content), 40)

    def test_recall_requires_match(self):
        """显式检索必须真的命中。"""

        scope = {"platform": "qq", "chat_type": "private", "chat_id": "u1"}
        self._post("/remember", {"scope": scope, "content": "许杏玖喜欢辣的食物", "importance": 0.9})
        result = self._post("/recall", {"scope": scope, "query": "完全不相关的东西xyz", "top_k": 8})
        self.assertEqual(len(result["hits"]), 0)
        result = self._post("/recall", {"scope": scope, "query": "辣", "top_k": 8})
        self.assertEqual(len(result["hits"]), 1, "单字查询也要能查到")


if __name__ == "__main__":
    unittest.main()

