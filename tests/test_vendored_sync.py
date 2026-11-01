"""检查适配器里自带的核心副本与源目录一致。

MaiBot 没有插件依赖安装机制，插件目录必须自带核心源码。
但复制品很容易和源目录漂移——改了 ``memfade/`` 忘了同步，
用户装到的就是旧版。这里把它钉死。
"""

from __future__ import annotations

import filecmp
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE = REPO_ROOT / "memfade"
VENDORED_DIRS = [
    REPO_ROOT / "adapters" / "maibot" / "memfade",
    REPO_ROOT / "adapters" / "langbot" / "memfade",
    REPO_ROOT / "adapters" / "astrbot" / "memfade",
    REPO_ROOT / "adapters" / "nonebot2" / "src" / "memfade",
]


def _files(root: Path) -> set:
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


class TestVendoredSync(unittest.TestCase):
    def test_same_file_set(self):
        for vendored in VENDORED_DIRS:
            with self.subTest(dir=str(vendored)):
                self.assertTrue(
                    vendored.is_dir(),
                    "适配器目录里应该有一份核心副本（跑 scripts/build_plugin.py 生成）",
                )
                self.assertEqual(_files(CORE), _files(vendored), "核心副本的文件清单与源目录不一致")

    def test_same_content(self):
        for vendored in VENDORED_DIRS:
            if not vendored.is_dir():
                continue
            for rel in sorted(_files(CORE)):
                with self.subTest(dir=vendored.name, rel=rel):
                    self.assertTrue(
                        filecmp.cmp(CORE / rel, vendored / rel, shallow=False),
                        f"{rel} 与源目录不一致，跑 scripts/build_plugin.py 同步",
                    )


if __name__ == "__main__":
    unittest.main()

