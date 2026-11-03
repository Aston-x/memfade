"""测试公共件。"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

from memfade.config import MemFadeConfig
from memfade.engine import MemoryEngine

# 临时目录登记表。
#
# 为什么要有这个：`tempfile.mkdtemp()` 建出来的目录**不会**被自动删——
# 每次跑测试都会在系统临时目录里留下一地 `memfade-*` 空壳。
# 跑一次几百个，跑几十次就是几千个（C 盘就这么被吃掉了）。
#
# 所以这里统一登记，进程退出时一次性清掉：
# 就算测试中途炸了、断言失败了，atexit 也会兜底。
_TEMP_DIRS: list[str] = []


def temp_dir(prefix: str = "memfade-test-") -> str:
    """建一个临时目录，进程退出时自动删。

    替代裸的 `tempfile.mkdtemp()`——后者在测试里留下的垃圾没人管。
    """

    path = tempfile.mkdtemp(prefix=prefix)
    _TEMP_DIRS.append(path)
    return path


@atexit.register
def _cleanup_temp_dirs() -> None:
    """进程退出时清掉所有登记过的临时目录。"""

    while _TEMP_DIRS:
        shutil.rmtree(_TEMP_DIRS.pop(), ignore_errors=True)


def temp_engine(**config_overrides) -> tuple:
    """建一个落在临时目录里的引擎，返回 (engine, 路径, config)。"""

    handle, path = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    os.remove(path)
    config = MemFadeConfig.from_dict(config_overrides or None)
    config.storage.db_path = path
    engine = MemoryEngine(config)
    return engine, path, config

