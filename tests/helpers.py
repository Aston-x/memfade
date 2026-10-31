"""测试公共件。"""

from __future__ import annotations

import os
import tempfile

from memfade.config import MemFadeConfig
from memfade.engine import MemoryEngine


def temp_engine(**config_overrides) -> tuple:
    """建一个落在临时目录里的引擎，返回 (engine, 路径, config)。"""

    handle, path = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    os.remove(path)
    config = MemFadeConfig.from_dict(config_overrides or None)
    config.storage.db_path = path
    engine = MemoryEngine(config)
    return engine, path, config

