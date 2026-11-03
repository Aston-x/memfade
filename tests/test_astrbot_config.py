"""AstrBot 适配器的配置一致性测试。

防的是两类老问题：

1. **配了不生效**——``_conf_schema.json`` 里写着的键，代码根本没读；
2. **代码读了但面板没有**——代码读了一个 schema 里不存在的键，用户在面板上
   根本找不到这个选项。

还守住三处命名/版本的一致性：目录名、metadata name、``@register`` 名一旦
对不上，用户的配置文件和数据目录就会失联。
"""

import json
import os
import re
import unittest

from tests.helpers import temp_dir

ADAPTER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "adapters", "astrbot"
)


def _install_stubs():
    """装最小 astrbot.api 桩件（复用适配器测试里那套）。"""

    from tests.test_astrbot_adapter import _install_stubs as install

    install()


def _read_metadata():
    """手工解析 metadata.yaml（不引 PyYAML——测试环境不一定装了）。"""

    out = {}
    for line in _read("metadata.yaml").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip().strip(chr(34) + chr(39))
    return out


def _read(name):
    with open(os.path.join(ADAPTER_DIR, name), encoding="utf-8") as f:
        return f.read()


class TestSchemaMatchesCode(unittest.TestCase):
    """schema 与代码读写的键必须对得上。"""

    def setUp(self):
        _install_stubs()
        self.schema = json.loads(_read("_conf_schema.json"))
        self.source = _read("main.py")

    def _code_keys(self):
        keys = set()
        for pattern in (r'raw\.get\(\s*"([a-z0-9_]+)"', r'self\.config\.get\(\s*"([a-z0-9_]+)"'):
            keys.update(re.findall(pattern, self.source))
        return keys

    def test_schema_has_no_dead_keys(self):
        """schema 里的每个键，代码都要真的读它。"""

        code_keys = self._code_keys()
        dead = sorted(set(self.schema) - code_keys)
        self.assertEqual(
            dead, [], "这些配置项在面板上有，但代码从不读取（配了等于没配）：%s" % dead
        )

    def test_code_has_no_missing_keys(self):
        """代码读的每个键，schema 里都要有。"""

        code_keys = self._code_keys()
        missing = sorted(code_keys - set(self.schema))
        self.assertEqual(
            missing, [], "这些配置项代码在读，但面板上没有入口：%s" % missing
        )

    def test_importance_default_matches_core(self):
        """入库门槛必须和核心默认一致（0.45），否则噪声会进库。"""

        self.assertEqual(self.schema["min_importance"]["default"], 0.45)

    def test_cross_scope_defaults_off(self):
        """cross_scope=1 的记忆对所有作用域可见，多用户部署必须默认关。"""

        self.assertFalse(self.schema["cross_scope"]["default"])


class TestNaming(unittest.TestCase):
    """四处名称必须一致，否则用户配置会失联。"""

    def setUp(self):
        _install_stubs()

    def test_metadata_name_matches_register_name(self):
        meta = _read_metadata()
        source = _read("main.py")
        match = re.search(r'PLUGIN_NAME\s*=\s*"([^"]+)"', source)
        self.assertIsNotNone(match, "main.py 里找不到 PLUGIN_NAME")
        self.assertEqual(meta["name"], match.group(1))

    def test_version_matches_core(self):
        from memfade import __version__ as core_version

        meta = _read_metadata()
        self.assertEqual(
            str(meta["version"]),
            str(core_version),
            "metadata 的版本和核心 __version__ 对不上",
        )

    def test_plugin_name_is_a_valid_python_identifier(self):
        """AstrBot 以 data.plugins.<目录名> 导入，目录名带横杠会直接导入失败。"""

        meta = _read_metadata()
        name = str(meta["name"])
        self.assertTrue(name.isidentifier(), "插件名不是合法 Python 标识符：%s" % name)


class TestConfigMapping(unittest.TestCase):
    """配置翻译到核心配置的正确性。"""

    def setUp(self):
        _install_stubs()

    def _plugin(self, raw):
        """造一个不碰真实数据库的插件实例。"""

        import tempfile

        from adapters.astrbot.main import MemFadePlugin

        cfg = dict(raw)
        cfg.setdefault("db_path", os.path.join(temp_dir("memfade-cfg-"), "m.db"))
        plugin = MemFadePlugin(context=None, config=cfg)
        self.addCleanup(plugin.engine.close)
        return plugin

    def test_archive_after_days_is_no_longer_dead(self):
        """以前 archive_after_days 在 schema 里却从没生效——现在要真的传到 fade 节。"""

        plugin = self._plugin({"archive_after_days": 7})
        self.assertEqual(plugin.engine.config.fade.archive_after_days, 7)

    def test_recall_alpha_reaches_core(self):
        plugin = self._plugin({"recall_alpha": 0.5})
        self.assertEqual(plugin.engine.config.fade.recall_alpha, 0.5)

    def test_maintenance_interval_is_translated_to_seconds(self):
        plugin = self._plugin({"maintain_interval_minutes": 15})
        self.assertEqual(plugin.engine.config.maintenance.interval_seconds, 900)

    def test_vector_channel_off_without_embedder(self):
        """没有 embedder 时必须关掉向量通道，否则面板显示开了实际没开。"""

        plugin = self._plugin({"use_embedding": True})
        self.assertFalse(plugin.engine.config.storage.enable_vector)
        self.assertIsNone(plugin._embedder)

    def test_db_path_is_absolute(self):
        """相对路径会落到进程 CWD，几个插件互相踩。"""

        plugin = self._plugin({"db_path": "memfade.db"})
        self.assertTrue(os.path.isabs(plugin.engine.config.storage.db_path))

    def test_config_none_is_tolerated(self):
        """宿主没给配置时只传 context，插件不能崩。"""

        import tempfile

        from adapters.astrbot.main import MemFadePlugin

        plugin = MemFadePlugin(
            context=None,
            config=None,
        )
        self.addCleanup(plugin.engine.close)
        self.assertEqual(plugin.config, {})


if __name__ == "__main__":
    unittest.main()
