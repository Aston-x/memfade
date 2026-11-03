"""把核心包打进适配器目录，产出可直接安装的插件包。

为什么需要这一步：MaiBot 没有插件依赖安装机制，插件目录必须**自带**
所有代码。所以发布前要把 ``memfade/`` 复制到 ``adapters/maibot/memfade/``。

用法：

    python scripts/build_plugin.py            # 只同步核心到适配器目录
    python scripts/build_plugin.py --zip      # 顺便打一个 zip
    python scripts/build_plugin.py --dest D:\\plugins   # 顺便拷到别处

核心包本身没有第三方依赖，所以产物就是纯源码 + 一个 SQLite 库，
装完即用，不联网、不装包。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "memfade"
# 每个适配器都要自带一份核心源码——宿主不一定装得��� memfade 这个包，
# 插件目录必须自包含。目录布局各不相同，见各适配器的 _bootstrap_import。
ADAPTERS = {
    "maibot": REPO_ROOT / "adapters" / "maibot",
    "langbot": REPO_ROOT / "adapters" / "langbot",
    "astrbot": REPO_ROOT / "adapters" / "astrbot",
    "nonebot2": REPO_ROOT / "adapters" / "nonebot2" / "src",
}
# 能直接拷进宿主的形态。其它框架各有各的安装方式（pip 包 / 插件市场 /
# 编译产物），不能一锅端——之前这里没分开，--deploy 会把四个适配器轮流
# 覆盖到同一个目录里去。
#
# 每个 spec 三件事：
#   source            —— 源目录
#   dir_name          —— 装进宿主时的目录名
#   preserve_in_dir   —— 覆盖时要保留下来的文件（宿主上的现场配置）
#
# 目录名很关键：AstrBot 以 data.plugins.<目录名> 导入，目录名必须是合法
# Python 标识符（不能带横杠！MaiBot 的 memfade-plugin 在 AstrBot 下会直接
# 导入失败），而且它同时决定了配置文件名（<目录名>_config.json）和数据目录
# （data/plugin_data/<目录名>/）——改名等于用户配置失联。
DEPLOYABLE = {
    "maibot": {
        "source": REPO_ROOT / "adapters" / "maibot",
        "dir_name": "memfade-plugin",
        # MaiBot 的 config.toml 在插件目录内，覆盖时要保住
        "preserve_in_dir": ["config.toml"],
        "tools": None,  # None = 用默认全套
    },
    "astrbot": {
        "source": REPO_ROOT / "adapters" / "astrbot",
        "dir_name": "astrbot_plugin_memfade",
        # AstrBot 的配置在 data/config/<目录名>_config.json、数据在
        # data/plugin_data/<目录名>/，都在插件目录外——天然就保住了，不用管。
        "preserve_in_dir": [],
        # backup_memory_stack / retire_legacy_memory 写死了 MaiBot 的目录结构
        # 和插件 ID，在 AstrBot 下没有意义，不跟着发货。
        # verify_against_chat 已经去掉了 MaiBot 硬依赖（--host-root 走宿主，
        # 否则走 --llm-base-url 的 OpenAI 兼容接口），所以可以发。
        "tools": (
            "clean_library.py",
            "edit_memory.py",
            "audit_db.py",
            "verify_against_chat.py",
            "acceptance.py",
        ),
    },
}
IGNORED = {"__pycache__", ".pytest_cache", ".mypy_cache"}

# 各宿主的插件目录（相对宿主根目录）。
# ⚠ 用户不该靠记路径来装插件：MaiBot 是 ``<根>/plugins``，
# AstrBot 是 ``<根>/data/plugins``——给个 ``--host`` 就能推出来。
HOST_PLUGIN_DIRS = {
    "maibot": ("plugins",),
    "astrbot": ("data", "plugins"),
}

# 随插件一起发货的运维工具。
#
# 为什么：**插件要能独立用**。装上它的人需要的是
# 「备份我的记忆」「把旧记忆系统退场」「体检」「清理」——
# 这些工具必须跟着插件走，而不是留在开发者的仓库里。
OPERATOR_TOOLS = (
    "backup_memory_stack.py",
    "retire_legacy_memory.py",
    "clean_library.py",
    "edit_memory.py",
    "audit_db.py",
    "verify_against_chat.py",
    "acceptance.py",
)


def sync_core(target: Path) -> None:
    """把核心包复制到适配器目录。"""

    dst = target / "memfade"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(CORE_DIR, dst, ignore=shutil.ignore_patterns(*IGNORED, "*.pyc"))
    print(f"[build] 核心已同步到 {dst}")


def sync_tools(target: Path, names=None) -> None:
    """把运维工具复制进适配器目录的 tools/ 下。"""

    # ⚠ 目录名带前缀：宿主会把插件目录放进 sys.path，
    # 一个叫 `tools` 的目录可能和别人的同名包打架。
    dst = target / "memfade_tools"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in (names or OPERATOR_TOOLS):
        src = REPO_ROOT / "scripts" / name
        if not src.exists():
            continue
        shutil.copy2(src, dst / name)
        copied += 1
    if copied:
        print(f"[build] 运维工具已同步到 {dst}（{copied} 个）")


def deploy(spec: dict, plugins_dir: Path) -> Path:
    """把适配器目录部署到宿主的 plugins 目录。

    目录名来自 spec["dir_name"]。MaiBot 用 memfade-plugin 而不是 memfade：
    宿主加载插件时会把 plugins/ 目录放进 sys.path，如果插件目录就叫 memfade，
    那个目录本身会被当成同名包，把插件自带的 memfade/ 核心包遮蔽掉。
    AstrBot 用 astrbot_plugin_memfade：它以 data.plugins.<目录名> 导入，
    目录名必须是合法 Python 标识符，而且它同时决定配置文件名和数据目录。
    """

    dst = plugins_dir / spec["dir_name"]
    # ⚠ 宿主上的现场配置（改过的阈值、开关都在里面）不能被仓库模板抹掉。
    saved = {}
    for rel in spec.get("preserve_in_dir") or []:
        p = dst / rel
        if p.exists():
            saved[rel] = p.read_bytes()
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        spec["source"],
        dst,
        ignore=shutil.ignore_patterns(*IGNORED, "*.pyc"),
    )
    for rel, blob in saved.items():
        (dst / rel).write_bytes(blob)
        print(f"[build] 已保留宿主上的 {rel}")
    print(f"[build] 已部署到 {dst}")
    return dst


def make_zip(spec: dict, out_dir: Path) -> Path:
    """打成可分发的 zip。

    zip 里的根目录名用 spec["dir_name"]——用 target.name 的话，AstrBot 那边
    解压出来会是 ``astrbot/``，而它的配置文件名和数据目录都跟着目录名走，
    名字对不上就等于用户配置失联。
    """

    target = spec["source"]
    root = spec["dir_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"memfade-{root}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(target.rglob("*")):
            if path.is_dir() or any(part in IGNORED for part in path.parts):
                continue
            if path.suffix == ".pyc":
                continue
            zf.write(path, Path(root) / path.relative_to(target))
    print(f"[build] 已打包 {archive}")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 memfade 适配器插件包")
    parser.add_argument("--zip", action="store_true", help="额外产出 zip")
    parser.add_argument("--dest", default="", help="额外复制一份到指定目录")
    parser.add_argument("--deploy", default="", help="部署到指定的宿主 plugins 目录")
    parser.add_argument(
        "--host",
        default="",
        choices=sorted(HOST_PLUGIN_DIRS),
        help="按宿主部署（配合 --host-root 自动推出插件目录，不用记路径）",
    )
    parser.add_argument(
        "--host-root",
        default="",
        help="宿主根目录（--host 用）：MaiBot 的项目根 / AstrBot 的安装目录",
    )
    args = parser.parse_args()

    if not CORE_DIR.is_dir():
        print(f"[build] 找不到核心包 {CORE_DIR}", file=sys.stderr)
        return 1

    for name, target in ADAPTERS.items():
        sync_core(target)
        sync_tools(target)
    deploy_dir = args.deploy
    if args.host:
        if not args.host_root:
            print(
                "[build] --host 要配 --host-root（宿主根目录）："
                "MaiBot 的项目根 / AstrBot 的安装目录",
            )
            return 2
        # --host 时只部署那一个适配器：另一个宿主的目录结构不一样，
        # 一起部署会把文件铺到不该铺的地方。
        deploy_dir = os.path.join(
            args.host_root, *HOST_PLUGIN_DIRS[args.host]
        )
        print("[build] 目标：%s（%s）" % (deploy_dir, args.host))

    for name, spec in DEPLOYABLE.items():
        # 工具清单按适配器裁剪：AstrBot 不发 MaiBot 专属的那两个
        # ⚠ 这一步是**整理源码树**，跟部署到哪个宿主无关——
        # 所以必须放在 --host 的过滤**之前**。
        # 放后面的话，「只部署 maibot」会把 astrbot 的工具清单打乱
        # （两个工具又冒出来），源码树跟着部署行为漂移。
        tools = spec.get("tools")
        if tools is not None:
            sync_tools(spec["source"], tools)
        if args.host and name != args.host:
            continue
        if args.zip:
            make_zip(spec, REPO_ROOT / "dist")
        if args.dest:
            dest = Path(args.dest) / f"memfade-{name}"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(
                spec["source"], dest, ignore=shutil.ignore_patterns(*IGNORED, "*.pyc")
            )
            print(f"[build] 已复制到 {dest}")
        if deploy_dir:
            deploy(spec, Path(deploy_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

