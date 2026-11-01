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
IGNORED = {"__pycache__", ".pytest_cache", ".mypy_cache"}


def sync_core(target: Path) -> None:
    """把核心包复制到适配器目录。"""

    dst = target / "memfade"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(CORE_DIR, dst, ignore=shutil.ignore_patterns(*IGNORED, "*.pyc"))
    print(f"[build] 核心已同步到 {dst}")


def deploy(target: Path, plugins_dir: Path) -> Path:
    """把适配器目录部署到宿主的 plugins 目录。

    目录名用 ``memfade-plugin`` 而不是 ``memfade``：
    宿主加载插件时会把 plugins/ 目录放进 sys.path，如果插件目录就叫 memfade，
    那个目录本身会被当成同名包，把插件自带的 memfade/ 核心包遮蔽掉。
    """

    dst = plugins_dir / "memfade-plugin"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        target,
        dst,
        ignore=shutil.ignore_patterns(*IGNORED, "*.pyc"),
    )
    print(f"[build] 已部署到 {dst}")
    return dst


def make_zip(target: Path, out_dir: Path) -> Path:
    """打成可分发的 zip。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"memfade-{target.name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(target.rglob("*")):
            if path.is_dir() or any(part in IGNORED for part in path.parts):
                continue
            if path.suffix == ".pyc":
                continue
            zf.write(path, Path(target.name) / path.relative_to(target))
    print(f"[build] 已打包 {archive}")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 memfade 适配器插件包")
    parser.add_argument("--zip", action="store_true", help="额外产出 zip")
    parser.add_argument("--dest", default="", help="额外复制一份到指定目录")
    parser.add_argument("--deploy", default="", help="部署到指定的宿主 plugins 目录")
    args = parser.parse_args()

    if not CORE_DIR.is_dir():
        print(f"[build] 找不到核心包 {CORE_DIR}", file=sys.stderr)
        return 1

    for name, target in ADAPTERS.items():
        sync_core(target)
        if args.zip:
            make_zip(target, REPO_ROOT / "dist")
        if args.dest:
            dest = Path(args.dest) / f"memfade-{name}"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(target, dest, ignore=shutil.ignore_patterns(*IGNORED, "*.pyc"))
            print(f"[build] 已复制到 {dest}")
        if args.deploy:
            deploy(target, Path(args.deploy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

