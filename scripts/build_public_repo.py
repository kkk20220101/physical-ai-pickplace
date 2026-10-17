#!/usr/bin/env python
"""公開リポジトリを**作業ツリーとは別の場所に**組み立てる。

**作業ツリーで `git init` しない**（意思決定 DEC-33）。
作業ツリーには社内限りの文書（`docs/02_要件定義.md`・`tasks/` 等）が同居しており、
`git add` を一度でも誤ると**履歴に残って取り消せない**。
∴ **allow-list に載ったものだけを空のディレクトリへ複製してから init する。**
出せないものは「除外し忘れる」のではなく、**そもそもそこに無い**状態にする。

push は行わない（外部送信は人の確認が要る）。最後に push コマンドだけ表示する。

使い方:
    python scripts/build_public_repo.py
    python scripts/build_public_repo.py --account <GitHubアカウント>
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from check_publish_safety import (  # noqa: E402
        MANIFEST,
        ROOT,
        is_ignored,
        is_published,
        load_gitignore,
        load_manifest,
        scan_personal_paths,
        scan_terms,
    )
except ImportError:
    # 検査器（禁止語一覧を持つ）は社内限りで、公開リポジトリには入っていない（DEC-34）。
    # 公開リポジトリを組む操作は検査器なしでは安全に行えないので、traceback でなく
    # 理由を言って止める。
    raise SystemExit(
        "このスクリプトは社内環境専用です。検査器 scripts/check_publish_safety.py"
        "（社内限り・DEC-34）が見つからないため、公開リポジトリは組み立てられません。")

DEFAULT_OUT = ROOT / "outputs" / "public_repo"
REPO_NAME = "physical-ai-pickplace"
PUBLIC_README = "README_public.md"


def collect(root: Path) -> list[str]:
    patterns = load_manifest(MANIFEST)
    rules = load_gitignore(root / ".gitignore")
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if not is_published(relative, patterns) or is_ignored(relative, rules):
            continue
        files.append(relative)
    return files


def stage(root: Path, out: Path, files: list[str]) -> list[str]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    staged = []
    for relative in files:
        # 公開側の入口は README.md でなければ GitHub が表示しない。
        target = "README.md" if relative == PUBLIC_README else relative
        destination = out / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
        staged.append(target)
    shutil.copy2(root / ".gitignore", out / ".gitignore")
    staged.append(".gitignore")
    return staged


def rescan(out: Path) -> dict[str, list[dict]]:
    """複製後の実物をもう一度見る。**一覧を信じず、置かれたものを数える。**

    禁止語だけでなく**個人環境パスも見る**（C-35 はブロッキング。
    一覧側の検査と複製後の検査で対象が違うと、後者だけ素通りする）。
    """
    hits = {}
    for path in sorted(out.rglob("*")):
        if not path.is_file() or ".git/" in str(path.relative_to(out)):
            continue
        found = scan_terms(path) + [
            {**h, "term": "個人環境パス"} for h in scan_personal_paths(path)]
        if found:
            hits[str(path.relative_to(out))] = found
    return hits


def git(out: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=out, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--account", default=None, help="GitHub アカウント（U-08）")
    p.add_argument("--repo", default=REPO_NAME)
    args = p.parse_args()

    safety = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "check_publish_safety.py")],
        capture_output=True,
        text=True,
    )
    print(safety.stdout.strip())
    if safety.returncode != 0:
        print("\n✘ 公開安全確認が通っていないので組み立てない。")
        return 1

    out = Path(args.out)
    files = collect(ROOT)
    staged = stage(ROOT, out, files)

    hits = rescan(out)
    if hits:
        print("\n✘ 複製後の再走査で禁止語を検出した。")
        for relative, found in hits.items():
            print(f"  ✘ {relative}:{found[0]['line']} [{found[0]['term']}]")
        return 1

    git(out, "init", "-q", "-b", "main")
    git(out, "add", "-A")
    git(out, "-c", "user.name=pan", "-c", "user.email=pan@example.com",
        "commit", "-q", "-m", "Physical AI 最終課題: 言語指示で卓上を仕分けるロボット")
    tracked = git(out, "-c", "core.quotepath=false", "ls-files").splitlines()

    total_mb = sum((out / f).stat().st_size for f in tracked) / 1e6
    print(f"\n=== 公開リポジトリを組み立てた: {out} ===")
    print(f"複製したファイル: {len(staged)} 件 / git 追跡: {len(tracked)} 件 / 合計 {total_mb:.2f} MB")
    if len(staged) != len(tracked):
        print("✘ 複製数と追跡数が一致しない（.gitignore が公開側で効いている可能性）")
        missing = sorted(set(staged) - set(tracked))
        for relative in missing[:10]:
            print(f"  - 未追跡: {relative}")
        return 1
    print("README.md:", "あり" if "README.md" in tracked else "✘ 無い")

    account = args.account or "<GitHubアカウント>"
    print("\n--- push する手順（人の確認後に実行する） ---")
    print(f"cd {out}")
    # このスクリプトは出力先を作り直す（rmtree）。∴ 再実行すると origin も履歴も消える。
    print(f"git remote add origin https://github.com/{account}/{args.repo}.git"
          "   # 再実行した直後だけ必要")
    print("git push -u origin main")
    print(f"\n※ SSH を使う場合: git remote set-url origin "
          f"git@github.com:{account}/{args.repo}.git")
    print("※ push 後、GitHub の Settings → General → Danger Zone で "
          "**Public へ変更**すること（要項10/8版で必須）。")
    if args.account is None:
        print("\n※ U-08（GitHub アカウント）が未確定のため push しない。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
