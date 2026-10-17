#!/usr/bin/env python
"""提出物に載る数値が、台帳に無いものになっていないか見る。

**2026-09-29、提出動画のまとめスライドが `10/15`、直前のグラフが `41/60` だった**
——同じ動画が自分と矛盾していた（R-28）。機械チェックは全項目 OK のままで、
**目で見るまで誰も気づかなかった。**

原因は「同じ事実を複数の場所に手書きしていた」こと。図と動画の焼き込みは
実測 JSON から生成するようにしたので（DEC-28 / DEC-40）、**手書きが残っているのは
レポート原稿・動画台本・公開用 README の3つ**である。∴ **そこを台帳と突き合わせる。**
（README_public.md は 2026-09-29 のレビューで「手書き数値が6行あるのに未検査」と
分かって対象へ足した。）

判定は「台帳に載っているか」だけ。**値が正しいかどうかは決めない**——
正しさの根拠は台帳に書いた出典であり、それは人が確かめる。

使い方:
    python scripts/check_numbers.py
    python scripts/check_numbers.py --pdf submission/xxx.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "configs" / "numbers_ledger.txt"
TARGETS = [
    Path("docs/40_報告/レポート原稿.md"),
    Path("docs/40_報告/動画台本.md"),
    Path("README_public.md"),
]

FRACTION = re.compile(r"\d+\s*/\s*\d+")
PERCENT = re.compile(r"\d+(?:\.\d+)?\s*[%％]")
# 年月日は指標ではないが、**行ごと免除にはしない**——「2026/09/27 に測定: 41/60」の
# 41/60 まで素通りする（旧実装の穴。2026-09-29 レビューで検出）。
# 日付の形（YYYY/MM/DD・YYYY/MM・MM/DD/YYYY・MM/YYYY）だけを行から取り除いてから走査する。
DATE_LIKE = re.compile(
    r"\d{4}\s*/\s*\d{1,2}(?:\s*/\s*\d{1,2})?"   # 2026/09/27, 2026/09
    r"|\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}"      # 9/27/2026
    r"|\d{1,2}\s*/\s*\d{4}")                     # 9/2026


def normalise(token: str) -> str:
    return token.replace(" ", "").replace("％", "%")


def load_ledger(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"台帳が無い: {path}")
    entries = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("::")]
        if len(parts) != 3:
            raise SystemExit(f"台帳の書式が違う（3項目必要）: {raw}")
        entries[normalise(parts[0])] = parts[2]
    return entries


def scan(text: str) -> list[tuple[int, str]]:
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        cleaned = DATE_LIKE.sub(" ", line)
        for match in FRACTION.findall(cleaned) + PERCENT.findall(cleaned):
            found.append((number, normalise(match)))
    return found


def pdf_text(path: Path) -> str:
    import pymupdf

    doc = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", default=None, help="提出 PDF も対象に含める")
    args = p.parse_args()

    ledger = load_ledger(LEDGER)
    sources: list[tuple[str, str]] = []
    for relative in TARGETS:
        path = ROOT / relative
        if not path.exists():
            print(f"✘ 対象が無い: {relative}")
            return 1
        sources.append((str(relative), path.read_text(errors="ignore")))
    if args.pdf:
        sources.append((args.pdf, pdf_text(Path(args.pdf))))

    unknown: list[tuple[str, int, str]] = []
    used: set[str] = set()
    for name, text in sources:
        for line_number, token in scan(text):
            if token in ledger:
                used.add(token)
            else:
                unknown.append((name, line_number, token))

    print(f"台帳の登録数: {len(ledger)} ／ 走査した文書: {len(sources)} 件")
    print(f"台帳に載っていない数値: {len(unknown)} 件")
    for name, line_number, token in unknown:
        print(f"  ✘ {name}:{line_number}  {token}")

    unused = sorted(set(ledger) - used)
    if unused:
        # 使われていない行は、消し忘れか、書き忘れのどちらかを示す。
        print(f"\n台帳にあるが提出物に出てこない数値: {len(unused)} 件（参考）")
        for token in unused:
            print(f"  - {token}（{ledger[token]}）")

    print(f"\n判定: {'不合格' if unknown else '合格'}")
    return 1 if unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())
