#!/usr/bin/env python
"""提出前の「人の目視」6項目を、人が1画面で判定できる材料に変える。

**機械が判定してはいけない項目でも、材料を揃えることはできる**（C-23）。
2026-09-29 に、焼き込み文字が拡大でにじんで読めない不具合を**目視で初めて発見した**。
機械チェックは全項目 OK のままだった——**誰も動画を見ていなかったからである**（R-28）。

出力は `outputs/review_pack/`（公開対象外）。**判定は書かない。材料だけ置く。**

使い方:
    python scripts/build_review_pack.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
# パス文字列の置換で導出しない（パスに "ffmpeg" が2回現れる環境で壊れる）
FFPROBE = shutil.which("ffprobe") or "ffprobe"
DEFAULT_VIDEO = ROOT / "submission" / "提出動画_未署名.mp4"
DEFAULT_OUT = ROOT / "outputs" / "review_pack"
N_FRAMES = 12
COLUMNS = 3
# 「AI を使っている」と読者が判断する根拠になる語。存在の有無だけを数える。
AI_EVIDENCE = ["OWLv2", "Qwen", "言語モデル", "物体検出", "推論"]
# 倍速バッジは右上に黄色で焼き込む（`build_submission_video.py` の drawtext）。
# 描画は `speed != 1.0` のときだけなので、**バッジの有無＝倍速区間の有無**である。
BADGE_MIN_PIXELS = 150


def duration(video: Path) -> float:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def extract(video: Path, out: Path, seconds: list[float]) -> list[Path]:
    paths = []
    for index, t in enumerate(seconds):
        path = out / f"{index:02d}_{int(t):04d}s.png"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{t:.2f}",
                        "-i", str(video), "-frames:v", "1", str(path)], check=True)
        paths.append(path)
    return paths


def contact_sheet(frames: list[Path], seconds: list[float], out: Path) -> Path:
    from PIL import Image, ImageDraw

    thumbs = [Image.open(p).convert("RGB") for p in frames]
    tw, th = thumbs[0].width // 3, thumbs[0].height // 3
    rows = (len(thumbs) + COLUMNS - 1) // COLUMNS
    label = 26
    sheet = Image.new("RGB", (COLUMNS * tw, rows * (th + label)), (20, 20, 24))
    draw = ImageDraw.Draw(sheet)
    for index, (thumb, t) in enumerate(zip(thumbs, seconds)):
        x, y = (index % COLUMNS) * tw, (index // COLUMNS) * (th + label)
        sheet.paste(thumb.resize((tw, th), Image.LANCZOS), (x, y + label))
        draw.text((x + 6, y + 6), f"{int(t // 60)}:{int(t % 60):02d}", fill=(255, 220, 120))
    path = out / "contact_sheet.png"
    sheet.save(path)
    return path


def pdf_evidence(pdf: Path) -> dict[str, int]:
    import pymupdf

    with pymupdf.open(pdf) as doc:
        text = "".join(page.get_text() for page in doc)
    return {term: text.count(term) for term in AI_EVIDENCE}


def pdf_sheet(pdf: Path, out: Path) -> tuple[Path, int]:
    """PDF を全ページ画像化して1枚に並べる。**表の罫線と改ページは見ないと分からない。**"""
    import pymupdf
    from PIL import Image

    folder = out / "pdf"
    folder.mkdir(parents=True, exist_ok=True)
    pages = []
    with pymupdf.open(pdf) as doc:
        for index, page in enumerate(doc):
            path = folder / f"p{index + 1:02d}.png"
            page.get_pixmap(dpi=110).save(path)
            pages.append(Image.open(path).convert("RGB"))
    width, height = pages[0].size
    rows = (len(pages) + COLUMNS - 1) // COLUMNS
    sheet = Image.new("RGB", (COLUMNS * width, rows * height), (255, 255, 255))
    for index, page in enumerate(pages):
        sheet.paste(page, ((index % COLUMNS) * width, (index // COLUMNS) * height))
    sheet.thumbnail((2200, 2200), Image.LANCZOS)
    path = folder / "pdf_sheet.png"
    sheet.save(path)
    return path, len(pages)


def speed_badge_ranges(video: Path, total: float, out: Path) -> list[tuple[float, float]]:
    """黄色バッジが出ている区間を 1 秒刻みで調べる。

    「倍速の焼き込みがコードにある」では材料にならない。
    **出来上がった映像のどこに出ているか**まで示して初めて人が1分で判定できる。
    """
    import numpy as np
    from PIL import Image

    present: list[float] = []
    frame = out / "_probe.png"
    for step in range(int(total)):
        subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{step}",
                        "-i", str(video), "-frames:v", "1", str(frame)], check=True)
        array = np.asarray(Image.open(frame).convert("RGB")).astype(int)
        roi = array[10:90, int(array.shape[1] * 0.60):]
        yellow = ((roi[:, :, 0] > 170) & (roi[:, :, 1] > 170) & (roi[:, :, 2] < 110)).sum()
        if yellow > BADGE_MIN_PIXELS:
            present.append(float(step))
    frame.unlink(missing_ok=True)

    ranges: list[list[float]] = []
    for t in present:
        if ranges and t - ranges[-1][1] <= 1.5:
            ranges[-1][1] = t
        else:
            ranges.append([t, t])
    return [(a, b + 1) for a, b in ranges]


def borrowed_sources() -> list[str]:
    """他人のコードを貼っていないことの材料。**機械が見られるのは著作権表示の有無まで。**"""
    marks = []
    for folder in ("src", "scripts"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            head = path.read_text(errors="ignore")[:2000]
            if "Copyright" in head or "SPDX-License" in head:
                marks.append(str(path.relative_to(ROOT)))
    return marks


def deadline() -> str | None:
    # 要項は公開対象外（社内限り）なので、公開リポジトリ側で動かしたときは黙って諦める。
    source = ROOT / "docs" / "01_課題要項.md"
    if not source.exists():
        return None
    text = source.read_text(errors="ignore")
    found = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日.*?(?:AM|午前)?\s*(\d{1,2}):(\d{2})", text)
    if not found:
        return None
    year, month, day, hour, minute = (int(v) for v in found.groups())
    target = datetime(year, month, day, hour, minute)
    return f"{target:%Y-%m-%d %H:%M}（残り {(target - datetime.now()).days} 日）"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", default=None, help="省略時は submission/ の動画を探す")
    p.add_argument("--pdf", default=None, help="省略時は submission/ の PDF を探す")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    # アカウント名を入れた本番版があればそれを見る。無ければ未署名版へ落とす。
    # **目視の材料は、実際に提出する物から作らなければ意味がない**（DEC-46）。
    if args.video:
        video = Path(args.video)
    else:
        video = next(iter(sorted((ROOT / "submission").glob("動画_PAI最終課題_*.mp4"))),
                     DEFAULT_VIDEO)
    out = Path(args.out)
    if not video.exists():
        raise SystemExit(f"動画が無い: {video}")
    pdf = Path(args.pdf) if args.pdf else next(
        iter(sorted((ROOT / "submission").glob("説明資料_*.pdf"))), None)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    total = duration(video)
    # 末尾は「まとめ」で、主張の要約が出る最も重要なコマである。等間隔だと届かないので必ず入れる。
    seconds = [total * (i + 0.5) / N_FRAMES for i in range(N_FRAMES - 1)] + [total - 1.0]
    frames = extract(video, out, seconds)
    sheet = contact_sheet(frames, seconds, out)

    evidence = pdf_evidence(pdf) if pdf else {}
    pages_sheet, pages = pdf_sheet(pdf, out) if pdf else (None, 0)
    badges = speed_badge_ranges(video, total, out)
    borrowed = borrowed_sources()
    limit = deadline()

    badge_text = "／".join(f"{a:.0f}〜{b:.0f}秒" for a, b in badges) or "**検出なし**"
    covered = sum(b - a for a, b in badges)

    lines = [
        "# 目視確認シート（材料のみ。判定は人が書く）",
        "",
        f"- 動画: `{video.name}` / {total:.1f} 秒 / {video.stat().st_size / 1e6:.2f} MB",
        f"- PDF: `{pdf.name if pdf else '(無し)'}` / {pages} ページ",
        f"- コンタクトシート（動画）: `{sheet.relative_to(ROOT)}`（{N_FRAMES}コマ・末尾を含む）",
        f"- ページシート（PDF）: `{pages_sheet.relative_to(ROOT) if pages_sheet else '(無し)'}`",
        "",
        "| # | 確認項目 | 揃えた材料 | 判定（人が記入） |",
        "|:-:|---|---|:-:|",
        f"| 1 | 映像の内容が伝わるか | コンタクトシート {N_FRAMES}コマ（**末尾のまとめを含む**）。焼き込み文字が読めるかを見る | |",
        f"| 2 | 倍速の倍率が画面に出ているか | **黄色バッジを検出した区間: {badge_text}**（全体 {total:.0f} 秒のうち {covered:.0f} 秒）。"
        "バッジは `speed != 1.0` のときだけ描かれるので、**検出区間＝倍速区間**である | |",
        f"| 3 | AI を使っていることが分かるか | PDF 本文の語の出現数（下表・計 {sum(evidence.values())} 箇所） | |",
        "| 4 | GitHub の URL を出せるか | **U-08 待ち**（中身は組み立て済み・commit 済み） | |",
        f"| 5 | 剽窃・無関係ファイルが無いか | Zip の中身はちょうど2ファイル（機械確認済み）。"
        f"`src/`・`scripts/` に他者の著作権表示を持つファイル **{len(borrowed)} 件**"
        f"{'（' + '／'.join(borrowed) + '）' if borrowed else ''}。図は実測 JSON から生成 | |",
        f"| 6 | 締切前に提出したか | 締切 **{limit or '（要項から読めず）'}**。提出操作そのものは人が行う | |",
        "",
        "## PDF に出てくる語（項目3の材料）",
        "",
        "| 語 | 出現数 |",
        "|---|---:|",
    ]
    lines += [f"| {term} | {count} |" for term, count in evidence.items()]
    lines += [
        "",
        "> **この表の「判定」列を機械が埋めてはならない**（C-23）。",
        "> 機械は**見えるものを数えるところまで**を行う。**見て納得したかどうかは人にしか書けない。**",
    ]

    sheet_md = out / "確認シート.md"
    sheet_md.write_text("\n".join(lines) + "\n")

    print(f"目視確認の材料: {out}")
    print(f"  コンタクトシート: {sheet.name}（{N_FRAMES}コマ・末尾込み）")
    print(f"  ページシート: {pages_sheet.name if pages_sheet else '(無し)'}（{pages}ページ）")
    print(f"  倍速バッジ: {badge_text}")
    print(f"  確認シート: {sheet_md.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
