"""提出パッケージ（動画 + 説明資料PDF + Zip）を組み立てる。

PDF 変換は LibreOffice 経路。選定理由は
docs/30_評価/20260928_PDF変換経路の選定と提出パッケージ_結果.md
（実測で容量 72分の1。現存する一次情報 experiments/exp007_pdf_route/out/result.json）。

使い方:
    python scripts/build_submission_package.py                 # PLACEHOLDER 名で組む
    python scripts/build_submission_package.py --account xxxx  # 提出名で組む
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import markdown
import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "docs" / "40_報告" / "レポート原稿.md"
FIG_DIR = ROOT / "docs" / "40_報告" / "figures"
DEFAULT_VIDEO = ROOT / "submission" / "提出動画_未署名.mp4"
OUT_DIR = ROOT / "submission"

PLACEHOLDER = "PLACEHOLDER"
# 提出名にそのまま差し込む唯一の外部入力なので、ここだけは検査する（DEC-48）。
# 空白・`/`・非ASCII は **Zip 内の名前が相手の環境で化ける**。
# Windows で使えない文字（\ / : * ? " < > | と制御文字）もこの正規表現の外なので同時に落ちる。
ACCOUNT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SUBMIT_TITLE = "指示文から物体を仕分けるロボット"
SUBMIT_SUBTITLE = "Physical AI 最終課題 説明資料"

# LibreOffice の Writer/Web は CSS の img{width} も HTML の width 属性も無視し、
# PNG 自身の寸法（px と DPI）で置く。よって画像側をリサイズするしかない。
IMG_WIDTH_PX = 620
IMG_DPI = 96

CSS = """
@page { size: A4; margin: 18mm 15mm; }
body { font-family: 'Noto Sans CJK JP'; font-size: 10pt; line-height: 1.55; }
h1 { font-size: 17pt; margin-bottom: 2pt; }
h1 + p.subtitle { font-size: 10pt; color: #444; margin-top: 0; }
h2 { font-size: 12.5pt; border-bottom: 1px solid #666; padding-bottom: 2px; margin-top: 16pt; }
h3 { font-size: 10.5pt; margin-top: 11pt; }
p, li { font-size: 10pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; }
th, td { border: 1px solid #999; padding: 3px 4px; vertical-align: top; }
th { background-color: #eeeeee; }
img { width: 15cm; }
pre { background-color: #f6f6f6; border: 1px solid #dddddd; padding: 4px; font-size: 8pt; }
code { font-family: 'Noto Sans Mono CJK JP'; font-size: 8.5pt; }
blockquote { border-left: 3px solid #bbb; margin-left: 0; padding-left: 8px; color: #333; }
"""


def strip_internal_front_matter(md_text: str) -> str:
    """原稿の先頭にある社内向けメタ情報（最終更新・状態・数値の正）を落とす。

    提出物に「PDF化が未」などの進行状況が残るのを防ぐ。
    """
    lines = md_text.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("原稿の1行目が H1 ではない。書式が変わった可能性がある")

    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = lines[idx + 1:]
            break
    else:
        raise ValueError("先頭のメタ情報ブロックを終える '---' が見つからない")

    head = [f"# {SUBMIT_TITLE}", "", f'<p class="subtitle">{SUBMIT_SUBTITLE}</p>', ""]
    return "\n".join(head + body)


def md_to_html(md_text: str) -> str:
    body = markdown.markdown(
        md_text, extensions=["tables", "fenced_code", "sane_lists", "md_in_html"]
    )
    # LibreOffice は表の CSS（罫線・幅）を無視し、HTML 属性だけを見る（DEC-44）。
    # 幅が効かないと列が狭くなり、`Bread` が `Brea`/`d` に分断される。
    body = body.replace(
        "<table>", '<table border="1" cellspacing="0" cellpadding="3" width="100%">'
    )
    return (
        "<html><head><meta charset='utf-8'>"
        f"<title>{SUBMIT_TITLE}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def stage_figures(dest: Path) -> list[str]:
    """紙面に収まる幅へ図を縮めて置く。"""
    dest.mkdir(parents=True, exist_ok=True)
    staged = []
    for png in sorted(FIG_DIR.glob("*.png")):
        with Image.open(png) as im:
            scale = IMG_WIDTH_PX / im.width
            size = (IMG_WIDTH_PX, max(1, round(im.height * scale)))
            im.convert("RGB").resize(size, Image.LANCZOS).save(
                dest / png.name, dpi=(IMG_DPI, IMG_DPI))
        staged.append(f"{png.name} -> {size[0]}x{size[1]}")
    return staged


def html_to_pdf(html: str, out_pdf: Path) -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        stage_figures(tmp / "figures")
        src = tmp / "report.html"
        src.write_text(html, encoding="utf-8")
        cmd = [
            "libreoffice", "--headless", "--norestore",
            f"-env:UserInstallation=file://{tmp}/loprofile",
            "--convert-to", "pdf:writer_web_pdf_Export",
            "--outdir", str(tmp), str(src),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        produced = tmp / "report.pdf"
        if not produced.exists():
            raise RuntimeError(
                "LibreOffice が PDF を作らなかった。\n"
                f"stdout={proc.stdout}\nstderr={proc.stderr}"
            )
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(produced, out_pdf)


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF


def verify_pdf(pdf: Path) -> dict:
    """日本語が文字として入っているか・図が入っているかを機械で確かめる。"""
    doc = pymupdf.open(pdf)
    text = "".join(p.get_text() for p in doc)
    flat = re.sub(r"\s+", "", text)

    cjk_fonts: dict[str, int] = {}
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if any(is_cjk(c) for c in span["text"]):
                        cjk_fonts[span["font"]] = cjk_fonts.get(span["font"], 0) + 1

    xrefs = {img[0] for page in doc for img in page.get_images(full=True)}

    # フォントの同梱状況。**`ext` は無いとき文字列 `"n/a"` になる**（空ではない）。
    # Type3 は字形の描画手順を PDF 内に持つので**相手の環境でも出る**が、
    # それ以外で実体が無いものは**相手のフォントで代用される**（DEC-49）。
    missing_fonts, type3_fonts = [], []
    for page in doc:
        for xref, ext, kind, basefont, *_ in page.get_fonts(full=True):
            if ext not in ("", "n/a"):
                continue
            (type3_fonts if kind == "Type3" else missing_fonts).append(basefont)
    missing_fonts, type3_fonts = sorted(set(missing_fonts)), sorted(set(type3_fonts))

    # 「埋め込まれている」と「紙面に収まっている」は別。実際に右端が切れていた。
    overflow = []
    for page in doc:
        limit = page.rect + (-1, -1, 1, 1)
        for img in page.get_images(full=True):
            for rect in page.get_image_rects(img[0]):
                if not limit.contains(rect):
                    overflow.append({
                        "page": page.number + 1,
                        "rect": [round(v) for v in rect],
                        "page_rect": [round(v) for v in page.rect],
                    })

    info = {
        "pages": len(doc),
        "size_kb": round(pdf.stat().st_size / 1024, 1),
        "cjk_chars": sum(1 for c in flat if is_cjk(c)),
        "cjk_span_fonts": cjk_fonts,
        "unique_images": len(xrefs),
        "images_overflowing_page": overflow,
        # 罫線は CSS では出ない。出ている＝ HTML 属性が効いている証拠（DEC-44）。
        "vector_drawings": sum(len(page.get_drawings()) for page in doc),
        "internal_front_matter_removed": "最終更新" not in flat[:400],
        # 実体が無い＝相手のフォントで代用される。Type3 は自己完結なので別に数える。
        "fonts_missing": missing_fonts,
        "fonts_type3": type3_fonts,
    }
    doc.close()

    problems = []
    if info["cjk_chars"] < 500:
        problems.append(f"日本語が少なすぎる（{info['cjk_chars']}字）")
    bad = [f for f in cjk_fonts if not any(
        k in f for k in ("Noto", "CJK", "Gothic", "Mincho", "IPA", "Droid"))]
    if bad:
        problems.append(f"日本語が非CJKフォントで描かれている（豆腐の疑い）: {bad}")
    if info["unique_images"] < 6:
        problems.append(f"図が足りない（{info['unique_images']}/6）")
    if overflow:
        problems.append(f"図が紙面からはみ出ている（{len(overflow)}箇所）")
    if info["vector_drawings"] < 100:
        problems.append(
            f"表の罫線が描かれていない（描画要素 {info['vector_drawings']}）。"
            "CSS だけでは出ない＝HTML 属性が落ちている疑い"
        )
    if not info["internal_front_matter_removed"]:
        problems.append("社内向けメタ情報が残っている")
    if missing_fonts:
        problems.append(
            f"実体の無いフォントがある: {missing_fonts}。"
            "**手元では正しく見えても、相手の環境では別の字になる**"
        )
    if type3_fonts:
        problems.append(
            f"絵文字書体（Type3）が混ざっている: {type3_fonts}。"
            "**出はするが絵文字の見た目になる**。`✔` を `（合格）` 等の文字へ置き換える"
        )
    info["problems"] = problems
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", default=PLACEHOLDER,
                    help="omnicampus アカウント名（未確定なら省略）")
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    ap.add_argument("--outdir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if not args.video.exists():
        print(f"動画が無い: {args.video}", file=sys.stderr)
        return 2

    acct = args.account
    if not ACCOUNT_RE.match(acct):
        print(
            f"アカウント名が使えない: {acct!r}\n"
            "  英数字で始まり、英数字と . _ - だけで64文字以内にしてください。\n"
            "  空白や日本語を入れると、**Zip を解凍した環境でファイル名が化けます**。",
            file=sys.stderr,
        )
        return 2
    args.outdir.mkdir(parents=True, exist_ok=True)

    md_text = strip_internal_front_matter(REPORT_MD.read_text(encoding="utf-8"))
    pdf_path = args.outdir / f"説明資料_PAI最終課題_{acct}.pdf"
    print("PDF を生成中（LibreOffice 経路）…")
    html_to_pdf(md_to_html(md_text), pdf_path)

    info = verify_pdf(pdf_path)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if info["problems"]:
        print("\nPDF の検査に落ちた。Zip は作らない。", file=sys.stderr)
        return 1

    video_path = args.outdir / f"動画_PAI最終課題_{acct}.mp4"
    if video_path.resolve() != args.video.resolve():
        shutil.copy(args.video, video_path)

    zip_path = args.outdir / f"提出_PAI最終課題_{acct}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(video_path, video_path.name)
        zf.write(pdf_path, pdf_path.name)

    print("\n--- 生成物 ---")
    for p in (video_path, pdf_path, zip_path):
        # 要項の上限「200MB」は10進。MiB で出すと同じファイルが別の数字になる（DEC-41）。
        print(f"{p.name:44s} {p.stat().st_size / 1e6:7.2f} MB")
    if acct == PLACEHOLDER:
        print("\n※ --account 未指定。U-02 が決まったら --account を付けて再実行するだけでよい。")
    print(f"\n確認: python scripts/check_submission.py --zip {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
