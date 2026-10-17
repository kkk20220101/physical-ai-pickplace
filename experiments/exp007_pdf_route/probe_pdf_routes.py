"""PDF 変換経路の実測比較（H-A の前段）。

この環境には pandoc も weasyprint も Chromium も無い。使えるのは
PyMuPDF(Story) と LibreOffice の2経路だけなので、その2つを同じ原稿で比べる。

合否は目視でなく機械で決める:
  1. 日本語が「文字として」入っているか（抽出テキストに期待語が出るか）
  2. 日本語スパンが CJK フォントで描かれているか（豆腐＝非CJKフォントへの代替）
  3. 図が6枚とも埋め込まれているか
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import markdown
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
SRC_MD = ROOT / "docs" / "40_報告" / "レポート原稿.md"
FIG_DIR = ROOT / "docs" / "40_報告" / "figures"
OUT_DIR = Path(__file__).resolve().parent / "out"

CJK_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
CJK_FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
FONT_CACHE = Path(__file__).resolve().parent / "fonts"

# 抽出テキストに必ず現れるべき語（原稿から取った実在の文字列）
EXPECT_WORDS = ["システム構成", "アブレーション", "言語モデル", "失敗モード"]


def ensure_otf(ttc: Path, index: int, out: Path) -> Path:
    """MuPDF の CSS フォント読み込みは .ttc を扱えないので単体 .otf へ取り出す。"""
    if out.exists():
        return out
    from fontTools.ttLib import TTCollection

    out.parent.mkdir(parents=True, exist_ok=True)
    with TTCollection(str(ttc), lazy=True) as tc:
        tc.fonts[index].save(str(out))
    return out


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x3040 <= o <= 0x30FF  # かな
        or 0x4E00 <= o <= 0x9FFF  # 漢字
        or 0xFF00 <= o <= 0xFFEF  # 全角英数
    )


def md_to_html(md_path: Path, css: str) -> str:
    body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    return f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"


STORY_CSS = """
@font-face {{ font-family: jp; src: url({reg}); }}
@font-face {{ font-family: jp; font-weight: bold; src: url({bold}); }}
* {{ font-family: jp; }}
body {{ font-size: 10px; }}
h1 {{ font-size: 18px; }}
h2 {{ font-size: 14px; }}
h3 {{ font-size: 11px; }}
table {{ width: 100%; }}
td, th {{ border: 1px solid #999; padding: 2px; font-size: 8px; }}
img {{ width: 460px; }}
"""

LO_CSS = """
body { font-family: 'Noto Sans CJK JP'; font-size: 10pt; }
table { border-collapse: collapse; width: 100%; }
td, th { border: 1px solid #999; padding: 2px; font-size: 8pt; }
img { width: 16cm; }
"""


def route_pymupdf(out_pdf: Path) -> dict:
    """markdown -> HTML -> PyMuPDF Story -> PDF"""
    reg = ensure_otf(CJK_FONT, 0, FONT_CACHE / "NotoSansCJKjp-Regular.otf")
    bold = ensure_otf(CJK_FONT_BOLD, 0, FONT_CACHE / "NotoSansCJKjp-Bold.otf")
    css = STORY_CSS.format(reg=reg.name, bold=bold.name)
    html = md_to_html(SRC_MD, css)
    arch = pymupdf.Archive()
    arch.add(FIG_DIR, "figures")
    arch.add(FONT_CACHE)

    story = pymupdf.Story(html=html, archive=arch)
    writer = pymupdf.DocumentWriter(out_pdf)
    page_rect = pymupdf.paper_rect("a4")
    area = page_rect + (40, 40, -40, -40)

    pages = 0
    more = True
    while more:
        dev = writer.begin_page(page_rect)
        more, _ = story.place(area)
        story.draw(dev)
        writer.end_page()
        pages += 1
        if pages > 60:
            raise RuntimeError("ページが増え続けている（レイアウト破綻）")
    writer.close()
    return {"pages_written": pages}


def route_libreoffice(out_pdf: Path) -> dict:
    """markdown -> HTML -> LibreOffice -> PDF"""
    html = md_to_html(SRC_MD, LO_CSS)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        shutil.copytree(FIG_DIR, tmp / "figures")
        src_html = tmp / "report.html"
        src_html.write_text(html, encoding="utf-8")
        cmd = [
            "libreoffice", "--headless", "--norestore",
            f"-env:UserInstallation=file://{tmp}/loprofile",
            "--convert-to", "pdf:writer_web_pdf_Export",
            "--outdir", str(tmp), str(src_html),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        produced = tmp / "report.pdf"
        if not produced.exists():
            raise RuntimeError(f"LibreOffice が PDF を作らなかった: {proc.stdout}{proc.stderr}")
        shutil.copy(produced, out_pdf)
    return {"stdout": proc.stdout.strip()[:200]}


def inspect(pdf_path: Path) -> dict:
    doc = pymupdf.open(pdf_path)
    text = "".join(p.get_text() for p in doc)

    cjk_fonts: dict[str, int] = {}
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if any(is_cjk(c) for c in span["text"]):
                        cjk_fonts[span["font"]] = cjk_fonts.get(span["font"], 0) + 1

    images = sum(len(p.get_images(full=True)) for p in doc)
    missing = [w for w in EXPECT_WORDS if w not in text]
    pages = len(doc)
    doc.close()

    return {
        "pages": pages,
        "size_kb": round(pdf_path.stat().st_size / 1024, 1),
        "cjk_chars_extracted": sum(1 for c in text if is_cjk(c)),
        "cjk_span_fonts": cjk_fonts,
        "images_embedded": images,
        "expect_words_missing": missing,
    }


def verdict(info: dict) -> tuple[bool, list[str]]:
    reasons = []
    if info["cjk_chars_extracted"] < 500:
        reasons.append(f"日本語の抽出が少なすぎる({info['cjk_chars_extracted']}字)")
    if info["expect_words_missing"]:
        reasons.append(f"期待語が出ない: {info['expect_words_missing']}")
    if not info["cjk_span_fonts"]:
        reasons.append("日本語スパンが1つも無い")
    else:
        bad = [f for f in info["cjk_span_fonts"] if not any(
            k in f for k in ("Noto", "CJK", "Gothic", "Mincho", "IPA", "Droid", "Sans-JP"))]
        if bad:
            reasons.append(f"日本語が非CJKフォントで描かれている(豆腐の疑い): {bad}")
    if info["images_embedded"] < 6:
        reasons.append(f"図が足りない({info['images_embedded']}/6)")
    return (not reasons), reasons


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CJK_FONT.exists():
        print(f"日本語フォントが無い: {CJK_FONT}", file=sys.stderr)
        return 2

    routes = {"pymupdf_story": route_pymupdf, "libreoffice": route_libreoffice}
    results = {}
    for name, fn in routes.items():
        out_pdf = OUT_DIR / f"{name}.pdf"
        t0 = time.time()
        try:
            extra = fn(out_pdf)
            info = inspect(out_pdf)
            ok, reasons = verdict(info)
            results[name] = {
                "ok": ok, "reasons": reasons, "seconds": round(time.time() - t0, 1),
                **info, **extra,
            }
        except Exception as exc:  # 経路ごとの失敗は比較結果の一部
            results[name] = {"ok": False, "reasons": [f"{type(exc).__name__}: {exc}"],
                             "seconds": round(time.time() - t0, 1)}
        print(f"\n=== {name} ===")
        print(json.dumps(results[name], ensure_ascii=False, indent=2))

    (OUT_DIR / "result.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n保存: {OUT_DIR / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
