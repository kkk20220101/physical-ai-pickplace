"""提出物を機械で検査する。

[docs/01_課題要項.md §10] の12項目のうち、**機械で判定できるものだけ**を判定する。
判定できない項目は「人の目視」と出す。勝手に OK にしない——
「たぶん大丈夫」で通した項目は、通していないのと同じである。
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

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from check_publish_safety import FORBIDDEN_TERMS  # noqa: E402  禁止語を二重管理しない
except ImportError:
    # 禁止語の一覧を持つ検査器は社内限りで、公開リポジトリには入っていない（DEC-34）。
    # ここで落とすと**公開リポジトリでこのスクリプト自体が動かなくなる**ので、
    # 一覧が無い環境では項目10を「人の目視」へ倒す（黙って OK にはしない）。
    FORBIDDEN_TERMS = None

ROOT = Path(__file__).resolve().parents[1]
# 他スクリプト（build_review_pack.py等）と同じ規約: PATH にあればそれを使う。
# これが無いと、公開リポジトリを他人の環境で実行した瞬間にこの固定パスで落ちる（DEC-51 で実証）。
FFPROBE = shutil.which("ffprobe") or "ffprobe"

VIDEO_RE = re.compile(r"^動画_PAI最終課題_(.+)\.mp4$")
PDF_RE = re.compile(r"^説明資料_PAI最終課題_(.+)\.pdf$")
ZIP_RE = re.compile(r"^提出_PAI最終課題_(.+)\.zip$")

PLACEHOLDER = "PLACEHOLDER"
MAX_ZIP_MB = 200
MIN_SEC, MAX_SEC = 60, 300
REQUIRED_SECTIONS = ["概要", "目的", "工夫した点"]


class Result:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, no: int, item: str, status: str, detail: str) -> None:
        self.rows.append({"no": no, "item": item, "status": status, "detail": detail})

    @property
    def failed(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == "NG"]

    @property
    def manual(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == "人の目視"]


def video_duration(path: Path) -> float:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def mp4_box_order(path: Path) -> list[str]:
    """MP4 の先頭ボックスを並び順に返す。`moov` が前方なら先頭から再生できる。"""
    head = path.read_bytes()[:4096]
    order, offset = [], 0
    while offset < len(head) - 8 and len(order) < 6:
        size = int.from_bytes(head[offset:offset + 4], "big")
        order.append(head[offset + 4:offset + 8].decode("latin1", "replace"))
        if size < 8:
            break
        offset += size
    return order


def check(zip_path: Path) -> Result:
    r = Result()
    acct_from_zip = ZIP_RE.match(zip_path.name)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            zf.extractall(tmp)

        videos = [n for n in names if VIDEO_RE.match(n)]
        pdfs = [n for n in names if PDF_RE.match(n)]

        # 1: 中身が映っているかは見られない
        r.add(1, "ロボットがタスクを実行している様子が映っている", "人の目視",
              "映像の内容は機械では判定しない")

        # 2: 尺
        if videos:
            sec = video_duration(tmp / videos[0])
            ok = MIN_SEC <= sec <= MAX_SEC
            r.add(2, f"動画の長さが{MIN_SEC}〜{MAX_SEC}秒", "OK" if ok else "NG",
                  f"{sec:.1f} 秒")
        else:
            r.add(2, "動画の長さ", "NG", "動画が Zip に無い")

        # 3: 倍速表記（動画内に焼き込んでいるため機械では読めない）
        r.add(3, "倍速の旨を明記", "人の目視",
              "2倍速区間にバッジを焼き込んである。目で確認する")

        # 4/5: ファイル名
        r.add(4, "動画が MP4 で規定のファイル名", "OK" if len(videos) == 1 else "NG",
              videos[0] if videos else f"該当なし（中身: {names}）")
        r.add(5, "レポートが PDF で規定のファイル名", "OK" if len(pdfs) == 1 else "NG",
              pdfs[0] if pdfs else f"該当なし（中身: {names}）")

        # 6: レポートの節
        if pdfs:
            doc = pymupdf.open(tmp / pdfs[0])
            text = re.sub(r"\s+", "", "".join(p.get_text() for p in doc))
            pages = len(doc)
            doc.close()
            missing = [s for s in REQUIRED_SECTIONS if s not in text]
            r.add(6, "レポートに概要・目的・工夫した点がある",
                  "OK" if not missing else "NG",
                  f"{pages}ページ" + (f" / 欠落: {missing}" if missing else " / 3節とも有"))

            if FORBIDDEN_TERMS is None:
                r.add(10, "公開できないデータを含んでいない", "人の目視",
                      "禁止語一覧（社内限り・DEC-34）が無い環境のため機械判定不可")
            else:
                hits = [t for t in FORBIDDEN_TERMS if t.lower() in text.lower()]
                r.add(10, "公開できないデータを含んでいない", "OK" if not hits else "NG",
                      "PDF 本文に禁止語なし" if not hits else f"禁止語: {hits}")
        else:
            r.add(6, "レポートの必須節", "NG", "PDF が Zip に無い")
            r.add(10, "公開できないデータを含んでいない", "NG", "PDF が Zip に無い")

        # 7: 容量
        mb = zip_path.stat().st_size / 1e6
        r.add(7, f"Zip が {MAX_ZIP_MB}MB 以下", "OK" if mb <= MAX_ZIP_MB else "NG",
              f"{mb:.2f} MB")

        # 余計なものが入っていないか（要項にはないが事故防止）
        extra = [n for n in names if n not in videos + pdfs]
        r.add(0, "Zip の中身がちょうど2ファイル",
              "OK" if len(names) == 2 and not extra else "NG",
              f"{len(names)}件: {names}")

        # 索引（moov）の位置は先頭 4KB を読めば分かる。tmp が消える前に取る。
        box_order = mp4_box_order(tmp / videos[0]) if videos else []

    r.add(8, "AI を使用している", "人の目視",
          "OWLv2 と Qwen2.5-1.5B を推論で使用。レポート §3・§4 に記載")
    r.add(9, "GitHub にコード一式があり URL を出せる", "人の目視",
          "中身は公開可（check_publish_safety.py）。push は U-08 待ち")
    r.add(11, "剽窃・丸写し・無関係ファイルを含まない", "人の目視",
          "判断は人が行う")
    r.add(12, "締切より前に提出した", "人の目視", "提出操作は人が行う")

    # Zip 名・動画名・PDF 名のアカウント名が**互いに一致しているか**。
    # 名前だけ手で差し替えた Zip（動画は PLACEHOLDER のまま等）を素通りさせない。
    accounts = {m.group(1) for m in
                [VIDEO_RE.match(videos[0]) if videos else None,
                 PDF_RE.match(pdfs[0]) if pdfs else None,
                 acct_from_zip] if m}
    if len(accounts) > 1:
        r.add(15, "3ファイルのアカウント名が一致している", "NG",
              f"不一致: {sorted(accounts)}")

    # ファイル名にアカウント名が入っているか
    acct = acct_from_zip.group(1) if acct_from_zip else None
    if acct == PLACEHOLDER:
        r.add(13, "ファイル名が omnicampus アカウント名になっている", "NG",
              "PLACEHOLDER のまま（U-02 待ち）。--account を付けて再生成する")
    elif acct:
        r.add(13, "ファイル名が omnicampus アカウント名になっている", "OK", acct)
    else:
        r.add(13, "Zip のファイル名が規定形式", "NG", zip_path.name)

    # 索引（moov）が末尾にあると、全部落とすまで再生が始まらない。
    # **手元では一瞬で開くので気づけない**——相手がどう開くかは選べない。
    if box_order:
        early = "moov" in box_order[:3]
        r.add(14, "動画が先頭から再生できる（moov が前方）",
              "OK" if early else "NG",
              " → ".join(box_order[:3]) if early
              else "moov が後方。`-movflags +faststart` を付けて作り直す")

    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if not args.zip.exists():
        print(f"Zip が無い: {args.zip}", file=sys.stderr)
        return 2

    r = check(args.zip)
    rows = sorted(r.rows, key=lambda x: (x["no"] == 0, x["no"]))

    print(f"対象: {args.zip}\n")
    print(f"{'#':>3}  {'判定':<6} 項目 / 実測")
    print("-" * 88)
    for row in rows:
        no = "-" if row["no"] == 0 else str(row["no"])
        print(f"{no:>3}  {row['status']:<6} {row['item']}")
        print(f"{'':>3}  {'':<6}   → {row['detail']}")

    print("-" * 88)
    print(f"NG: {len(r.failed)} 件 / 人の目視: {len(r.manual)} 件 / "
          f"機械OK: {len(rows) - len(r.failed) - len(r.manual)} 件")
    if r.failed:
        print("\n判定: **提出不可**")
        for row in r.failed:
            print(f"  - {row['item']}: {row['detail']}")
    else:
        print("\n判定: **機械で判定できる項目はすべて通過**（残りは人の目視）")

    if args.json:
        args.json.write_text(json.dumps(
            {"zip": str(args.zip), "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n保存: {args.json}")

    return 1 if r.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
