#!/usr/bin/env python
"""提出用の動画を1本に組み立てる。

構成は docs/40_報告/動画台本.md に従う。
**素材の切り出し位置は run の `summary.json` の `timeline` から取る**（目分量で切らない）。

倍速を使う区間は**画面に倍率を焼き込む**（要件 R-04。過剰側に倒す）。

使い方:
    python scripts/build_submission_video.py \
        --run outputs/20260927_212018_c4_master
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import FAILURE_MODES  # noqa: E402
from src.planning.llm_planner import OBJECT_LABELS  # noqa: E402

FIGURES = ROOT / "docs" / "40_報告" / "figures"
JP_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
SIZE = (1280, 720)
FPS = 20

# 焦き込み文字の下地（DEC-38）が素材に入っているかを判定する境。
# 実測（2026-09-29）: 帯あり 0.51–0.67 ／ 帯なし 0.93–1.15。間を取る。
OVERLAY_BAND_RATIO = 0.85

TITLE_CARDS = [
    ("AI×ロボティクスによる創造的タスクの実現",
     "あいまいな日本語の指示で、ロボットが卓上を仕分ける",
     "計画・知覚・姿勢のすべてをAIが担当（真値は一切使わない）", 9),
]


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg 失敗:\n{' '.join(cmd)}\n{result.stderr[-1500:]}")
    # drawtext は `Stray %` などで**文字を丸ごと落としても終了コード 0 を返す**。
    # `-v error` なので正常時の stderr は空。∴ 何か出たら失敗として扱う（DEC-39）。
    if result.stderr.strip():
        raise SystemExit(
            f"ffmpeg が警告を出した（終了コードは 0 だが文字が消えている可能性がある）:\n"
            f"{' '.join(cmd)}\n{result.stderr[-1500:]}")


def _escape(text: str) -> str:
    # `%` は drawtext の展開記号。`\\%` と書かないとその drawtext が丸ごと無視される。
    return (text.replace("\\", "\\\\")
                .replace(":", r"\:")
                .replace("'", r"\'")
                .replace("%", r"\\%"))


def make_title(path: Path, lines: list[tuple[str, int, int]], seconds: float) -> None:
    """黒背景に文字を焼いた区間を作る。lines は (文字列, 文字サイズ, y座標)。"""
    draws = ",".join(
        f"drawtext=fontfile={JP_FONT}:text='{_escape(text)}':"
        f"fontcolor=white:fontsize={size}:x=(w-text_w)/2:y={y}"
        for text, size, y in lines
    )
    run([FFMPEG, "-y", "-v", "error", "-f", "lavfi",
         "-i", f"color=c=0x1a202c:s={SIZE[0]}x{SIZE[1]}:d={seconds}:r={FPS}",
         "-vf", draws, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])


def make_figure_card(path: Path, image: Path, caption: str, seconds: float) -> None:
    """図を白背景の中央に置き、下に一行の説明を焼く。"""
    if not image.exists():
        raise SystemExit(f"図が無い: {image}（先に scripts/make_figures.py を実行）")
    vf = (f"scale={SIZE[0]-120}:{SIZE[1]-170}:force_original_aspect_ratio=decrease,"
          f"pad={SIZE[0]}:{SIZE[1]}:(ow-iw)/2:60:white,"
          f"drawtext=fontfile={JP_FONT}:text='{_escape(caption)}':"
          f"fontcolor=0x1a202c:fontsize=30:x=(w-text_w)/2:y=h-72")
    run([FFMPEG, "-y", "-v", "error", "-loop", "1", "-t", str(seconds), "-i", str(image),
         "-vf", vf, "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])


def make_clip(path: Path, source: Path, start: float, duration: float,
              speed: float, caption: str) -> None:
    """素材を切り出す。speed>1 のときは**倍率を画面に焼き込む**。"""
    # 拡大は lanczos。既定の bicubic だと焼き込み文字が二重ににじむ（DEC-37）。
    filters = [f"scale={SIZE[1]}:{SIZE[1]}:flags=lanczos",
               f"pad={SIZE[0]}:{SIZE[1]}:(ow-iw)/2:0:black"]
    if speed != 1.0:
        filters.append(f"setpts=PTS/{speed}")
        filters.append(
            f"drawtext=fontfile={JP_FONT}:text='{int(speed)}倍速':fontcolor=yellow:"
            f"fontsize=34:box=1:boxcolor=black@0.6:boxborderw=8:x=w-text_w-30:y=30")
    if caption:
        filters.append(
            f"drawtext=fontfile={JP_FONT}:text='{_escape(caption)}':fontcolor=white:"
            f"fontsize=30:box=1:boxcolor=black@0.6:boxborderw=10:"
            f"x=(w-text_w)/2:y=h-70")
    run([FFMPEG, "-y", "-v", "error", "-ss", str(start), "-t", str(duration),
         "-i", str(source), "-vf", ",".join(filters), "-r", str(FPS),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path)])


def _mean_luma(image, box: tuple[int, int, int, int]) -> float:
    from PIL import ImageStat

    return ImageStat.Stat(image.crop(box).convert("L")).mean[0]


def assert_overlay_is_legible(source: Path, timeline: list[dict]) -> None:
    """素材に**下地付きの焼き込み文字**（DEC-38）が入っているか、実物を見て確かめる。

    焼き込み文字は**素材の中にある**。∴ `src/recording.py` を直しても、
    **録り直すまで提出動画には届かない**。
    2026-09-29 に実際にそうなった——下地付きの素材を録った後で
    **古い素材から提出動画を組み**、機械チェックは全項目 OK のままだった（R-35）。

    判定は更新時刻でなく**画素**で行う。帯は下地の明るさを 0.25 倍にするので、
    帯の領域は**真下の同じ列より暗くなる**。更新時刻は複製やチェックアウトで動くので使わない。
    """
    from PIL import Image

    # 帯は手順の実行中だけ描かれる（エピソードの切れ目では消える）。
    # ∴ 各エピソードの**真ん中**を拾う。
    times = [t["start_sec"] + t["duration_sec"] / 2 for t in timeline][:5]
    if not times:
        raise SystemExit("timeline が空で素材を検査できない")

    ratios = []
    with tempfile.TemporaryDirectory() as td:
        for index, t in enumerate(times):
            frame = Path(td) / f"{index}.png"
            run([FFMPEG, "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(source),
                 "-frames:v", "1", str(frame)])
            with Image.open(frame) as image:
                width, height = image.size
                band_h = round(68 / 512 * height)
                band_w = round(300 / 512 * width)
                band = _mean_luma(image, (0, 0, band_w, band_h))
                below = _mean_luma(image, (0, band_h, band_w, band_h * 2))
            ratios.append(band / max(below, 1.0))

    ratios.sort()
    median = ratios[len(ratios) // 2]
    if median >= OVERLAY_BAND_RATIO:
        raise SystemExit(
            f"素材の焼き込み文字に下地が無い: {source}\n"
            f"  帯の明るさ比 {median:.2f}（境 {OVERLAY_BAND_RATIO} 未満で合格）\n"
            "  → DEC-38 以前に録った古い素材である。焼き込み文字は録り直してしか直らない。\n"
            "  録り直し: MUJOCO_GL=egl python scripts/run_episode.py --episodes 5 \\\n"
            "           --planner llm --perception model --yaw-source model --recovery --tag c5_master")


def miss_caption(run_dir: Path, episode: int) -> str:
    """失敗区間の字幕を**その回のログから**作る。

    手打ちにすると、素材を差し替えた瞬間に**画面が実態と違う主張をする**。
    """
    events = run_dir / "events.jsonl"
    if not events.exists():
        return "失敗も隠さない：この回は手順を最後まで通せませんでした"
    for line in events.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("episode") != episode or record.get("success"):
            continue
        name = OBJECT_LABELS.get(record["target"], record["target"])
        reason = FAILURE_MODES.get(record.get("failure_mode"), "失敗")
        return f"失敗も隠さない：この回は{name}で{reason}（{record.get('failure_mode')}）"
    return "失敗も隠さない：この回は手順を最後まで通せませんでした"


def recovery_segment(run_dir: Path, timeline: list[dict]) -> tuple[dict, str] | None:
    """「気づいてやり直して成功した」区間を**ログから選ぶ**（手打ちしない。DEC-62）。

    選び方: やり直しの回数が最も多く、かつ**最終的に成功した**手順を持つエピソード。
    見せたいのは「失敗した」ことではなく「失敗に気づいて立て直した」ことである。
    """
    events = run_dir / "events.jsonl"
    if not events.exists():
        return None

    best = None
    for line in events.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if not record.get("success"):
            continue
        retries = (record.get("grasp_attempts", 1) - 1) + record.get("detect_level", 0)
        if retries <= 0:
            continue
        if best is None or retries > best[0]:
            best = (retries, record)
    if best is None:
        return None

    _, record = best
    segment = next((t for t in timeline if t["episode"] == record["episode"]), None)
    if segment is None:
        return None

    kinds = [e["kind"] for e in record.get("recovery_events", [])]
    name = OBJECT_LABELS.get(record["target"], record["target"])
    if "grasp_miss" in kinds:
        caption = f"空振りを指の開きで検知 → 見る位置からやり直して{name}を掴み直しました"
    else:
        caption = f"見つからない → 言葉を変えて探し直し、{name}を見つけました"
    return segment, caption


def ablation_metrics() -> tuple[str, str, str, int]:
    """動画に焼く数値を、図と同じ実測 JSON から読む（手打ちしない。DEC-28 / DEC-40）。

    **提出構成は C5（自己監視と再試行あり）であり、焼く数値も C5 から取る**（DEC-67）。
    画面に出す数値と実際に映っている構成が違うと、同じ指標が2つの値を持つ。
    """
    path = ROOT / "experiments" / "exp003_ablation" / "result.json"
    results = json.loads(path.read_text())["results"]
    submitted = next(v for k, v in results.items() if k.startswith("C5"))
    c0 = next(v for k, v in results.items() if k.startswith("C0 "))
    step = submitted["step_level"]["step_success_fraction"]
    baseline = c0["step_level"]["step_success_fraction"]
    retention = round(100 * submitted["step_level"]["n_success"] / c0["step_level"]["n_success"])
    return step, submitted["identification_accuracy_fraction"], baseline, retention


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="素材となる run ディレクトリ")
    p.add_argument("--out", default=str(ROOT / "submission" / "提出動画_未署名.mp4"))
    args = p.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    source = run_dir / "video.mp4"
    summary = json.loads((run_dir / "summary.json").read_text())
    timeline = summary["timeline"]

    # 焼き込み文字は素材に入っているので、組み立てる前に素材側を見る（R-35 / C-43）。
    assert_overlay_is_legible(source, timeline)

    success = [t for t in timeline if t["success"]]
    failure = [t for t in timeline if not t["success"]]
    if not success:
        raise SystemExit("成功エピソードが素材に無い。別の run を使うこと")
    hero = success[0]
    # 全エピソード成功の素材では「失敗も隠さない」区間を作れない。落とさず飛ばして知らせる。
    miss = failure[0] if failure else None
    if miss is None:
        print("※ 失敗エピソードが素材に無いため、失敗区間は入れない（台本 §1 の該当行は欠番になる）")

    # 数値は**図と同じ出典**（アブレーション20エピソード）から読む。
    # 素材 run は見せるための5エピソードなので、そちらを使うと
    # **同じ動画の中でグラフとまとめの数字が食い違う**（DEC-40）。
    step, ident, baseline, retention = ablation_metrics()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        parts: list[Path] = []

        def add(name: str) -> Path:
            path = tmp_dir / f"{len(parts):02d}_{name}.mp4"
            parts.append(path)
            return path

        title, subtitle, note, seconds = TITLE_CARDS[0]
        make_title(add("title"), [(title, 52, 250), (subtitle, 34, 350),
                                  (note, 28, 420), ("※一部2倍速（画面に表示）", 26, 500)], seconds)

        make_figure_card(add("task"), FIGURES / "fig2_task.png",
                         "シリアルは指示に合わない＝選んではいけない", 10)
        make_figure_card(add("arch"), FIGURES / "fig1_architecture.png",
                         "計画も知覚も姿勢もAI。学習はせず推論だけを使う", 11)

        make_clip(add("hero"), source, hero["start_sec"], hero["duration_sec"], 1.0,
                  "真値を一切使わず、カメラ画像と言語だけで動いています")

        # **この課題の中心なので、図より先に見せる**。
        # 区間はログから選ぶ（手打ちしない）。やり直しは手順の後半で起きるので末尾を切る（DEC-65）。
        recovery = recovery_segment(run_dir, timeline)
        if recovery is not None:
            segment, caption = recovery
            length = min(segment["duration_sec"], 15.0)
            make_clip(add("recover"), source,
                      segment["start_sec"] + segment["duration_sec"] - length, length, 1.0,
                      caption)
        else:
            print("※ やり直して成功した手順が素材に無いため、回復区間は入れない")

        if miss is not None:
            # 失敗は**エピソードの後ろ**で起きる（手順を順に実行するため）。
            # 先頭から切ると、字幕が言っている失敗が画面に出ないまま区間が終わる。
            miss_len = min(miss["duration_sec"], 8.0)
            make_clip(add("miss"), source,
                      miss["start_sec"] + miss["duration_sec"] - miss_len, miss_len, 1.0,
                      miss_caption(run_dir, miss["episode"]))
        # 通し区間の尺を**手打ちしない**。以前は 47.0 秒固定で、それは
        # この素材では ep0–ep3 だけ（全て失敗回）を指していた——
        # **「5エピソードを通しで」という字幕は事実と違っていた**（R-36）。
        #
        # 字幕には**成功率を書かない**。素材 run は見せるための5エピソードであり、
        # 図が使うアブレーションとは母数が違う。**同じ指標が2つの値を持つ**（DEC-40）。
        recap_sec = sum(t["duration_sec"] for t in timeline)
        make_clip(add("fast"), source, 0.0, recap_sec, 2.0,
                  f"{len(timeline)}エピソードを通しで")

        make_figure_card(add("para"), FIGURES / "fig5_paraphrase.png",
                         "固定ルールでは扱えない言い換えにも対応する", 10)
        make_figure_card(add("abl"), FIGURES / "fig3_ablation.png",
                         f"提出構成で {step}。真値を使う上限（{baseline}）を上回った", 13)
        make_figure_card(add("order"), FIGURES / "fig4_plan_order.png",
                         "運ぶ順番だけで12手順動いた（AI置き換えのコストの3倍）", 9)
        make_figure_card(add("limit"), FIGURES / "fig6_failure_modes.png",
                         "残る律速は把持失敗。真値を使っても同じだけ起きる", 9)

        make_title(add("end"), [
            ("まとめ", 46, 180),
            (f"全層AI＋自己監視で手順成功 {step} ／ 対象の同定 {ident}", 30, 300),
            ("失敗に気づいてやり直すのも、真値を使わずセンサだけで行う", 28, 365),
            ("限界：把持の失敗が残る／やり直しは時間を 1.6 倍使う", 28, 430),
        ], 10)

        list_file = tmp_dir / "parts.txt"
        list_file.write_text("".join(f"file '{p}'\n" for p in parts))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", str(list_file), "-c:v", "libx264", "-crf", "23",
             # moov を先頭へ。提出先がブラウザで先頭再生するときに全体を落とさずに済む。
             "-movflags", "+faststart",
             "-pix_fmt", "yuv420p", str(out)])

    # ffprobe は PATH から引く。FFMPEG のパス文字列置換だと、パスに "ffmpeg" が
    # 2回現れる環境（/opt/ffmpeg/bin/ffmpeg 等）でディレクトリ側まで置換されて壊れる。
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries",
         "format=duration,size", "-of", "default=nw=1", str(out)],
        capture_output=True, text=True).stdout.strip()
    info = dict(line.split("=") for line in probe.splitlines())
    seconds, megabytes = float(info["duration"]), int(info["size"]) / 1e6

    print(f"出力: {out}")
    print(f"長さ: {seconds:.1f} 秒   容量: {megabytes:.1f} MB")
    # 要項の10/8版で上限が **5分 → 3分** に短縮された（DEC-66）。
    ok_len = 60 <= seconds <= 180
    ok_size = megabytes <= 200
    print(f"尺 1〜3分: {'✔' if ok_len else '✘'}   容量 200MB以下: {'✔' if ok_size else '✘'}")
    if not ok_len:
        print(f"  → 上限 180 秒に対し {seconds - 180:+.1f} 秒。図カードの秒数か通し区間を削ること")
    return 0 if (ok_len and ok_size) else 1


if __name__ == "__main__":
    raise SystemExit(main())
