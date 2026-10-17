#!/usr/bin/env python
"""提出動画・レポートに使う図を作る。

**数値はすべて実測ファイルから読む。** 図に手打ちした数値は残さない
（文書と図で数値が割れる事故を構造的に防ぐ）。

使い方:
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "docs" / "40_報告" / "figures"
JP_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

ACCENT = "#2b6cb0"
MUTED = "#a0aec0"
WARN = "#c53030"
GOOD = "#2f855a"   # 提出構成（C5）を他と区別する


def setup_font() -> None:
    """日本語が豆腐にならないようフォントを登録する。無ければ例外で止める。"""
    if not Path(JP_FONT).exists():
        raise SystemExit(f"日本語フォントが無い: {JP_FONT}")
    fm.fontManager.addfont(JP_FONT)
    plt.rcParams["font.family"] = fm.FontProperties(fname=JP_FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")
    return path


# --- 図1: システム構成 ---------------------------------------------------
def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axis("off")
    layers = [
        ("指示文「飲み物を左の箱へ、食べ物を右の箱へ」", "#edf2f7", "入力"),
        ("計画層  Qwen2.5-1.5B（言語モデル）\n指示 → 手順のリスト", "#bee3f8", "AI"),
        ("知覚層  OWLv2（オープン語彙の物体検出）\n画像＋語 → 物体の位置と向き", "#bee3f8", "AI"),
        ("制御層  動作プリミティブ\n掴む → 運ぶ → 置く", "#e2e8f0", "規則"),
        ("robosuite / MuJoCo（シミュレータ）", "#edf2f7", "環境"),
    ]
    for i, (text, color, tag) in enumerate(layers):
        y = 4 - i
        ax.add_patch(plt.Rectangle((0.5, y), 7.4, 0.78, facecolor=color,
                                   edgecolor="#4a5568", linewidth=1.2))
        ax.text(4.2, y + 0.39, text, ha="center", va="center", fontsize=11)
        ax.text(8.3, y + 0.39, tag, ha="left", va="center", fontsize=10,
                color=ACCENT if tag == "AI" else "#718096",
                fontweight="bold" if tag == "AI" else "normal")
        if i < len(layers) - 1:
            ax.annotate("", xy=(4.2, y - 0.02), xytext=(4.2, y - 0.22),
                        arrowprops=dict(arrowstyle="-|>", color="#4a5568"))
    ax.set_xlim(0, 9.6)
    ax.set_ylim(-0.4, 5.1)
    ax.set_title("システム構成：計画・知覚はAI、制御は規則", fontsize=13, pad=12)
    _save(fig, "fig1_architecture.png")


# --- 図2: タスク設定 -----------------------------------------------------
def fig_task() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.axis("off")
    objects = [
        ("牛乳パック", "飲み物", "左の箱", True),
        ("ジュースの缶", "飲み物", "左の箱", True),
        ("パン", "食べ物", "右の箱", True),
        ("シリアルの箱", "どちらでもない", "触らない", False),
    ]
    ax.text(4.5, 3.6, "指示：「飲み物を左の箱へ、食べ物を右の箱へ」",
            ha="center", fontsize=13, fontweight="bold")
    for i, (name, cat, dest, target) in enumerate(objects):
        y = 2.6 - i * 0.62
        color = "#bee3f8" if target else "#fed7d7"
        ax.add_patch(plt.Rectangle((0.6, y), 2.4, 0.5, facecolor=color,
                                   edgecolor="#4a5568"))
        ax.text(1.8, y + 0.25, name, ha="center", va="center", fontsize=11)
        ax.text(3.4, y + 0.25, cat, ha="left", va="center", fontsize=10, color="#4a5568")
        ax.annotate("", xy=(6.1, y + 0.25), xytext=(5.4, y + 0.25),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=ACCENT if target else WARN))
        ax.text(6.3, y + 0.25, dest, ha="left", va="center", fontsize=11,
                color=ACCENT if target else WARN,
                fontweight="normal" if target else "bold")
    ax.text(4.5, 0.02, "シリアルは指示に合わない＝選んではいけない（妨害物）",
            ha="center", fontsize=10.5, color=WARN)
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.2, 4)
    _save(fig, "fig2_task.png")


# --- 図3: アブレーション -------------------------------------------------
def fig_ablation() -> None:
    data = load(ROOT / "experiments" / "exp003_ablation" / "result.json")["results"]
    picked = [
        ("真値を使う\n（AIなしの上限）", "C0 "),
        ("位置を深度から\n（初期実装）", "C1d"),
        ("知覚と姿勢がAI", "C3 "),
        ("計画もAI\n（全層AI）", "C4 "),
        ("＋自己監視と再試行\n（提出構成）", "C5 "),
    ]
    labels, values, totals = [], [], []
    for label, prefix in picked:
        key = next(k for k in data if k.startswith(prefix))
        frac = data[key]["step_level"]["step_success_fraction"]
        num, den = (int(v) for v in frac.split("/"))
        labels.append(label)
        values.append(num)
        totals.append(den)

    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    # 提出構成（C5）だけを強調し、真値の上限を破線で示す。
    colors = [MUTED, WARN, ACCENT, ACCENT, GOOD]
    bars = ax.bar(labels, values, color=colors[:len(values)], width=0.62)
    for bar, num, den in zip(bars, values, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, num + 0.8, f"{num}/{den}",
                ha="center", fontsize=12, fontweight="bold")
    ax.axhline(values[0], color=MUTED, linestyle="--", linewidth=1)
    ax.text(0.5, values[0] + 1.2, "真値を使う上限", ha="left", fontsize=9.5, color=MUTED)
    ax.set_ylabel("手順成功数（60手順中）", fontsize=11)
    ax.set_ylim(0, max(totals) * 0.92)
    ax.set_title("何が効いたか（20エピソード×各条件、初期配置は完全一致）", fontsize=12.5, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig3_ablation.png")


# --- 図4: 実行順の影響 ---------------------------------------------------
def fig_plan_order() -> None:
    data = load(ROOT / "experiments" / "exp006_plan_order" / "result.json")["results"]
    labels, values, totals = [], [], []
    for name, summary in data.items():
        frac = summary["step_level"]["step_success_fraction"]
        num, den = (int(v) for v in frac.split("/"))
        labels.append(name.split("（")[0].replace(" ", "\n", 1))
        values.append(num)
        totals.append(den)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.bar(labels, values, color=[ACCENT, WARN, MUTED], width=0.55)
    for bar, num, den in zip(bars, values, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, num + 0.8, f"{num}/{den}",
                ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("手順成功数（60手順中）", fontsize=11)
    ax.set_ylim(0, max(totals) * 0.92)
    ax.set_title("運ぶ順番を変えるだけで成功率が変わる（中身は同じ3手順）",
                 fontsize=12.5, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig4_plan_order.png")


# --- 図5: 言い換えへの対応 -----------------------------------------------
def fig_paraphrase() -> None:
    # _ex2.json（2要素 few-shot＝採用条件・DEC-22 の本実行）が数値の正
    # （docs/30_評価/20260927_言語計画層_結果.md の出典）。
    # 無印 .json は probe_planner.py の既定出力で、回帰時に上書きされ得るため出典にしない。
    data = load(ROOT / "experiments" / "exp005_llm_planner" / "Qwen2.5-1.5B-Instruct_ex2.json")
    rows = data["言い換え"]["内訳"]
    jp = {"Milk": "牛乳", "Can": "缶", "Bread": "パン", "Cereal": "シリアル"}

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axis("off")
    ax.text(0.02, 0.93, "固定ルールでは1つも扱えない言い換えに、言語モデルが対応した",
            fontsize=12.5, fontweight="bold", transform=ax.transAxes)
    for i, row in enumerate(rows):
        y = 0.74 - i * 0.18
        mark = "OK" if row["一致"] else "NG"
        ax.text(0.02, y, mark, fontsize=11, fontweight="bold",
                color=ACCENT if row["一致"] else WARN, transform=ax.transAxes)
        ax.text(0.07, y, f"「{row['instruction']}」", fontsize=11, transform=ax.transAxes)
        got = " / ".join(f"{jp.get(t, t)}→{d}" for t, d in row["got"])
        ax.text(0.52, y, got or "（手順なし）", fontsize=10.5, color="#4a5568",
                transform=ax.transAxes)
    _save(fig, "fig5_paraphrase.png")


# --- 図6: 限界 -----------------------------------------------------------
def fig_limits() -> None:
    data = load(ROOT / "experiments" / "exp003_ablation" / "result.json")["results"]
    key = next(k for k in data if k.startswith("C4 "))
    modes = data[key]["step_level"]["failure_modes"]
    names = {"F-3": "未検出", "F-5": "把持失敗", "F-7": "配置失敗",
             "F-4": "位置誤差", "F-6": "搬送失敗", "F-8": "タイムアウト"}
    items = sorted(modes.items(), key=lambda x: -x[1]["count"])
    labels = [f"{names.get(k, k)}\n({k})" for k, _ in items]
    counts = [v["count"] for _, v in items]
    success = int(data[key]["step_level"]["step_success_fraction"].split("/")[0])

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bars = ax.bar(["成功"] + labels, [success] + counts,
                  color=[ACCENT] + [WARN] * len(counts), width=0.6)
    for bar, value in zip(bars, [success] + counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, str(value),
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("手順数（合計60）", fontsize=11)
    ax.set_title("全層AIの内訳：成功と失敗の理由", fontsize=12.5, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig6_failure_modes.png")


def main() -> int:
    setup_font()
    print("図を生成:")
    fig_architecture()
    fig_task()
    fig_ablation()
    fig_plan_order()
    fig_paraphrase()
    fig_limits()
    print(f"\n保存先: {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
