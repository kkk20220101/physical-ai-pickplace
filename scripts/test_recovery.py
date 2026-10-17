#!/usr/bin/env python
"""自己監視と再試行（C5）が壊れていないことを確かめる。シミュレータを起動しない。

守りたいのは2つ:
  1. **既定では何も変わらない**。`recovery=False` が既定であり、再探索も掴み直しも
     起きない。C0〜C4 の記録済みの数値（41/60 等）はこれに依存している。
  2. **緩めた検出は妥当性検査を通る**。閾値を下げると「未検出」が「取り違え」に化ける
     ので、台の外や他物体と重なる候補は落とす。

使い方: python scripts/test_recovery.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.control.primitives import GRASP_GAP_THRESHOLD, finger_gap  # noqa: E402
from src.eval import summarize_recovery  # noqa: E402
from src.perception.owlvit_detector import (  # noqa: E402
    RETRY_QUERIES, PerceptionConfig, escalated_config,
)
from src.pipeline import RunConfig  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'✔' if condition else '✘'} {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_defaults_unchanged() -> None:
    """既定が変わっていたら、記録済みの C0〜C4 が再現しなくなる。"""
    print("既定値（これが False でないと過去の数値が再現しない）")
    cfg = RunConfig()
    check("RunConfig.recovery の既定は False", cfg.recovery is False, f"実際は {cfg.recovery}")
    check("探索の段0は既定設定そのもの",
          escalated_config(PerceptionConfig(), 0) == PerceptionConfig())


def test_escalation_monotonic() -> None:
    """段を上げるほど『緩む』こと。緩まないなら探し直す意味がない。"""
    print("段階的再探索")
    base = PerceptionConfig()
    lv1 = escalated_config(base, 1)
    lv2 = escalated_config(base, 2)
    check("段1で語が増える",
          all(len(lv1.queries[n]) > len(base.queries[n]) for n in base.queries))
    check("段1では閾値を動かさない", lv1.score_threshold == base.score_threshold)
    check("段2で閾値が下がる", lv2.score_threshold < base.score_threshold)
    check("段1の語は段2でも保たれる", lv1.queries == lv2.queries)
    check("既定の設定を書き換えていない",
          base.score_threshold == PerceptionConfig().score_threshold
          and base.queries == PerceptionConfig().queries)
    flat = [w for words in RETRY_QUERIES.values() for w in words]
    banned = ("red", "blue", "green", "cylinder", "yellow", "white", "colour", "color")
    check("追加語に見た目語が混ざっていない（言語で指定したという主張を守る）",
          not any(b in w.lower() for w in flat for b in banned),
          f"見た目語: {[w for w in flat if any(b in w.lower() for b in banned)]}")


def test_plausibility_gate() -> None:
    """緩めた候補の歯止め。台の外・他物体との重なりを落とすこと。"""
    print("妥当性検査（真値を使わずに緩めた検出を守る門）")

    class FakeEnv:
        bin1_pos = np.array([0.1, -0.25, 0.8])
        bin_size = np.array([0.39, 0.49, 0.82])

    from src.perception.owlvit_detector import is_plausible, pickup_table_bounds

    env = FakeEnv()
    low, high = pickup_table_bounds(env)
    check("台の範囲が左のテーブルを囲っている",
          low[0] < 0.0 < high[0] and low[1] < -0.25 < high[1], f"{low} {high}")

    ok, _ = is_plausible(env, np.array([0.10, -0.25, 0.85]))
    check("台の上の点は通る", ok)

    ng, reason = is_plausible(env, np.array([0.10, 0.40, 0.85]))
    check("台の外の点は落とす", not ng, reason)

    others = {"Milk": {"position_3d": np.array([0.10, -0.25, 0.85])}}
    ng2, reason2 = is_plausible(env, np.array([0.11, -0.25, 0.85]), others)
    check("他物体と重なる点は落とす（同じ物を2つの名前で指さない）", not ng2, reason2)

    ok2, _ = is_plausible(env, np.array([0.10, -0.40, 0.85]), others)
    check("十分離れていれば通る", ok2)

    ng3, _ = is_plausible(env, None)
    check("位置が無いものは通さない", not ng3)


def test_grasp_sensor() -> None:
    """把持判定は固有受容感覚だけで行うこと（真値を使わない）。"""
    print("把持の自己判定（指の開き幅）")
    # 実測値（scripts/probe_grasp_sensor.py・36試行）: 成功時 最小 0.0372 / 失敗時 最大 0.0015
    grasped = {"robot0_gripper_qpos": np.array([0.0186, -0.0186])}
    missed = {"robot0_gripper_qpos": np.array([0.0005, -0.0005])}
    check("掴めている指の開きは閾値より大きい", finger_gap(grasped) > GRASP_GAP_THRESHOLD,
          f"{finger_gap(grasped)}")
    check("空振りの指の開きは閾値より小さい", finger_gap(missed) < GRASP_GAP_THRESHOLD,
          f"{finger_gap(missed)}")
    check("閾値が実測の2値の間にある", 0.0015 < GRASP_GAP_THRESHOLD < 0.0372)


def test_recovery_summary() -> None:
    """集計が『やり直した件数』を隠さないこと。"""
    print("再試行の集計")
    steps = [
        {"success": True, "failure_mode": None, "detect_level": 0, "grasp_attempts": 1,
         "recovery_events": []},
        {"success": True, "failure_mode": None, "detect_level": 2, "grasp_attempts": 1,
         "recovery_events": [{"kind": "redetect", "level": 2}]},
        {"success": False, "failure_mode": "F-5", "detect_level": 0, "grasp_attempts": 3,
         "recovery_events": [{"kind": "grasp_miss"}, {"kind": "reject"}]},
    ]
    s = summarize_recovery(steps)
    check("やり直した手順を数える", s["n_retried"] == 2, str(s))
    check("やり直して成功した手順だけを救済に数える", s["n_rescued"] == 1, str(s))
    check("1回で成功した手順は救済に数えない", s["rescued_fraction"] == "1/2", str(s))
    check("却下件数を出す", s["n_plausibility_rejections"] == 1, str(s))
    check("再試行していない実験では集計自体を出さない",
          summarize_recovery([{"success": True, "failure_mode": None}]) is None)


def main() -> int:
    for test in (test_defaults_unchanged, test_escalation_monotonic, test_plausibility_gate,
                 test_grasp_sensor, test_recovery_summary):
        test()
    print()
    if FAILURES:
        print(f"不合格 {len(FAILURES)} 件: {FAILURES}")
        return 1
    print("すべて合格")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
