#!/usr/bin/env python
"""把持の成否を「真値を見ずに」判定できるかを確かめる書き捨ての探り。

現状の F-5（把持失敗）は `object_position()`（シミュレータの真値）で持ち上がりを測っている。
これは**採点**には正しいが、**制御の判断材料には使えない**（実機に真値は無い）。

∴ 固有受容感覚＝グリッパの指の開き幅（`robot0_gripper_qpos`）で判定できるかを測る。
物体を掴めていれば指は物体の幅で止まり、空振りなら最後まで閉じる。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.control.primitives import ControlConfig, Primitives  # noqa: E402
from src.envs.pickplace import (  # noqa: E402
    GRASP_Z_OFFSET, EnvConfig, make_env, object_position, object_yaw,
)


def finger_gap(obs) -> float:
    """指の開き幅 [m]。Panda は2つの指関節を持ち、qpos は (右, 左) で符号が逆。"""
    q = np.asarray(obs["robot0_gripper_qpos"], dtype=float)
    return float(abs(q[0]) + abs(q[1]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--out", default=str(ROOT / "experiments" / "exp009_recovery_probe" / "grasp_sensor.json"))
    args = p.parse_args()

    env = make_env(EnvConfig(camera_depths=False))
    rows = []

    for ep in range(args.episodes):
        for target in ("Milk", "Can", "Bread"):
            np.random.seed(ep)
            prim = Primitives(env, ControlConfig())
            prim.reset()

            gap_open = finger_gap(prim.obs)
            truth_before = object_position(env, target).copy()
            prim.cfg.grasp_z_offset = GRASP_Z_OFFSET.get(target, 0.0)
            prim.pick(truth_before, object_yaw=object_yaw(env, target))

            gap_after = finger_gap(prim.obs)
            lift = float(object_position(env, target)[2] - truth_before[2])

            row = {"episode": ep, "target": target,
                   "gap_open": round(gap_open, 4), "gap_after": round(gap_after, 4),
                   "lift_m": round(lift, 4), "grasped_truth": bool(lift >= 0.05)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2))

    ok = [r for r in rows if r["grasped_truth"]]
    ng = [r for r in rows if not r["grasped_truth"]]
    print(f"\n把持成功 {len(ok)} / 失敗 {len(ng)}（真値基準）")
    if ok:
        print(f"  成功時の指の開き: 中央値 {np.median([r['gap_after'] for r in ok]):.4f} "
              f"最小 {min(r['gap_after'] for r in ok):.4f}")
    if ng:
        print(f"  失敗時の指の開き: 中央値 {np.median([r['gap_after'] for r in ng]):.4f} "
              f"最大 {max(r['gap_after'] for r in ng):.4f}")

    # 分離できる閾値があるか
    print("\n閾値ごとの一致率（指の開き > 閾値 を「掴めている」と判定）:")
    for thr in (0.005, 0.008, 0.010, 0.012, 0.015, 0.020, 0.025):
        agree = sum(1 for r in rows if (r["gap_after"] > thr) == r["grasped_truth"])
        print(f"  {thr:.3f}: {agree}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
