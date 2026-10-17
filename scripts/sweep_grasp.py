#!/usr/bin/env python
"""物体ごとに「掴んで持ち上げられるか」を実測する。

目的: どの物体が安定して扱えるかを数値で決め、タスク対象を確定させる（意思決定 DEC-09）。
把持点の z 補正を振って、持ち上げ成功率を物体×補正値の表にする。

使い方:
    MUJOCO_GL=egl python scripts/sweep_grasp.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.control.primitives import GRIPPER_OPEN, ControlConfig, Primitives  # noqa: E402
from src.envs.pickplace import EnvConfig, make_env, object_position, object_yaw  # noqa: E402

LIFT_SUCCESS_M = 0.05  # これ以上持ち上がったら把持成功（docs/20_設計/02_評価設計.md F-5 の定義）


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--offsets", type=float, nargs="+", default=[-0.01, 0.0, 0.01, 0.02])
    p.add_argument("--objects", nargs="+", default=["Milk", "Bread", "Cereal", "Can"])
    p.add_argument("--out", default=str(ROOT / "experiments" / "exp001_grasp_sweep" / "result.json"))
    args = p.parse_args()

    table: dict[str, dict[str, dict]] = {obj: {} for obj in args.objects}

    for offset in args.offsets:
        cfg = ControlConfig(grasp_z_offset=offset)
        for seed in range(args.seeds):
            env = make_env(EnvConfig(camera_height=64, camera_width=64, seed=seed))
            prim = Primitives(env, cfg)
            for obj in args.objects:
                prim.reset()  # 物体ごとに初期化し、前の試行の影響を残さない
                pos = object_position(env, obj)
                ok = prim.pick(pos, object_yaw=object_yaw(env, obj))
                lift = float(object_position(env, obj)[2] - pos[2]) if ok else 0.0
                bucket = table[obj].setdefault(f"{offset:+.2f}", {"n": 0, "lifted": 0, "lifts": []})
                bucket["n"] += 1
                bucket["lifted"] += int(lift >= LIFT_SUCCESS_M)
                bucket["lifts"].append(round(lift, 3))
                prim.set_gripper(GRIPPER_OPEN)
            env.close()
        print(f"offset={offset:+.2f} 完了", flush=True)

    for obj, by_offset in table.items():
        for key, bucket in by_offset.items():
            bucket["rate"] = round(bucket["lifted"] / bucket["n"], 3)
            bucket["fraction"] = f"{bucket['lifted']}/{bucket['n']}"
            bucket["median_lift_m"] = round(float(np.median(bucket["lifts"])), 3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2))

    print("\n物体 \\ 把持点zの補正 | " + " | ".join(f"{o:+.2f}" for o in args.offsets))
    for obj in args.objects:
        cells = " | ".join(table[obj][f"{o:+.2f}"]["fraction"] for o in args.offsets)
        print(f"{obj:8s} | {cells}")
    print(f"\n保存先: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
