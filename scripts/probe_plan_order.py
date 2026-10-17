#!/usr/bin/env python
"""手順の**実行順**が成功率にどれだけ効くかを測る。

なぜ必要か:
  言語モデルの計画層（C4）は、固定ルール（C3）と**同じ3手順**を出すが**順番が違う**。
  C4 が 29/60、C3 が 41/60 だったとき、この差を「言語モデルが劣る」と読むのは誤りで、
  **順番だけで説明できる可能性**がある。交絡を切り分けないと結論を出せない。

比較する3条件（**計画層は使わず、手順を直接与える**。他の設定はすべて同じ）:
  順A: Milk → Can → Bread （固定ルールが出す順）
  順B: Can → Milk → Bread （言語モデルが出す順）
  順C: Bread → Milk → Can （先に別の箱へ行く順）

使い方:
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 \
      python scripts/probe_plan_order.py --episodes 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.control.primitives import ControlConfig  # noqa: E402
from src.envs.pickplace import EnvConfig, make_env  # noqa: E402
from src.eval import summarize_episodes  # noqa: E402
from src.pipeline import RunConfig, run_episode  # noqa: E402

ORDERS = {
    "順A Milk→Can→Bread（固定ルールの順）": [("Milk", "左の箱"), ("Can", "左の箱"), ("Bread", "右の箱")],
    "順B Can→Milk→Bread（言語モデルの順）": [("Can", "左の箱"), ("Milk", "左の箱"), ("Bread", "右の箱")],
    "順C Bread→Milk→Can（先に別の箱へ）": [("Bread", "右の箱"), ("Milk", "左の箱"), ("Can", "左の箱")],
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--camera-size", type=int, default=512)
    p.add_argument("--perception", default="model", choices=["ground_truth", "model"])
    p.add_argument("--out", default=str(ROOT / "experiments" / "exp006_plan_order" / "result.json"))
    args = p.parse_args()

    env = make_env(EnvConfig(camera_height=args.camera_size, camera_width=args.camera_size,
                             camera_depths=True))

    results = {}
    for name, plan in ORDERS.items():
        cfg = RunConfig(seed=args.seed, control=ControlConfig(), explicit_plan=plan,
                        perception=args.perception, position_mode="ray_plane",
                        yaw_source="model" if args.perception == "model" else "truth")
        episodes = [run_episode(env, cfg, episode_index=i) for i in range(args.episodes)]
        summary = summarize_episodes(episodes)
        summary["plan"] = [f"{t}→{d}" for t, d in plan]
        results[name] = summary
        modes = summary["step_level"]["failure_modes"]
        top = " ".join(f"{k}({v['count']})" for k, v in
                       sorted(modes.items(), key=lambda x: -x[1]["count"]))
        print(f"{name:34s} タスク {summary['episode_success_fraction']:>6s}  "
              f"手順 {summary['step_level']['step_success_fraction']:>6s}  "
              f"{summary['step_level']['ledger_check']}  {top}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"episodes": args.episodes, "perception": args.perception,
                               "results": results}, ensure_ascii=False, indent=2, default=str))
    print(f"\n保存先: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
