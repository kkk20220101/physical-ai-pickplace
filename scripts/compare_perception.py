#!/usr/bin/env python
"""知覚の条件（モデル × 画角）を比較し、採用条件を実測で決める。

比較するのは2軸:
  モデル: OWL-ViT (base-patch32) 対 OWLv2 (base-patch16-ensemble)
  画角:   全体画像 対 卓上の切り出し（物体が大きく写る）

判定は「語がどの物体に当たったか」を真値の投影位置と突き合わせて行う。

使い方:
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt HF_HUB_DISABLE_XET=1
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 \
      python scripts/compare_perception.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.device import default_device  # noqa: E402
from src.envs.pickplace import EnvConfig, make_env, object_position  # noqa: E402
from src.perception.owlvit_detector import PerceptionConfig, camera_intrinsics, detect  # noqa: E402

OBJECTS = ["Milk", "Bread", "Cereal", "Can"]
CAMERA = "agentview"
MATCH_RADIUS_PX = 45

# 意味語（人が指示に使う語）で統一する。見た目語に逃げると「言語で指定した」と言えなくなる。
QUERIES = {
    "Milk": ["a carton of milk", "a milk box"],
    "Bread": ["a loaf of bread", "a piece of bread"],
    "Cereal": ["a cereal box", "a box of cereal"],
    "Can": ["a soda can", "a can of soda"],
}

# 卓上（物体が置かれている左のテーブル）だけを切り出す比率。実画像を見て決めた。
TABLE_CROP = (0.02, 0.25, 0.56, 0.72)


def world_to_pixel(env, name, size):
    cam_id = env.sim.model.camera_name2id(CAMERA)
    focal, cx, cy = camera_intrinsics(env, CAMERA, size, size)
    cam_pos = np.array(env.sim.data.cam_xpos[cam_id])
    cam_mat = np.array(env.sim.data.cam_xmat[cam_id]).reshape(3, 3)
    rel = cam_mat.T @ (object_position(env, name) - cam_pos)
    depth = -rel[2]
    return np.array([cx + rel[0] * focal / depth, cy - rel[1] * focal / depth])


def run_condition(env, name: str, cfg: PerceptionConfig, seeds: int, size: int) -> dict:
    correct = defaultdict(int)
    wrong = defaultdict(int)
    missing = defaultdict(int)
    pixel_err: dict[str, list[float]] = defaultdict(list)

    for seed in range(seeds):
        np.random.seed(seed)
        obs = env.reset()
        rgb = np.flipud(obs[f"{CAMERA}_image"]).copy()
        truth = {n: world_to_pixel(env, n, size) for n in OBJECTS}
        found = detect(rgb, cfg)

        for obj in OBJECTS:
            if obj not in found:
                missing[obj] += 1
                continue
            x1, y1, x2, y2 = found[obj]["bbox"]
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            distance = float(np.linalg.norm(center - truth[obj]))
            nearest = min(truth, key=lambda n: np.linalg.norm(center - truth[n]))
            if nearest == obj and distance <= MATCH_RADIUS_PX:
                correct[obj] += 1
                pixel_err[obj].append(distance)
            else:
                wrong[obj] += 1

    total = seeds * len(OBJECTS)
    n_correct = sum(correct.values())
    return {
        "condition": name,
        "identification_fraction": f"{n_correct}/{total}",
        "identification_rate": round(n_correct / total, 3),
        "per_object": {
            o: {"正解": correct[o], "取り違え": wrong[o], "未検出": missing[o],
                "画素誤差中央値": round(float(np.median(pixel_err[o])), 1) if pixel_err[o] else None}
            for o in OBJECTS
        },
        "ledger_check": f"{n_correct} + {sum(wrong.values())} + {sum(missing.values())} = {total} "
                        f"{'✔' if n_correct + sum(wrong.values()) + sum(missing.values()) == total else '✘'}",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--device", default=default_device())
    p.add_argument("--out", default=str(ROOT / "experiments" / "exp002_perception_probe" / "compare.json"))
    args = p.parse_args()

    env = make_env(EnvConfig(camera_height=args.size, camera_width=args.size, camera_depths=True))

    # 「全体」条件は crop=None を**明示**する。PerceptionConfig の既定は採用条件
    # （卓上切り出し）に書き換わったため、既定に頼ると P1/P3 が P2/P4 と同じ条件になり、
    # 記録済みの 5/20・11/20 を再現できない（2026-09-29 レビューで検出）。
    conditions = {
        "P1 OWL-ViT / 全体": PerceptionConfig(
            model_id="google/owlvit-base-patch32", queries=QUERIES,
            score_threshold=args.threshold, device=args.device, crop=None),
        "P2 OWL-ViT / 卓上切出し": PerceptionConfig(
            model_id="google/owlvit-base-patch32", queries=QUERIES,
            score_threshold=args.threshold, device=args.device, crop=TABLE_CROP),
        "P3 OWLv2 / 全体": PerceptionConfig(
            model_id="google/owlv2-base-patch16-ensemble", queries=QUERIES,
            score_threshold=args.threshold, device=args.device, crop=None),
        "P4 OWLv2 / 卓上切出し": PerceptionConfig(
            model_id="google/owlv2-base-patch16-ensemble", queries=QUERIES,
            score_threshold=args.threshold, device=args.device, crop=TABLE_CROP),
    }

    results = []
    for name, cfg in conditions.items():
        res = run_condition(env, name, cfg, args.seeds, args.size)
        results.append(res)
        print(f"{name:26s} 同定 {res['identification_fraction']:>6s} "
              f"({res['identification_rate']:.0%})  {res['ledger_check']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": args.seeds, "queries": QUERIES,
                               "crop": TABLE_CROP, "results": results},
                              ensure_ascii=False, indent=2))

    best = max(results, key=lambda r: r["identification_rate"])
    print(f"\n最良: {best['condition']}  同定 {best['identification_fraction']}")
    print(f"{'物体':8s} {'正解':>4s} {'取違':>4s} {'未検出':>6s}  画素誤差中央値")
    for o in OBJECTS:
        d = best["per_object"][o]
        print(f"{o:8s} {d['正解']:>4d} {d['取り違え']:>4d} {d['未検出']:>6d}  {d['画素誤差中央値']}")
    print(f"\n保存先: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
