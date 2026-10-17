#!/usr/bin/env python
"""検出に使う語を実測で選ぶ。

OWL-ViT は実写で学習されているため、シミュレータの低精細な描画では
「牛乳パック」のような意味語が当たらないことがある。
**どの語がどの物体に当たるかを真値と突き合わせて測り、語を決める。**

使い方:
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt HF_HUB_DISABLE_XET=1
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 \
      python scripts/sweep_prompts.py --seeds 3
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
from src.perception.owlvit_detector import (  # noqa: E402
    PerceptionConfig,
    camera_intrinsics,
    detect,
)

OBJECTS = ["Milk", "Bread", "Cereal", "Can"]
CAMERA = "agentview"

# 見た目に基づく語も含めて広く試す（意味語だけでは当たらない可能性があるため）
CANDIDATES = [
    "a carton of milk", "a white carton", "a white box", "a tall white object",
    "a loaf of bread", "a brown cube", "a small brown box", "a brown object",
    "a cereal box", "a tall red box", "a dark red box", "a red carton",
    "a soda can", "a red can", "a red cylinder", "a red bottle",
]

MATCH_RADIUS_PX = 45  # この画素距離以内なら「その物体を指した」とみなす


def world_to_pixel(env, name, size):
    """真値の3D位置を画素に投影する（当たり外れの判定に使う）。"""
    cam_id = env.sim.model.camera_name2id(CAMERA)
    focal, cx, cy = camera_intrinsics(env, CAMERA, size, size)
    cam_pos = np.array(env.sim.data.cam_xpos[cam_id])
    cam_mat = np.array(env.sim.data.cam_xmat[cam_id]).reshape(3, 3)
    rel = cam_mat.T @ (object_position(env, name) - cam_pos)
    depth = -rel[2]
    return np.array([cx + rel[0] * focal / depth, cy - rel[1] * focal / depth]), depth


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.02)
    p.add_argument("--device", default=default_device())
    p.add_argument("--out", default=str(ROOT / "experiments" / "exp002_perception_probe" / "prompts.json"))
    args = p.parse_args()

    # 語ごとに別々の「物体」として投げ、当たった先を真値で判定する
    cfg = PerceptionConfig(score_threshold=args.threshold, device=args.device,
                           queries={q: [q] for q in CANDIDATES})

    env = make_env(EnvConfig(camera_height=args.size, camera_width=args.size, camera_depths=True))
    hits: dict[str, dict[str, int]] = {q: defaultdict(int) for q in CANDIDATES}
    scores: dict[str, list[float]] = {q: [] for q in CANDIDATES}

    for seed in range(args.seeds):
        np.random.seed(seed)
        obs = env.reset()
        rgb = np.flipud(obs[f"{CAMERA}_image"]).copy()
        truth_px = {n: world_to_pixel(env, n, args.size)[0] for n in OBJECTS}

        for query, det in detect(rgb, cfg).items():
            x1, y1, x2, y2 = det["bbox"]
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            distances = {n: float(np.linalg.norm(center - px)) for n, px in truth_px.items()}
            nearest = min(distances, key=distances.get)
            target = nearest if distances[nearest] <= MATCH_RADIUS_PX else "（外れ）"
            hits[query][target] += 1
            scores[query].append(det["score"])
        print(f"seed{seed} 完了", flush=True)

    table = {
        q: {"当たり": dict(hits[q]),
            "平均スコア": round(float(np.mean(scores[q])), 3) if scores[q] else None}
        for q in CANDIDATES
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2))

    print(f"\n{'語':32s} {'当たった先（回数）':34s} 平均スコア")
    for q in CANDIDATES:
        hit = " ".join(f"{k}:{v}" for k, v in sorted(hits[q].items(), key=lambda x: -x[1])) or "検出なし"
        print(f"{q:32s} {hit:34s} {table[q]['平均スコア']}")

    print("\n--- 各物体に最も安定して当たる語 ---")
    for name in OBJECTS:
        ranked = sorted(((hits[q][name], q) for q in CANDIDATES if hits[q][name]), reverse=True)
        print(f"{name:8s} " + (", ".join(f"{q}({c}/{args.seeds})" for c, q in ranked[:3]) or "なし"))
    print(f"\n保存先: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
