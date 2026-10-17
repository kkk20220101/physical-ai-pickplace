#!/usr/bin/env python
"""把持姿勢の向きをAIの検出結果だけから求められるかを実測する。

背景: 位置はAIから求めているが、**向きは真値か固定値**のままだった。
「完全にAIだけで動いた」と言えるかはここに掛かっている（U-10 / S-07）。

**当てずっぽうの基準線と必ず比べる。**
向きが一様ランダムなら、90度対称性を畳んだ誤差は [0, 45] の一様分布になり、
中央値は約22.5度・15度以内は約33%になる。**これを超えなければ推定できていない。**

使い方:
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt HF_HUB_DISABLE_XET=1
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 \
      python scripts/probe_yaw.py --seeds 10
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
from src.control.primitives import wrap_to_jaw_symmetry  # noqa: E402
from src.envs.pickplace import EnvConfig, make_env, object_half_height, object_yaw  # noqa: E402
from src.perception.owlvit_detector import PerceptionConfig, locate_from_image  # noqa: E402

OBJECTS = ["Milk", "Bread", "Cereal", "Can"]
CAMERA = "agentview"


def measure(env, cfg: PerceptionConfig, table_z: float, seeds: int) -> dict:
    errors: dict[str, list[float]] = defaultdict(list)
    no_estimate: dict[str, int] = defaultdict(int)
    not_detected: dict[str, int] = defaultdict(int)

    for seed in range(seeds):
        np.random.seed(seed)
        obs = env.reset()
        rgb = np.flipud(obs[f"{CAMERA}_image"]).copy()
        depth = np.flipud(obs[f"{CAMERA}_depth"][:, :, 0]).copy()
        center_z = {n: table_z + object_half_height(env, n) for n in OBJECTS}
        found = locate_from_image(env, rgb, depth, CAMERA, cfg, center_z, table_z)

        for name in OBJECTS:
            if name not in found:
                not_detected[name] += 1
            elif found[name]["yaw"] is None:
                no_estimate[name] += 1
            else:
                # 90度対称性を畳んでから比較する
                errors[name].append(float(abs(np.rad2deg(
                    wrap_to_jaw_symmetry(found[name]["yaw"] - object_yaw(env, name))))))

    flat = [e for v in errors.values() for e in v]
    return {
        "per_object": {
            n: {"推定できた": len(errors[n]), "向き推定不可": no_estimate[n],
                "未検出": not_detected[n],
                "誤差中央値_deg": round(float(np.median(errors[n])), 1) if errors[n] else None,
                "検算": f"{len(errors[n])}+{no_estimate[n]}+{not_detected[n]}={seeds} "
                        f"{'✔' if len(errors[n]) + no_estimate[n] + not_detected[n] == seeds else '✘'}"}
            for n in OBJECTS
        },
        "全体": {
            "推定できた": len(flat), "母数": seeds * len(OBJECTS),
            "誤差中央値_deg": round(float(np.median(flat)), 1) if flat else None,
            "15度以内": f"{sum(1 for e in flat if e <= 15)}/{len(flat)}" if flat else "0/0",
            "15度以内の割合": round(sum(1 for e in flat if e <= 15) / len(flat), 3) if flat else None,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--device", default=default_device())
    p.add_argument("--modes", nargs="+", default=["minrect", "top_face"])
    p.add_argument("--out", default=str(ROOT / "experiments" / "exp004_yaw_estimation" / "result.json"))
    args = p.parse_args()

    env = make_env(EnvConfig(camera_height=args.size, camera_width=args.size, camera_depths=True))
    table_z = float(env.bin2_pos[2])

    modes = {}
    for mode in args.modes:
        cfg = PerceptionConfig(device=args.device, estimate_yaw=True, yaw_mode=mode)
        modes[mode] = measure(env, cfg, table_z, args.seeds)
        print(f"{mode} 完了", flush=True)

    # 当てずっぽうの基準線。**これを超えなければ推定できていない。**
    rng = np.random.default_rng(0)
    random_errors = np.abs([np.rad2deg(wrap_to_jaw_symmetry(a))
                            for a in rng.uniform(-np.pi, np.pi, 4000)])
    baseline = {
        "誤差中央値_deg": round(float(np.median(random_errors)), 1),
        "15度以内の割合": round(float((random_errors <= 15).mean()), 3),
    }

    summary = {"seeds": args.seeds, "modes": modes, "ランダムの基準線": baseline}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n{'条件':14s} {'推定できた':>10s} {'誤差中央値':>10s} {'15度以内':>10s} {'割合':>8s}")
    for mode, res in modes.items():
        w = res["全体"]
        print(f"{mode:14s} {w['推定できた']:>10d} {str(w['誤差中央値_deg']):>10s} "
              f"{w['15度以内']:>10s} {str(w['15度以内の割合']):>8s}")
    print(f"{'ランダム(基準)':14s} {'—':>10s} {baseline['誤差中央値_deg']:>10} "
          f"{'—':>10s} {baseline['15度以内の割合']:>8}")
    print(f"\n保存先: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
