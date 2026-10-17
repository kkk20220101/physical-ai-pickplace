#!/usr/bin/env python
"""知覚モデル（採用構成の OWLv2 ＋ 卓上切り出し）が使い物になるかを実測する。
モデル×画角の比較は scripts/compare_perception.py が担う。

判定するのは3つ:
  1. 4物体を区別できるか（同定の正解率）
  2. 3D位置の誤差は何センチか
  3. **決定的テスト T-1〜T-3**: モデルが本当に画像と語を使っているか
     （過去に「重みが壊れていてどんな入力でも同じ出力を返すモデル」を掴んだ事例があるため必須）

使い方:
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt HF_HUB_DISABLE_XET=1
    MUJOCO_GL=egl python scripts/probe_perception.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.device import default_device  # noqa: E402
from src.envs.pickplace import EnvConfig, make_env, object_position  # noqa: E402
from src.perception.owlvit_detector import (  # noqa: E402
    PerceptionConfig,
    detect,
    locate_from_image,
)

OBJECTS = ["Milk", "Bread", "Cereal", "Can"]
CAMERA = "agentview"


def grab(env, obs, size):
    """RGB と深度を取り出す。robosuite の画像は上下反転しているので戻す。"""
    rgb = np.flipud(obs[f"{CAMERA}_image"]).copy()
    depth = np.flipud(obs[f"{CAMERA}_depth"][:, :, 0]).copy()
    return rgb, depth


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--device", default=default_device())
    p.add_argument("--out", default=str(ROOT / "experiments" / "exp002_perception_probe" / "result.json"))
    args = p.parse_args()

    cfg = PerceptionConfig(score_threshold=args.threshold, device=args.device)

    detected = {o: 0 for o in OBJECTS}
    errors: dict[str, list[float]] = {o: [] for o in OBJECTS}
    xy_errors: dict[str, list[float]] = {o: [] for o in OBJECTS}
    per_seed = []

    env = None
    # 環境は1つだけ作る。シードごとに作り直すと EGL コンテキストが解放されず積み上がる。
    env = make_env(EnvConfig(camera_height=args.size, camera_width=args.size,
                             camera_depths=True))
    for seed in range(args.seeds):
        np.random.seed(seed)
        obs = env.reset()
        if f"{CAMERA}_depth" not in obs:
            print("深度が取れない。環境生成時に camera_depths=True が要る", file=sys.stderr)
            return 1
        rgb, depth = grab(env, obs, args.size)
        found = locate_from_image(env, rgb, depth, CAMERA, cfg)

        row = {"seed": seed, "objects": {}}
        for name in OBJECTS:
            truth = object_position(env, name)
            if name not in found:
                row["objects"][name] = {"detected": False}
                continue
            detected[name] += 1
            est = np.asarray(found[name]["position_3d"])
            err = float(np.linalg.norm(est - truth))
            err_xy = float(np.linalg.norm(est[:2] - truth[:2]))
            errors[name].append(err)
            xy_errors[name].append(err_xy)
            row["objects"][name] = {
                "detected": True, "score": round(found[name]["score"], 3),
                "query": found[name]["query"],
                "error_m": round(err, 4), "error_xy_m": round(err_xy, 4),
                "est": [round(v, 3) for v in est.tolist()],
                "truth": [round(v, 3) for v in truth.tolist()],
            }
        per_seed.append(row)
        print(f"[seed{seed}] " + " ".join(
            f"{n}:{'✓' if row['objects'][n]['detected'] else '✗'}"
            f"{row['objects'][n].get('error_m', '')}" for n in OBJECTS), flush=True)

    # --- 決定的テスト -----------------------------------------------------
    rgb, depth = grab(env, env.reset(), args.size)
    base = detect(rgb, cfg)
    zeroed = detect(np.zeros_like(rgb), cfg)
    swapped_cfg = PerceptionConfig(
        score_threshold=args.threshold, device=args.device,
        queries={"Milk": ["a soda can"], "Can": ["a carton of milk"]},
    )
    swapped = detect(rgb, swapped_cfg)

    decisive = {
        "T-1_画像ゼロ化": {
            "通常の検出数": len(base), "ゼロ画像の検出数": len(zeroed),
            "判定": "合格" if len(zeroed) < len(base) else "不合格（画像を使っていない疑い）",
        },
        "T-2_語の入れ替え": {
            "説明": "Milk に『缶』、Can に『牛乳』の語を与えた",
            "元の画素": {k: [round(x) for x in v["pixel"]] for k, v in
                         locate_from_image(env, rgb, depth, CAMERA, cfg).items()
                         if k in ("Milk", "Can")},
            "入替後の画素": {k: [round(x) for x in v["bbox"][:2]] for k, v in swapped.items()},
            "判定": "合格" if swapped and any(
                abs(swapped[k]["bbox"][0] - base[k]["bbox"][0]) > 5
                for k in swapped if k in base) else "要確認",
        },
        "T-3_出力の固定確認": {
            "異なるシード間でスコアが変動するか": len({
                r["objects"][n].get("score") for r in per_seed for n in OBJECTS
                if r["objects"][n].get("detected")}) > 1,
        },
    }

    summary = {
        "model": cfg.model_id, "seeds": args.seeds, "image_size": args.size,
        "threshold": args.threshold,
        "detection": {
            n: {"fraction": f"{detected[n]}/{args.seeds}",
                "median_error_m": round(float(np.median(errors[n])), 4) if errors[n] else None,
                "median_error_xy_m": round(float(np.median(xy_errors[n])), 4) if xy_errors[n] else None}
            for n in OBJECTS
        },
        "decisive_tests": decisive,
        "per_seed": per_seed,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    print("\n物体      検出      3D誤差中央値   XY誤差中央値")
    for n in OBJECTS:
        d = summary["detection"][n]
        print(f"{n:8s} {d['fraction']:>6s}   {str(d['median_error_m']):>10s} m  {str(d['median_error_xy_m']):>10s} m")
    print("\n決定的テスト:")
    for k, v in decisive.items():
        print(f"  {k}: {v.get('判定', v)}")
    print(f"\n保存先: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
