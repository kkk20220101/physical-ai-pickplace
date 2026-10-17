#!/usr/bin/env python
"""F-3（未検出）が「探し直せば見つかる」ものかを確かめる書き捨ての探り。

C4 で Bread が 7/20 回まったく検出されなかった（score 0.0）。
これが (a) 閾値のすぐ下にいるのか (b) 語が合っていないのか (c) 切り出しから外れているのか
を切り分ける。**(a)(b)(c) のどれかなら再探索で救える。**
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
from src.envs.pickplace import EnvConfig, make_env, object_position  # noqa: E402
from src.perception.owlvit_detector import (  # noqa: E402
    TABLE_CROP, PerceptionConfig, detect,
)

# 既定の語（意味語2つ）に対し、追加で試す語。すべて**意味語**で、見た目語には逃げない。
EXTRA_BREAD_QUERIES = [
    "bread", "a bread roll", "a bun", "a baguette", "a loaf", "food", "a bread loaf on a table",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--target", default="Bread")
    p.add_argument("--out", default=str(ROOT / "experiments" / "exp009_recovery_probe" / "detect_probe.json"))
    args = p.parse_args()

    env = make_env(EnvConfig(camera_depths=True))
    rows = []

    for ep in range(args.episodes):
        np.random.seed(ep)  # run_ablation と同じ seed 規則（seed=0 → 0*1000+ep）
        prim = Primitives(env, ControlConfig())
        prim.reset()
        rgb = np.flipud(prim.obs["agentview_image"]).copy()

        base_cfg = PerceptionConfig(crop=TABLE_CROP)
        base = detect(rgb, base_cfg)
        base_score = base.get(args.target, {}).get("score", 0.0)

        row = {"episode": ep, "base_score": round(base_score, 4)}

        # (a) 閾値を下げたら出るか
        low_cfg = PerceptionConfig(crop=TABLE_CROP, score_threshold=0.005)
        low = detect(rgb, low_cfg)
        row["low_thresh_score"] = round(low.get(args.target, {}).get("score", 0.0), 4)

        # (b) 語を増やしたら出るか（閾値は既定のまま）
        q = dict(PerceptionConfig().queries)
        q[args.target] = q[args.target] + EXTRA_BREAD_QUERIES
        more_cfg = PerceptionConfig(crop=TABLE_CROP, queries=q)
        more = detect(rgb, more_cfg)
        row["more_queries_score"] = round(more.get(args.target, {}).get("score", 0.0), 4)
        row["more_queries_word"] = more.get(args.target, {}).get("query")

        # (c) 切り出しを広げたら出るか
        wide_cfg = PerceptionConfig(crop=(0.0, 0.15, 0.68, 0.85))
        wide = detect(rgb, wide_cfg)
        row["wide_crop_score"] = round(wide.get(args.target, {}).get("score", 0.0), 4)

        # (b)+(a) 併用
        both_cfg = PerceptionConfig(crop=TABLE_CROP, queries=q, score_threshold=0.005)
        both = detect(rgb, both_cfg)
        row["both_score"] = round(both.get(args.target, {}).get("score", 0.0), 4)
        row["both_word"] = both.get(args.target, {}).get("query")

        # 位置が合っているか（救えても取り違えでは意味がない）
        truth = object_position(env, args.target)
        for key, res in (("more_queries", more), ("both", both)):
            det = res.get(args.target)
            if det is None:
                row[f"{key}_pos_err"] = None
                continue
            from src.envs.pickplace import object_half_height
            from src.perception.owlvit_detector import pixel_to_plane
            x1, y1, x2, y2 = det["bbox"]
            table_z = float(env.bin2_pos[2])
            center_z = table_z + object_half_height(env, args.target)
            pos = pixel_to_plane(env, "agentview", (x1 + x2) / 2, (y1 + y2) / 2,
                                 center_z, *rgb.shape[:2])
            row[f"{key}_pos_err"] = round(float(np.linalg.norm(pos[:2] - truth[:2])), 4)

        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"target": args.target, "rows": rows},
                              ensure_ascii=False, indent=2))

    miss = [r for r in rows if r["base_score"] == 0.0]
    print(f"\n既定で未検出: {len(miss)}/{len(rows)}")
    for key in ("low_thresh_score", "more_queries_score", "wide_crop_score", "both_score"):
        saved = sum(1 for r in miss if r[key] > 0)
        print(f"  {key:22s} で検出できた: {saved}/{len(miss)}")
    ok = [r for r in miss if r.get("both_pos_err") is not None and r["both_pos_err"] < 0.08]
    print(f"  うち位置も正しい（<8cm）: {len(ok)}/{len(miss)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
