#!/usr/bin/env python
"""アブレーション: 何がタスク成功率に効いているかを、条件を1つずつ変えて測る。

比較する7条件（**初期配置は条件間で完全に揃える**。評価設計 §4-1 作法1）:

  C0  真値の位置 + 真値の向き        … 制御層の上限。ここが低いと他の議論ができない
  C0' 真値の位置 + 固定の向き        … 「向き合わせ」だけの寄与
  C1  推論の位置(視線交点) + 真値の向き … 「知覚AI」だけの寄与
  C1d 推論の位置(深度読取) + 真値の向き … 位置の求め方の寄与
  C2  推論の位置 + 固定の向き        … 向きの情報を失ったとき
  C3  推論の位置 + 推論の向き        … 知覚・姿勢がAI（真値を一切使わない）
  C4  言語計画 + 知覚AI             … 全層AI
  C5  C4 + 実行時の自己監視と再試行   … **提出構成**。C4 との差分が回復機構の寄与

使い方:
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt HF_HUB_DISABLE_XET=1
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 \
      python scripts/run_ablation.py --episodes 20
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

CONDITIONS = {
    "C0  真値位置+真値向き": {"perception": "ground_truth", "yaw_source": "truth"},
    "C0' 真値位置+固定向き": {"perception": "ground_truth", "yaw_source": "fixed"},
    "C1  推論位置(視線交点)+真値向き": {"perception": "model", "position_mode": "ray_plane", "yaw_source": "truth"},
    "C1d 推論位置(深度読取)+真値向き": {"perception": "model", "position_mode": "depth", "yaw_source": "truth"},
    "C2  推論位置+固定向き": {"perception": "model", "position_mode": "ray_plane", "yaw_source": "fixed"},
    "C3  推論位置+推論向き（知覚はAI）": {"perception": "model", "position_mode": "ray_plane", "yaw_source": "model"},
    "C4  言語計画+知覚AI（全層AI）": {"planner": "llm", "perception": "model", "position_mode": "ray_plane", "yaw_source": "model"},
    "C5  全層AI+自己監視と再試行": {"planner": "llm", "perception": "model", "position_mode": "ray_plane", "yaw_source": "model", "recovery": True},
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--camera-size", type=int, default=512)
    p.add_argument("--only", nargs="+", default=None,
                   help="条件名の先頭一致で絞る（例: C4）。既存結果へマージする")
    p.add_argument("--out", default=str(ROOT / "experiments" / "exp003_ablation" / "result.json"))
    args = p.parse_args()

    conditions = CONDITIONS
    if args.only:
        conditions = {k: v for k, v in CONDITIONS.items()
                      if any(k.startswith(prefix) for prefix in args.only)}
        if not conditions:
            print(f"条件が見つからない: {args.only}", file=sys.stderr)
            return 1

    # 全条件で同じ環境設定を使う（深度は知覚AIが要るので常に有効にしておく）
    env = make_env(EnvConfig(camera_height=args.camera_size, camera_width=args.camera_size,
                             camera_depths=True))

    results = {}
    for name, overrides in conditions.items():
        cfg = RunConfig(seed=args.seed, control=ControlConfig(), **overrides)
        episodes = [run_episode(env, cfg, episode_index=i) for i in range(args.episodes)]
        summary = summarize_episodes(episodes)
        summary["condition"] = name
        summary["settings"] = overrides
        summary["episodes"] = episodes
        results[name] = summary
        print(f"{name:32s} タスク {summary['episode_success_fraction']:>6s}  "
              f"手順 {summary['step_level']['step_success_fraction']:>6s}  "
              f"{summary['step_level']['ledger_check']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 一部の条件だけ走らせたときは、他の条件の結果を消さない。
    # ただし同じ条件コード（先頭トークン。例 "C3"）の旧結果は置き換える——
    # 条件の**表示名**を変えた後に旧名の行が残ると、同じ条件が2行になり
    # 図・動画の prefix 検索（startswith）が古い方を拾い得る。
    merged = {}
    if args.only and out.exists():
        merged = json.loads(out.read_text()).get("results", {})
        new_codes = {name.split()[0] for name in results}
        merged = {k: v for k, v in merged.items() if k.split()[0] not in new_codes}
    merged.update(results)
    # トップの episodes は「全条件で共通のときだけ」書く。--only で少数エピソードを
    # 併合したとき、20エピソードの他条件を含むファイルが "episodes": 5 を名乗らないように。
    per_condition = {s.get("n_episodes") for s in merged.values()}
    episodes_meta = per_condition.pop() if len(per_condition) == 1 else "mixed（各条件の n_episodes を見る）"
    out.write_text(json.dumps({"episodes": episodes_meta, "seed": args.seed,
                               "results": merged}, ensure_ascii=False, indent=2, default=str))

    print(f"\n{'条件':32s} {'タスク成功':>10s} {'手順成功':>10s}  主な失敗モード")
    for name, s in results.items():
        modes = s["step_level"]["failure_modes"]
        top = " ".join(f"{k}({v['count']})" for k, v in
                       sorted(modes.items(), key=lambda x: -x[1]["count"])[:3]) or "なし"
        print(f"{name:32s} {s['episode_success_fraction']:>10s} "
              f"{s['step_level']['step_success_fraction']:>10s}  {top}")

    for name, s in results.items():
        if "identification_accuracy_fraction" in s:
            print(f"\n{name} の対象同定: {s['identification_accuracy_fraction']} "
                  f"({s['identification_accuracy']:.0%})")
    for name, s in results.items():
        rec = s.get("recovery")
        if rec:
            print(f"\n{name} の再試行: 再探索 {rec['n_redetect']} 件 / 掴み直し {rec['n_regrasp']} 件 "
                  f"→ 救えた {rec['rescued_fraction']}（妥当性検査で却下 {rec['n_plausibility_rejections']} 件）")
    print(f"\n保存先: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
