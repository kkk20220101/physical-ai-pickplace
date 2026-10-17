#!/usr/bin/env python
"""仕分けエピソードを実行し、動画・ログ・指標を outputs/<run_id>/ に残す。

使い方:
    MUJOCO_GL=egl python scripts/run_episode.py \
        --episodes 3 --tag l1_baseline
    # 指示文を変えるとき（計画層が解釈できる語＝CATEGORY_RULES / 言語モデルの範囲で）:
    #   --planner llm --instruction "ジュースとミルクを左に寄せて"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.control.primitives import ControlConfig  # noqa: E402
from src.envs.pickplace import EnvConfig, make_env  # noqa: E402
from src.eval import summarize_episodes  # noqa: E402
from src.pipeline import RunConfig, run_episode  # noqa: E402
from src.recording import Recorder, new_run_id  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--instruction", default="飲み物を左の箱へ、食べ物を右の箱へ")
    p.add_argument("--perception", default="ground_truth", choices=["ground_truth", "model"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="run")
    p.add_argument("--camera-size", type=int, default=512)
    p.add_argument("--frame-stride", type=int, default=2, help="動画に残すフレームの間引き")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--yaw-source", default="truth", choices=["truth", "fixed", "model"],
                   help="把持姿勢の向きの出どころ。model にすると真値を一切使わない")
    p.add_argument("--planner", default="rule", choices=["rule", "llm"],
                   help="計画層。llm にすると指示文を言語モデルが解釈する")
    p.add_argument("--recovery", action="store_true",
                   help="実行時の自己監視と再試行を有効にする（提出構成 C5）")
    p.add_argument("--outputs", default=str(ROOT / "outputs"))
    args = p.parse_args()

    env_cfg = EnvConfig(camera_height=args.camera_size, camera_width=args.camera_size,
                        camera_depths=(args.perception == "model"), seed=args.seed)
    run_cfg = RunConfig(
        instruction=args.instruction,
        planner=args.planner,
        perception=args.perception,
        yaw_source=args.yaw_source,
        seed=args.seed,
        control=ControlConfig(),
        recovery=args.recovery,
    )

    run_id = new_run_id(args.tag)
    out_dir = Path(args.outputs) / run_id
    recorder = None if args.no_video else Recorder(
        out_dir, fps=env_cfg.control_freq, frame_stride=args.frame_stride
    )

    env = make_env(env_cfg)
    episodes = []
    for i in range(args.episodes):
        ep = run_episode(env, run_cfg, recorder=recorder, episode_index=i)
        episodes.append(ep)
        print(f"[ep{i}] success={ep['success']} steps={ep['n_env_steps']} "
              f"details={[(s['target'], s['success'], s['failure_mode']) for s in ep['steps']]}",
              flush=True)
    summary = summarize_episodes(episodes)
    summary["run_id"] = run_id
    # 動画のどこがどのエピソードかを後から特定できるよう、境界を残す
    fps, stride = env_cfg.control_freq, args.frame_stride
    cursor = 0.0
    timeline = []
    for ep in episodes:
        seconds = ep["n_env_steps"] / stride / fps
        timeline.append({"episode": ep["episode"], "success": ep["success"],
                         "n_env_steps": ep["n_env_steps"],
                         "start_sec": round(cursor, 2), "duration_sec": round(seconds, 2)})
        cursor += seconds
    summary["timeline"] = timeline

    out_dir.mkdir(parents=True, exist_ok=True)
    if recorder:
        recorder.save_config({"env": asdict(env_cfg), "run": asdict(run_cfg), "argv": vars(args)})
        recorder.save_events()
        recorder.save_summary(summary)
        video = recorder.save_video()
        summary["video"] = str(video) if video else None
        summary["video_duration_sec"] = round(recorder.duration_sec, 1)
        recorder.save_summary(summary)
    else:
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n出力先: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
