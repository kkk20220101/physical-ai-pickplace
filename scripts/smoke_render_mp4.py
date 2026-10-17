#!/usr/bin/env python
"""ヘッドレス環境で MuJoCo をオフスクリーン描画し、MP4 を書き出せるかを実測する。

このサーバには DISPLAY が無いため、提出物（MP4動画）を作れるかどうかが
プロジェクト最大の前提リスクになる。実装に入る前にここだけを検証する。

使い方:
    MUJOCO_GL=egl python scripts/smoke_render_mp4.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

HEIGHT, WIDTH = 480, 640

# 外部アセットに依存しない最小シーン（振り子1本＋床）。描画経路だけを確かめる。
SCENE_XML = """
<mujoco>
  <visual><global offheight="720" offwidth="1280"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.85 0.9 1"/>
    <body name="arm" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="link" type="capsule" fromto="0 0 0 0 0 -0.6" size="0.05" rgba="0.2 0.5 0.9 1"/>
      <body name="tip" pos="0 0 -0.6">
        <geom name="ball" type="sphere" size="0.1" rgba="0.9 0.4 0.2 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def encode_mp4(frames, out_path: Path, fps: int) -> str:
    """フレーム列を MP4 にする。OpenCV を第1経路、ffmpeg を第2経路とする。"""
    try:
        import cv2

        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
        )
        if writer.isOpened():
            for f in frames:
                writer.write(f[:, :, ::-1])  # RGB -> BGR
            writer.release()
            if out_path.exists() and out_path.stat().st_size > 0:
                return "opencv(mp4v)"
    except Exception as exc:  # noqa: BLE001 - 第2経路へ落とすための捕捉
        print(f"[warn] OpenCV 経路が使えない: {exc}", file=sys.stderr)

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    if not Path(ffmpeg).exists():
        raise RuntimeError("MP4 を書き出す手段が無い（OpenCV も ffmpeg も不可）")

    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(Path(tmp) / f"{i:05d}.png")
        subprocess.run(
            [ffmpeg, "-y", "-framerate", str(fps), "-i", f"{tmp}/%05d.png",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
            check=True, capture_output=True,
        )
    return f"ffmpeg({ffmpeg})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/smoke/smoke_render.mp4")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    import mujoco

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    n_frames = int(args.seconds * args.fps)
    steps_per_frame = max(1, round(1.0 / args.fps / model.opt.timestep))

    frames = []
    t0 = time.perf_counter()
    for _ in range(n_frames):
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        renderer.update_scene(data)
        frames.append(renderer.render())
    render_sec = time.perf_counter() - t0

    encoder = encode_mp4(frames, out_path, args.fps)

    result = {
        "ok": out_path.exists() and out_path.stat().st_size > 0,
        "mujoco_version": mujoco.__version__,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "display": os.environ.get("DISPLAY") or "(none)",
        "frames": n_frames,
        "resolution": f"{WIDTH}x{HEIGHT}",
        "fps": args.fps,
        "render_sec": round(render_sec, 2),
        "render_fps": round(n_frames / render_sec, 1),
        "encoder": encoder,
        "out": str(out_path),
        "bytes": out_path.stat().st_size if out_path.exists() else 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
