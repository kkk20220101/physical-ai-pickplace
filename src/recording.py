"""記録層: 動画・イベントログ・設定を run 単位で残す。

守ること:
  - run_id は日時ベース。**同じフォルダに上書きしない**（結果の取り違え防止）。
  - 設定は実行時の全項目を書き出す。既定値に依存した記録を残さない。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

FFMPEG_FALLBACK = "ffmpeg"


def new_run_id(tag: str = "run") -> str:
    return f"{datetime.now():%Y%m%d_%H%M%S}_{tag}"


class Recorder:
    def __init__(self, out_dir: Path | str, camera_key: str = "agentview_image",
                 fps: int = 20, frame_stride: int = 1, overlay: bool = True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.camera_key = camera_key
        self.fps = fps
        self.frame_stride = frame_stride
        self.overlay = overlay
        self._frames: list[np.ndarray] = []
        self._events: list[dict] = []
        self._counter = 0
        self._banner: list[str] = []
        self._box: tuple | None = None

    # --- 画面に重ねる情報 -----------------------------------------------
    def set_overlay(self, lines: list[str], box: tuple | None = None) -> None:
        """音声無しでも分かるよう、指示文・現在の手順・検出枠を映像に出す。"""
        self._banner = lines
        self._box = tuple(box) if box is not None else None

    # --- 収集 -----------------------------------------------------------
    def capture(self, obs: dict) -> None:
        self._counter += 1
        if self._counter % self.frame_stride:
            return
        frame = obs.get(self.camera_key)
        if frame is None:
            return
        frame = np.flipud(frame).copy()  # robosuite の画像は上下反転している
        if self.overlay:
            frame = _draw_overlay(frame, self._banner, self._box)
        self._frames.append(frame)

    def log_event(self, **fields: Any) -> None:
        self._events.append(fields)

    # --- 書き出し -------------------------------------------------------
    def save_config(self, config: Any) -> Path:
        path = self.out_dir / "config.json"
        payload = asdict(config) if is_dataclass(config) else config
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return path

    def save_events(self) -> Path:
        path = self.out_dir / "events.jsonl"
        with path.open("w") as fp:
            for ev in self._events:
                fp.write(json.dumps(ev, ensure_ascii=False, default=_json_default) + "\n")
        return path

    def save_summary(self, summary: dict) -> Path:
        path = self.out_dir / "summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
        return path

    def save_video(self, name: str = "video.mp4") -> Path | None:
        """H.264 で書き出す。OpenCV の mp4v は再生環境を選ぶうえファイルが 4倍大きい（実測）。"""
        if not self._frames:
            return None
        path = self.out_dir / name
        height, width = self._frames[0].shape[:2]
        if _write_with_ffmpeg(self._frames, path, self.fps, width, height):
            return path
        _write_with_opencv(self._frames, path, self.fps, width, height)
        return path

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    @property
    def duration_sec(self) -> float:
        return len(self._frames) / self.fps if self.fps else 0.0


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"JSON にできない型: {type(value)}")


def _draw_overlay(frame: np.ndarray, lines: list[str],
                  box: tuple | None) -> np.ndarray:
    """検出枚と説明行を描く。cv2 は日本語を描けないので入力は英数字に限る。

    **記録フレームは提出動画で拡大される**（512→720）。
    細い文字は拡大補間で二重ににじむので、**下地を塗って太く描く**（DEC-37）。
    描画は記録用のフレームにだけ行う。**知覚モデルは obs を直接読むので結果に影響しない。**
    """
    if not lines and box is None:
        return frame
    try:
        import cv2
    except ImportError:
        return frame

    canvas = np.ascontiguousarray(frame)
    if box is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if lines:
        font, scale, thickness = cv2.FONT_HERSHEY_DUPLEX, 0.62, 2
        sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
        pad, step = 8, 30
        band_w = max(w for w, _ in sizes) + pad * 2
        band_h = step * len(lines) + pad
        band = canvas[0:band_h, 0:band_w].astype(np.float32) * 0.25
        canvas[0:band_h, 0:band_w] = band.astype(canvas.dtype)
        for i, line in enumerate(lines):
            origin = (pad, pad + 18 + i * step)
            cv2.putText(canvas, line, origin, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(canvas, line, origin, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas


def _write_with_opencv(frames, path: Path, fps: int, width: int, height: int) -> bool:
    try:
        import cv2

        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            return False
        for frame in frames:
            writer.write(frame[:, :, ::-1])
        writer.release()
        return path.exists() and path.stat().st_size > 0
    except Exception:
        return False


def _write_with_ffmpeg(frames, path: Path, fps: int, width: int, height: int) -> bool:
    """生フレームを ffmpeg に流し込んで H.264 にする（PNG に落とさないので速い）。"""
    ffmpeg = shutil.which("ffmpeg") or FFMPEG_FALLBACK
    if not Path(ffmpeg).exists():
        return False
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-framerate", str(fps), "-i", "-",
        "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", str(path),
    ]
    # stderr を PIPE にして読まないと、ffmpeg がそこへ書き続けたときに
    # パイプが埋まり、**こちらの stdin への書き込みが永久に止まる**。
    # 両方を同時に扱う代わりに、エラー出力は一時ファイルへ逃がす。
    with tempfile.TemporaryFile() as errlog:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errlog)
        try:
            for frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            proc.stdin.close()
            proc.wait(timeout=600)
        except Exception as error:
            proc.kill()
            proc.wait()
            print(f"ffmpeg での書き出しに失敗した（OpenCV へ戻る）: {error}")
            return False
        if proc.returncode != 0:
            errlog.seek(0)
            message = errlog.read().decode("utf-8", "replace").strip()[-500:]
            # 黙って OpenCV へ戻ると、**mp4v の 4倍大きいファイルができた理由が残らない**。
            print(f"ffmpeg が異常終了した（OpenCV へ戻る）: {message}")
            return False
    return path.exists() and path.stat().st_size > 0
