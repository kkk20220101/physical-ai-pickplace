"""知覚層: オープン語彙の物体検出（OWLv2 / OWL-ViT の推論）。

なぜオープン語彙検出か（意思決定 DEC-15）:
  学習不要（推論のみ）で、**任意の語**で物体を探せる。
  「a carton of milk」のような語をそのまま投げられるので、言語指示と素直に繋がる。

既定値は実測で決めた（条件 P4。docs/30_評価/20260927_知覚AI_アブレーション結果.md）:
  モデル OWLv2 + 卓上の切り出しで **同定 18/20（90%）・取り違え 0件**。
  全体画像の OWL-ViT は 5/20（25%）で、**モデルと画角の両方が効く**。

このモジュールは真値を一切見ない。真値版（`ground_truth`）との比較がそのまま
アブレーション（AIあり/なし）になる。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

from src.device import default_device

# 社内プロキシ配下での取得に必要（未設定だと証明書検証と Xet CDN で落ちる）
os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

MODEL_ID = "google/owlv2-base-patch16-ensemble"

# 物体が置かれている左のテーブルだけを切り出す比率 (x1, y1, x2, y2)。
# 切り出さないと物体が小さく写り、同定率が 90% → 55% まで落ちる（実測）。
TABLE_CROP: tuple[float, float, float, float] = (0.02, 0.25, 0.56, 0.72)

# 物体名 → 検出に使う英語の問い合わせ語。**意味語で統一する**
# （「a red cylinder」のような見た目語に逃げると「言語で指定した」と言えなくなる）。
TEXT_QUERIES: dict[str, list[str]] = {
    "Milk": ["a carton of milk", "a milk box"],
    "Bread": ["a loaf of bread", "a piece of bread"],
    "Cereal": ["a cereal box", "a box of cereal"],
    "Can": ["a soda can", "a can of soda"],
}

# 見つからなかったときだけ追加で投げる語（段階的再探索。`escalated_config` が使う）。
# **ここも意味語だけ**にする——見た目語を入れたら主張が崩れるのは初回と同じである。
RETRY_QUERIES: dict[str, list[str]] = {
    "Milk": ["milk", "a milk carton on a table", "a dairy carton"],
    "Bread": ["a loaf", "a bun", "bread", "a bread loaf on a table"],
    "Cereal": ["a cardboard box", "a breakfast cereal package"],
    "Can": ["a can", "a beverage can", "an aluminum can on a table"],
}

# 段階的再探索の段数。**段0は従来と完全に同じ振る舞い**（既存の数値を動かさない）。
# 実測（`scripts/probe_redetect.py`・20エピソード）: 既定で未検出だった Bread 4件のうち、
# 語の追加だけで 1/4、閾値を下げるだけで 3/4、**両方で 4/4** が検出できた。
# ただし緩めると**「未検出」が「取り違え」に化ける**（1件は 32cm 先の別物を指した）。
# ∴ 緩めた候補は必ず `is_plausible()` を通す。
ESCALATION_LEVELS = 3


def escalated_config(cfg: "PerceptionConfig", level: int) -> "PerceptionConfig":
    """再探索の段に応じた設定を返す。**level<=0 は cfg をそのまま返す**。

    段1: 意味語を追加する（閾値はそのまま）
    段2: さらに検出閾値を下げる
    """
    if level <= 0:
        return cfg
    queries = {name: list(words) + RETRY_QUERIES.get(name, [])
               for name, words in cfg.queries.items()}
    threshold = cfg.score_threshold if level < 2 else 0.005
    return replace(cfg, queries=queries, score_threshold=threshold)


@dataclass
class PerceptionConfig:
    model_id: str = MODEL_ID
    score_threshold: float = 0.05
    device: str = field(default_factory=default_device)
    queries: dict[str, list[str]] = field(default_factory=lambda: dict(TEXT_QUERIES))
    # 物体が画面に小さく写ると検出が落ちる。切り出して拡大する範囲 (x1,y1,x2,y2) の比率。
    crop: tuple[float, float, float, float] | None = TABLE_CROP
    # 位置の求め方: "depth"（深度画像を読む）／"ray_plane"（同定結果から高さを知って平面と交える）
    position_mode: str = "ray_plane"
    # 把持姿勢の向きを検出結果から推定するか。
    # mode は "top_face"（採用）か "minrect"（**ランダム以下。使わない**）
    estimate_yaw: bool = False
    yaw_mode: str = "top_face"


@lru_cache(maxsize=4)
def _load(model_id: str, device: str):
    import torch
    import transformers

    if "owlv2" in model_id:
        processor_cls, model_cls = transformers.Owlv2Processor, transformers.Owlv2ForObjectDetection
    else:
        processor_cls, model_cls = transformers.OwlViTProcessor, transformers.OwlViTForObjectDetection

    processor = processor_cls.from_pretrained(model_id)
    model = model_cls.from_pretrained(model_id).to(device).eval()
    torch.set_grad_enabled(False)
    return processor, model


def _crop_box(cfg: PerceptionConfig, height: int, width: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = cfg.crop
    return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)


def detect(image: np.ndarray, cfg: PerceptionConfig) -> dict[str, dict]:
    """画像から各物体を1件ずつ探す。戻り値は 物体名 → {score, bbox, query}。

    bbox は**元画像の座標**で返す（切り出しを使っても呼び出し側は意識しなくてよい）。
    見つからなかった物体はキー自体を持たない（＝未検出 F-3 として扱える）。
    """
    import torch
    from PIL import Image

    processor, model = _load(cfg.model_id, cfg.device)

    labels = list(cfg.queries)
    prompts = [q for name in labels for q in cfg.queries[name]]
    owner = [name for name in labels for _ in cfg.queries[name]]

    offset_x = offset_y = 0
    if cfg.crop is not None:
        offset_x, offset_y, x2, y2 = _crop_box(cfg, *image.shape[:2])
        image = image[offset_y:y2, offset_x:x2]

    pil = Image.fromarray(image)
    inputs = processor(text=[prompts], images=pil, return_tensors="pt").to(cfg.device)
    outputs = model(**inputs)
    target = torch.tensor([[pil.height, pil.width]], device=cfg.device)
    # transformers 5.x で post_process_object_detection から改名された
    results = processor.post_process_grounded_object_detection(
        outputs, threshold=cfg.score_threshold, target_sizes=target
    )[0]

    best: dict[str, dict] = {}
    for score, box, idx in zip(results["scores"], results["boxes"], results["labels"]):
        name = owner[int(idx)]
        score = float(score)
        if score > best.get(name, {"score": -1.0})["score"]:
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            best[name] = {
                "score": score,
                "bbox": [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
                "query": prompts[int(idx)],
            }
    return best


# --- 2D → 3D ------------------------------------------------------------
# robosuite の camera_utils は h5py を要求するため、必要な変換だけ自前で持つ。

def camera_intrinsics(env, camera_name: str, height: int, width: int) -> tuple[float, float, float]:
    """(焦点距離[px], 主点x, 主点y)。MuJoCo は垂直画角 fovy で定義される。"""
    cam_id = env.sim.model.camera_name2id(camera_name)
    fovy = float(env.sim.model.cam_fovy[cam_id])
    focal = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    return focal, width / 2.0, height / 2.0


def real_depth(env, depth_buffer: np.ndarray) -> np.ndarray:
    """MuJoCo の深度バッファ（0-1 の非線形値）を実距離 [m] に直す。"""
    extent = env.sim.model.stat.extent
    near = env.sim.model.vis.map.znear * extent
    far = env.sim.model.vis.map.zfar * extent
    return near / (1.0 - depth_buffer * (1.0 - near / far))


def pixel_to_world(env, camera_name: str, u: float, v: float, depth_m: float,
                   height: int, width: int) -> np.ndarray:
    """画素 (u, v) と実距離からワールド座標を求める。

    MuJoCo のカメラ座標系は **-Z が前方・+Y が上**。画像の v は下向きに増えるので符号が反転する。
    """
    focal, cx, cy = camera_intrinsics(env, camera_name, height, width)
    cam_id = env.sim.model.camera_name2id(camera_name)
    cam_pos = np.array(env.sim.data.cam_xpos[cam_id])
    cam_mat = np.array(env.sim.data.cam_xmat[cam_id]).reshape(3, 3)

    point_cam = np.array([
        (u - cx) * depth_m / focal,
        -(v - cy) * depth_m / focal,
        -depth_m,
    ])
    return cam_pos + cam_mat @ point_cam


def pixel_to_plane(env, camera_name: str, u: float, v: float, plane_z: float,
                   height: int, width: int) -> np.ndarray:
    """画素を通る視線と、高さ plane_z の水平面との交点を求める。

    深度を読むと「物体の手前の面」の座標になり、中心から系統的にずれる。
    **物体が何か分かっていれば高さは既知**なので、その高さの面と交えるほうが正確になる。
    """
    focal, cx, cy = camera_intrinsics(env, camera_name, height, width)
    cam_id = env.sim.model.camera_name2id(camera_name)
    cam_pos = np.array(env.sim.data.cam_xpos[cam_id])
    cam_mat = np.array(env.sim.data.cam_xmat[cam_id]).reshape(3, 3)

    direction = cam_mat @ np.array([(u - cx) / focal, -(v - cy) / focal, -1.0])
    if abs(direction[2]) < 1e-9:
        raise ValueError("視線が水平で平面と交わらない")
    t = (plane_z - cam_pos[2]) / direction[2]
    return cam_pos + t * direction


def estimate_yaw(env, camera_name: str, depth_m: np.ndarray, bbox: list[float],
                 table_z: float, height: int, width: int,
                 clearance: float = 0.02, min_pixels: int = 40,
                 mode: str = "top_face", top_band: float = 0.015) -> float | None:
    """検出枠の中で物体の画素を拾い、その広がりの向きを把持姿勢に使う。

    真値を一切見ない（深度画像はセンサ出力として扱う）。

    mode:
      "minrect"  … 机より上の点全体の最小外接矩形
      "top_face" … **上面付近の点だけ**の最小外接矩形
                   （側面はカメラ向きに引きずられるので除く、という狙い）

    画素が少なすぎて向きを決められないときは **None** を返す（推定したふりをしない）。
    """
    import cv2

    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, width - 1), min(y2, height - 1)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None

    focal, cx, cy = camera_intrinsics(env, camera_name, height, width)
    cam_id = env.sim.model.camera_name2id(camera_name)
    cam_pos = np.array(env.sim.data.cam_xpos[cam_id])
    cam_mat = np.array(env.sim.data.cam_xmat[cam_id]).reshape(3, 3)

    vs, us = np.mgrid[y1:y2 + 1, x1:x2 + 1]
    d = depth_m[y1:y2 + 1, x1:x2 + 1]
    points_cam = np.stack([(us - cx) * d / focal, -(vs - cy) * d / focal, -d], axis=-1)
    points_world = points_cam.reshape(-1, 3) @ cam_mat.T + cam_pos

    on_object = points_world[:, 2] > table_z + clearance
    if int(on_object.sum()) < min_pixels:
        return None
    points = points_world[on_object]

    if mode == "top_face":
        points = points[points[:, 2] > points[:, 2].max() - top_band]
        if len(points) < min_pixels:
            return None

    xy = points[:, :2].astype(np.float32)
    (_, _), (_, _), angle_deg = cv2.minAreaRect(xy)
    return float(np.deg2rad(angle_deg))


def locate_from_image(env, rgb: np.ndarray, depth_buffer: np.ndarray, camera_name: str,
                      cfg: PerceptionConfig, object_center_z: dict[str, float] | None = None,
                      table_z: float | None = None) -> dict[str, dict]:
    """検出 → 3D 座標（必要なら向きも）まで通す。知覚層の入口はこれ1つ。

    object_center_z: 物体名 → 中心の高さ [m]。`position_mode="ray_plane"` のときに使う。
    これは**物体の形状情報（カタログ値）であって、その場の真値位置ではない。**
    table_z: 机の高さ。向き推定で物体と机を切り分けるのに使う。
    """
    height, width = rgb.shape[:2]
    detections = detect(rgb, cfg)
    depth_m = real_depth(env, depth_buffer) if depth_buffer is not None else None

    out: dict[str, dict] = {}
    for name, det in detections.items():
        x1, y1, x2, y2 = det["bbox"]
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if cfg.position_mode == "ray_plane" and object_center_z and name in object_center_z:
            position = pixel_to_plane(env, camera_name, u, v, object_center_z[name], height, width)
        else:
            ui = int(np.clip(u, 0, width - 1))
            vi = int(np.clip(v, 0, height - 1))
            position = pixel_to_world(env, camera_name, u, v, float(depth_m[vi, ui]), height, width)

        yaw = None
        if cfg.estimate_yaw and depth_m is not None and table_z is not None:
            yaw = estimate_yaw(env, camera_name, depth_m, det["bbox"], table_z, height, width,
                               mode=cfg.yaw_mode)

        out[name] = {**det, "pixel": [u, v], "position_3d": position, "yaw": yaw}
    return out


# --- 妥当性検査（再探索を緩めたときの歯止め） ---------------------------------
# 検出の閾値を下げると「見つからない」が「取り違え」に化ける。取り違えは未検出より悪い
# （ロボットが別の物を掴んで運んでしまう）。∴ 緩めた候補には必ずこの門を通す。
#
# **ここで使ってよいのは、事前に分かっている固定物の幾何と、他の検出結果だけ**である。
# 物体の真値位置は使わない（使ったら「見て動いた」と言えなくなる）。

# 物体が置かれている台は固定物であり、その範囲は設置時に分かっている。
# 既存コードが机の高さ（`env.bin2_pos[2]`）を使っているのと同じ種類の情報。
PICKUP_TABLE_MARGIN = 0.05   # 台の縁からこの距離まではみ出しても許す [m]
MIN_OBJECT_SEPARATION = 0.05  # 他の物体の検出位置とこれ以上離れていること [m]


def pickup_table_bounds(env, margin: float = PICKUP_TABLE_MARGIN):
    """物体が載っている台の xy 範囲 (low, high)。**固定物の幾何であり真値ではない。**"""
    center = np.asarray(env.bin1_pos, dtype=float)[:2]
    half = np.asarray(env.bin_size, dtype=float)[:2] / 2.0 + margin
    return center - half, center + half


def is_plausible(env, position, others: dict[str, dict] | None = None,
                 margin: float = PICKUP_TABLE_MARGIN,
                 min_separation: float = MIN_OBJECT_SEPARATION) -> tuple[bool, str]:
    """緩めた検出結果が物理的にありえるかを判定する。(可否, 理由) を返す。

    1. 台の上にあること（台の外を指していたら背景か別の家具を拾っている）
    2. 既に受理した他の物体の位置と重ならないこと（同じ物を2つの名前で指していない）
    """
    if position is None:
        return False, "位置が無い"
    low, high = pickup_table_bounds(env, margin)
    xy = np.asarray(position, dtype=float)[:2]
    if not (low[0] <= xy[0] <= high[0] and low[1] <= xy[1] <= high[1]):
        return False, f"台の外 ({xy[0]:.3f}, {xy[1]:.3f})"
    for name, det in (others or {}).items():
        other = det.get("position_3d")
        if other is None:
            continue
        distance = float(np.linalg.norm(xy - np.asarray(other, dtype=float)[:2]))
        if distance < min_separation:
            return False, f"{name} の検出位置と {distance * 100:.1f}cm しか離れていない"
    return True, "ok"
