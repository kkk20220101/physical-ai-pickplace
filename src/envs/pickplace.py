"""robosuite の PickPlace 環境を、言語指示による仕分けタスクとして使う。

PickPlace を選んだ理由（意思決定 DEC-08）:
  卓上に Milk / Bread / Cereal / Can の4物体があり、それぞれ実世界の名前を持つ。
  「飲み物を左の箱へ」のような**カテゴリ指示**と素直に接続できる。

タスク設計の実測上の制約（意思決定 DEC-09 / DEC-10。根拠は docs/30_評価/）:
  - Cereal は幅8cm で Panda のグリッパ開口幅とほぼ同じ。把持点をどう動かしても 2/5 しか
    持ち上がらない。**調整では治らないので操作対象から外し、妨害物として置く。**
  - 右奥の箱 (0.198, 0.402) はアームの可動範囲外（残差 5.5cm）。**手前2つの箱だけを使う。**
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

OBJECT_TO_BIN_ID: dict[str, int] = {"Milk": 0, "Bread": 1, "Cereal": 2, "Can": 3}

# 把持点の z 補正 [m]。experiments/exp001_grasp_sweep の実測で決めた値。
GRASP_Z_OFFSET: dict[str, float] = {
    "Milk": 0.02,    # 4/5
    "Bread": -0.01,  # 4/5
    "Can": 0.00,     # 5/5
    "Cereal": 0.00,  # 2/5（どの水準でも 2/5。開口幅の限界で調整では治らない）
}

# 配置先の呼び名 → 箱ID。**可動範囲内の手前2つだけ**を使う（DEC-10）。
DESTINATION_TO_BIN_ID: dict[str, int] = {"左の箱": 0, "右の箱": 1}

# 日本語カテゴリ → (対象物体, 配置先)。計画層（L2 の固定ルール版）が使う。
CATEGORY_RULES: dict[str, tuple[list[str], str]] = {
    "飲み物": (["Milk", "Can"], "左の箱"),
    "食べ物": (["Bread"], "右の箱"),
}

# 操作対象ではないがシーンに存在する物体。知覚層・計画層にとっての妨害物。
DISTRACTOR_OBJECTS: list[str] = ["Cereal"]


@dataclass
class EnvConfig:
    """環境設定。既定値をここ以外に書かない（設定の二重管理を作らない）。"""

    robot: str = "Panda"
    camera_names: list[str] = field(default_factory=lambda: ["agentview"])
    camera_height: int = 512
    camera_width: int = 512
    control_freq: int = 20
    horizon: int = 8000
    seed: int | None = None
    # 知覚層の 2D→3D 変換に必要。**真値を使う条件では不要で描画が重くなるだけ**なので
    # 既定は False。`perception="model"` で走らせる側が True を渡す（scripts/run_episode.py）。
    camera_depths: bool = False


def make_env(cfg: EnvConfig):
    import robosuite as suite
    from robosuite.controllers import load_controller_config

    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    controller = load_controller_config(default_controller="OSC_POSE")
    return suite.make(
        "PickPlace",
        robots=cfg.robot,
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=cfg.camera_names,
        camera_heights=cfg.camera_height,
        camera_widths=cfg.camera_width,
        camera_depths=cfg.camera_depths,
        control_freq=cfg.control_freq,
        horizon=cfg.horizon,
        single_object_mode=0,
        ignore_done=True,
        hard_reset=False,
    )


# --- 物体の状態 ---------------------------------------------------------

def _body_id(env, object_name: str) -> int:
    return env.sim.model.body_name2id(env.objects[OBJECT_TO_BIN_ID[object_name]].root_body)


def object_position(env, object_name: str) -> np.ndarray:
    """シミュレータ内部の真値（L1 縮退および対照条件 C0 で使う）。"""
    return np.array(env.sim.data.body_xpos[_body_id(env, object_name)])


def object_yaw(env, object_name: str) -> float:
    """物体の z 軸まわりの向き [rad]。把持姿勢の決定に使う。"""
    w, x, y, z = env.sim.data.body_xquat[_body_id(env, object_name)]  # MuJoCo は (w,x,y,z) 順
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def object_half_height(env, object_name: str) -> float:
    obj = env.objects[OBJECT_TO_BIN_ID[object_name]]
    return float(obj.top_offset[2] - obj.bottom_offset[2]) / 2.0


# --- 配置先 -------------------------------------------------------------

def bin_bounds(env, bin_id: int) -> tuple[np.ndarray, np.ndarray]:
    """箱の xy 範囲 (low, high)。robosuite の not_in_bin と同じ区切り方に揃える。"""
    low = np.array([env.bin2_pos[0], env.bin2_pos[1]], dtype=float)
    if bin_id in (0, 2):
        low[0] -= env.bin_size[0] / 2
    if bin_id < 2:
        low[1] -= env.bin_size[1] / 2
    high = low + np.array([env.bin_size[0] / 2, env.bin_size[1] / 2])
    return low, high


def bin_center(env, bin_id: int) -> np.ndarray:
    low, high = bin_bounds(env, bin_id)
    return np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, env.bin2_pos[2]])


def release_position(env, object_name: str, destination: str, slot: int = 0,
                     clearance: float = 0.01) -> np.ndarray:
    """手を離すときの**手先の**位置。

    高さの決め方（実測で F-7 の原因を潰した結果）:
      物体は把持点補正の分だけ手先より下にあるので、その分を足す。
      下げすぎると箱に突っ込み、上げすぎると落下して倒れ、箱の外へ出る。

    slot: 同じ箱に複数入れるときの置き場所をずらす。重ねて置くと崩れて箱の外へ出る。
    """
    bin_id = DESTINATION_TO_BIN_ID[destination]
    pos = bin_center(env, bin_id)
    low, high = bin_bounds(env, bin_id)
    span_y = (high[1] - low[1]) / 4.0
    pos[1] += (slot - 0.5) * span_y if slot else -0.5 * span_y
    pos[2] += object_half_height(env, object_name) + GRASP_Z_OFFSET.get(object_name, 0.0) + clearance
    return pos


def placed_correctly(env, object_name: str, destination: str) -> bool:
    """指定された箱の中に入っているか。"""
    low, high = bin_bounds(env, DESTINATION_TO_BIN_ID[destination])
    pos = object_position(env, object_name)
    return bool(
        low[0] < pos[0] < high[0]
        and low[1] < pos[1] < high[1]
        and env.bin2_pos[2] < pos[2] < env.bin2_pos[2] + 0.12
    )
