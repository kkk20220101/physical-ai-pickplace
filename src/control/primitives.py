"""制御層: 掴む・運ぶ・置く の動作プリミティブ。

学習を使わずスクリプトで動かす（方針 DEC-07: 学習は行わない）。
OSC_POSE コントローラに対し、目標位置との差分を比例制御で与える。

設計上の約束:
  - すべてのプリミティブに**ステップ上限**を持たせる。無限待ちを作らない。
  - 成否を戻り値で返す。黙って失敗を飲み込まない。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRIPPER_CLOSE = 1.0
GRIPPER_OPEN = -1.0

# 指の開き幅がこれを下回ったら「空振り」とみなす [m]。
# 実測（`scripts/probe_grasp_sensor.py` ・36試行）: 成功時の最小 0.0372 / 失敗時の最大 0.0015。
# 間が広く空いているので、この帯のどこに置いても判定は 36/36 一致する。
GRASP_GAP_THRESHOLD = 0.01


def finger_gap(obs) -> float:
    """グリッパの指の開き幅 [m]。**真値を使わない固有受容感覚である。**

    Panda は2つの指関節を持ち、`robot0_gripper_qpos` は (右, 左) で符号が逆になる。
    物体を掴めていれば指はその幅で止まり、空振りなら最後まで閉じる。
    """
    q = np.asarray(obs["robot0_gripper_qpos"], dtype=float)
    return float(abs(q[0]) + abs(q[1]))


def yaw_from_quat(quat) -> float:
    """robosuite の (x, y, z, w) クォータニオンから z 軸まわりの回転角を取り出す。"""
    x, y, z, w = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def wrap_to_jaw_symmetry(angle: float) -> float:
    """平行2指グリッパは90度回しても同じ掴み方になる。誤差を ±45度に畳む。"""
    half = np.pi / 4.0
    return float((angle + half) % (np.pi / 2.0) - half)


@dataclass
class ControlConfig:
    """制御パラメータ。実験で勝った値を既定へ書き戻さない（設定は configs/ で与える）。"""

    kp: float = 10.0           # 位置誤差 → 行動 の比例ゲイン
    kp_yaw: float = 2.0        # ヨー角誤差 → 行動 の比例ゲイン
    pos_tolerance: float = 0.012  # 到達とみなす距離 [m]
    yaw_tolerance: float = 0.05   # ヨー整合とみなす角度 [rad]
    hover_height: float = 0.12    # 物体の上に構える高さ [m]
    lift_height: float = 0.25     # 掴んだ後に持ち上げる高さ [m]
    grasp_z_offset: float = 0.0   # 把持点の z 補正 [m]
    max_steps_per_move: int = 150
    grip_settle_steps: int = 25   # 開閉後に安定を待つステップ数
    release_settle_steps: int = 20
    stall_steps: int = 40         # この歩数だけ動かなければ詰まりと判断する
    stall_distance: float = 0.004
    # 観測姿勢——探し直す前に腕をどける高さ [m]と横への退き幅 [m]。
    # **物体の真上に残ると自分のカメラを遮る**ので、見直す前に必ずどける。
    observe_height: float = 0.32
    observe_offset_x: float = -0.18


class Primitives:
    """1エピソード分のロボット操作。環境の step をここに集約する。"""

    def __init__(self, env, cfg: ControlConfig, on_step=None):
        self.env = env
        self.cfg = cfg
        self.on_step = on_step  # フレーム記録などの副作用をここに差し込む
        self.obs: dict | None = None
        self.steps = 0
        self._gripper = GRIPPER_OPEN

    # --- 基本操作 -------------------------------------------------------
    def reset(self) -> dict:
        self.obs = self.env.reset()
        self.steps = 0
        self._gripper = GRIPPER_OPEN
        if self.on_step:
            self.on_step(self.obs)
        return self.obs

    def _step(self, action: np.ndarray) -> dict:
        self.obs, _, _, _ = self.env.step(action)
        self.steps += 1
        if self.on_step:
            self.on_step(self.obs)
        return self.obs

    @property
    def eef_pos(self) -> np.ndarray:
        return np.array(self.obs["robot0_eef_pos"])

    @property
    def eef_yaw(self) -> float:
        return yaw_from_quat(self.obs["robot0_eef_quat"])

    @property
    def finger_gap(self) -> float:
        """現在の指の開き幅 [m]。"""
        return finger_gap(self.obs)

    def is_grasping(self, threshold: float = GRASP_GAP_THRESHOLD) -> bool:
        """何かを掴めているかを**真値を見ずに**判定する。

        実機でも同じ信号（グリッパの関節角）が取れるので、この判定は移植できる。
        """
        return self.finger_gap > threshold

    def hold(self, n_steps: int) -> None:
        """姿勢を保ったまま待つ（把持や解放の安定待ち）。"""
        for _ in range(n_steps):
            action = np.zeros(self.env.action_dim)
            action[-1] = self._gripper
            self._step(action)

    def set_gripper(self, value: float) -> None:
        self._gripper = value

    # --- 移動 -----------------------------------------------------------
    def move_to(self, target: np.ndarray, max_steps: int | None = None,
                target_yaw: float | None = None, stop_on_stall: bool = False) -> bool:
        """目標位置（必要ならヨー角も）へ比例制御で移動する。到達できたら True。

        stop_on_stall=True のとき、腕が物体や机に当たって進まなくなったら早期に抜ける。
        降下動作では「止まった＝接地した」を意味するため、打ち切っても失敗ではない。
        """
        max_steps = max_steps or self.cfg.max_steps_per_move
        target = np.asarray(target, dtype=float)
        last_pos = self.eef_pos
        stalled = 0

        for _ in range(max_steps):
            error = target - self.eef_pos
            yaw_error = (wrap_to_jaw_symmetry(target_yaw - self.eef_yaw)
                         if target_yaw is not None else 0.0)
            reached = (np.linalg.norm(error) < self.cfg.pos_tolerance
                       and abs(yaw_error) < self.cfg.yaw_tolerance)
            if reached:
                return True

            action = np.zeros(self.env.action_dim)
            action[:3] = np.clip(self.cfg.kp * error, -1.0, 1.0)
            action[5] = np.clip(self.cfg.kp_yaw * yaw_error, -1.0, 1.0)
            action[-1] = self._gripper
            self._step(action)

            if stop_on_stall:
                moved = np.linalg.norm(self.eef_pos - last_pos)
                stalled = stalled + 1 if moved < self.cfg.stall_distance else 0
                last_pos = self.eef_pos
                if stalled >= self.cfg.stall_steps:
                    return True

        return np.linalg.norm(target - self.eef_pos) < self.cfg.pos_tolerance * 2.5

    # --- 複合動作 -------------------------------------------------------
    def pick(self, object_pos: np.ndarray, object_yaw: float | None = None) -> bool:
        """物体の上へ移動 → ヨーを合わせて降下 → 把持 → 持ち上げ。

        object_yaw を与えると、掴む向きを物体の向きに合わせる。
        平行2指グリッパは向きが合っていないと対角線を掴むことになり、実測で倒してしまう。
        """
        object_pos = np.asarray(object_pos, dtype=float)
        hover = object_pos + np.array([0.0, 0.0, self.cfg.hover_height])
        grasp = object_pos + np.array([0.0, 0.0, self.cfg.grasp_z_offset])

        self.set_gripper(GRIPPER_OPEN)
        if not self.move_to(hover, target_yaw=object_yaw):
            return False
        self.move_to(grasp, target_yaw=object_yaw, stop_on_stall=True)

        self.set_gripper(GRIPPER_CLOSE)
        self.hold(self.cfg.grip_settle_steps)

        lift = np.array([grasp[0], grasp[1], object_pos[2] + self.cfg.lift_height])
        self.move_to(lift)
        return True

    def retreat(self, from_pos: np.ndarray) -> None:
        """空振りのあと、手を開いて**カメラの視界からどける**（再探索の準備）。

        2つとも実測で必要だった:
        - **先に開く**: 閉じたまま横に動くと、掴めていない物体を指で払ってしまう。
        - **真上からどける**: 物体の真上に残ると腕が自分のカメラを遮り、
          探し直した結果が**28cm 先の別物を指した**（ep9 Can。修正前の実測）。
        """
        self.set_gripper(GRIPPER_OPEN)
        self.hold(self.cfg.release_settle_steps)
        from_pos = np.asarray(from_pos, dtype=float)
        self.move_to(from_pos + np.array([0.0, 0.0, self.cfg.hover_height]), max_steps=80)
        observe = from_pos + np.array([self.cfg.observe_offset_x, 0.0, self.cfg.observe_height])
        self.move_to(observe, max_steps=120)

    def place(self, release_pos: np.ndarray, approach_height: float = 0.10) -> bool:
        """目的地の上へ運び、降ろしてから解放する。

        release_pos は「手を離す高さ」。高いまま放すと落下の衝撃で箱の外へ跨ねる。
        """
        release_pos = np.asarray(release_pos, dtype=float)
        above = release_pos + np.array([0.0, 0.0, approach_height])
        if not self.move_to(above, max_steps=self.cfg.max_steps_per_move * 2):
            return False
        self.move_to(release_pos, max_steps=self.cfg.max_steps_per_move, stop_on_stall=True)
        self.set_gripper(GRIPPER_OPEN)
        self.hold(self.cfg.release_settle_steps)
        self.move_to(above, max_steps=80)
        return True
