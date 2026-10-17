"""3層（計画→知覚→制御）を繋いでエピソードを実行する。

知覚層は差し替え可能にしてあり、これがそのまま対照条件になる:
  perception="ground_truth" … シミュレータの真値を使う（条件 C0 / 縮退 L1）
  perception="model"        … 視覚モデルの推論を使う（条件 C1 以降。src/perception/）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.control.primitives import ControlConfig, Primitives
from src.envs.pickplace import (
    OBJECT_TO_BIN_ID,
    CATEGORY_RULES,
    GRASP_Z_OFFSET,
    object_half_height,
    object_position,
    object_yaw,
    placed_correctly,
    release_position,
)


@dataclass
class Step:
    """計画層の出力1件。層の契約（docs/20_設計/01_システム設計.md §4）。"""

    target: str
    destination: str


@dataclass
class RunConfig:
    instruction: str = "飲み物を左の箱へ、食べ物を右の箱へ"
    planner: str = "rule"                  # "rule"（固定ルール）| "llm"（言語モデル）
    perception: str = "ground_truth"       # "ground_truth" | "model"
    position_mode: str = "ray_plane"       # "ray_plane" | "depth"
    yaw_source: str = "truth"              # "truth" | "fixed" | "model"
    seed: int = 0
    control: ControlConfig = field(default_factory=ControlConfig)
    max_steps_per_episode: int = 4000
    # 計画層をバイパスして手順を直接与える（**実行順の影響を切り分けるため**）
    explicit_plan: list[tuple[str, str]] | None = None
    # --- 実行時の自己監視と再試行 ----------------------------------------
    # **既定は False**。True にしたときだけ振る舞いが変わるので、C0〜C4 の記録済みの
    # 数値（41/60 等）はそのまま再現する（設計原則4: 実験で既定値を書き換えない）。
    recovery: bool = False
    max_detect_escalations: int = 2  # 未検出のときに語・閾値を緩めて探し直す回数
    max_grasp_retries: int = 2       # 空振りを検知したときに掴み直す回数
    # 掴み直しのたびに把持点の高さをずらす幅 [m]。
    # **同じことをもう一度やっても同じ結果になる**ので、再試行は必ず何かを変える。
    # 指が完全に閉じている（空振り）場合は高すぎた可能性が高いので、まず下げる。
    grasp_retry_dz: tuple[float, ...] = (0.0, -0.015, 0.015)

CAMERA = "agentview"

# cv2 は日本語を描けないので、画面に出すときだけローマ字に直す。
_ROMAJI = {"左の箱": "LEFT BIN", "右の箱": "RIGHT BIN"}


# 実行順を揃えるための基準順。**順番だけで手順成功が 29〜41/60 まで動く**
# （experiments/exp006_plan_order）ため、計画層の出力順に任せない。
# 順番は指示の意味に含まれないので、正規化しても「言語を理解した」という主張は損なわれない。
CANONICAL_ORDER = ["Milk", "Can", "Bread", "Cereal"]


def normalize_plan_order(steps: list[Step]) -> list[Step]:
    """手順の**中身は変えず**、実行順だけを基準順へ揃える。"""
    return sorted(steps, key=lambda s: CANONICAL_ORDER.index(s.target)
                  if s.target in CANONICAL_ORDER else len(CANONICAL_ORDER))


def plan_from_instruction(instruction: str, mode: str = "rule") -> list[Step]:
    """計画層: 指示文 → 手順の並び。

    mode="rule" … カテゴリ辞書による文字列一致（L2）
    mode="llm"  … 言語モデルの推論（L3）。**戻り値の形は同じ**なので、
                  差し替えるだけでアブレーション（C3 対 C4）になる。

    どちらも最後に `normalize_plan_order` を通す。
    """
    if mode == "llm":
        from src.planning.llm_planner import plan_with_llm

        result = plan_with_llm(instruction)
        return normalize_plan_order([Step(target=t, destination=d) for t, d in result["steps"]])

    steps: list[Step] = []
    for category, (objects, destination) in CATEGORY_RULES.items():
        if category in instruction:
            steps.extend(Step(target=obj, destination=destination) for obj in objects)
    return normalize_plan_order(steps)


def locate(env, object_name: str, cfg: RunConfig, obs: dict | None = None,
           escalation: int = 0) -> dict:
    """知覚層。戻り値は層の契約に従う。

    `perception="model"` は真値の位置を一切見ず、カメラ画像から位置を求める。
    向き（yaw）は別のノブ（`yaw_source`）で扱う。
    **位置と向きを同時に変えない**ためである（評価設計 §4-1 作法3）。

    escalation: 再探索の段。**0 は従来と完全に同じ**。1以上では語を足し閾値を下げるが、
    緩めた結果は `is_plausible()`（台の上か・他の検出と重ならないか）を通さないと採らない。
    """
    truth_yaw = object_yaw(env, object_name)

    if cfg.perception == "ground_truth":
        yaw = truth_yaw if cfg.yaw_source == "truth" else 0.0
        return {
            "label": object_name, "score": 1.0,
            "position_3d": object_position(env, object_name),
            "yaw": yaw, "source": "ground_truth",
            "identified_correctly": None,  # 真値のため同定の正誤は測らない
        }

    if cfg.perception != "model":
        raise ValueError(f"知覚モード '{cfg.perception}' は未定義")

    from src.perception.owlvit_detector import (
        PerceptionConfig, escalated_config, is_plausible, locate_from_image,
    )

    rgb = np.flipud(obs[f"{CAMERA}_image"]).copy()
    depth = np.flipud(obs[f"{CAMERA}_depth"][:, :, 0]).copy()
    table_z = float(env.bin2_pos[2])
    # 物体の高さはカタログ値（形状情報）であり、その場の真値位置ではない。
    center_z = {n: table_z + object_half_height(env, n) for n in OBJECT_TO_BIN_ID}
    base_cfg = PerceptionConfig(position_mode=cfg.position_mode,
                                estimate_yaw=(cfg.yaw_source == "model"))
    found = locate_from_image(env, rgb, depth, CAMERA,
                              escalated_config(base_cfg, escalation),
                              center_z, table_z)

    rejected = None
    if object_name in found and escalation > 0:
        # 緩めて拾ったものだけ門を通す（段0の結果は従来どおり無検査で採る）。
        others = {n: d for n, d in found.items() if n != object_name}
        ok, reason = is_plausible(env, found[object_name]["position_3d"], others)
        if not ok:
            rejected = reason
            found.pop(object_name)

    if object_name not in found:
        fallback = truth_yaw if cfg.yaw_source == "truth" else 0.0
        return {"label": object_name, "score": 0.0, "position_3d": None, "yaw": fallback,
                "source": "model", "identified_correctly": False,
                "escalation": escalation, "rejected_reason": rejected}

    det = found[object_name]
    if cfg.yaw_source == "truth":
        yaw = truth_yaw
    elif cfg.yaw_source == "model":
        yaw = det["yaw"] if det["yaw"] is not None else 0.0  # 推定できなければ固定向き
    else:
        yaw = 0.0

    truth = object_position(env, object_name)
    # 同定の正誤だけは真値で**採点**する（制御には渡さない）
    correct = bool(np.linalg.norm(np.asarray(det["position_3d"])[:2] - truth[:2]) < 0.08)
    return {
        "label": object_name, "score": det["score"],
        "position_3d": np.asarray(det["position_3d"]),
        "yaw": yaw, "source": "model", "identified_correctly": correct,
        "query": det["query"], "bbox": det["bbox"], "escalation": escalation,
        "rejected_reason": rejected,
    }


def perceive_with_escalation(env, prim, cfg: RunConfig, object_name: str,
                             trace: dict) -> dict:
    """見つかるまで段階的に探し直す。`cfg.recovery` が False なら1回だけ試す（従来どおり）。"""
    levels = cfg.max_detect_escalations + 1 if cfg.recovery else 1
    detection = None
    for level in range(levels):
        detection = locate(env, object_name, cfg, obs=prim.obs, escalation=level)
        if detection["position_3d"] is not None:
            if level > 0:
                trace["events"].append(
                    {"kind": "redetect", "level": level, "query": detection.get("query"),
                     "score": round(float(detection["score"]), 3)})
            break
        if detection.get("rejected_reason"):
            trace["events"].append(
                {"kind": "reject", "level": level, "reason": detection["rejected_reason"]})
    trace["detect_level"] = detection.get("escalation", 0) if detection else 0
    return detection


def run_episode(env, cfg: RunConfig, recorder=None, episode_index: int = 0) -> dict:
    # 条件間で初期配置を揃えるため、エピソードごとに種を固定する（評価設計 §4-1 作法1）。
    np.random.seed(cfg.seed * 1000 + episode_index)
    prim = Primitives(env, cfg.control, on_step=recorder.capture if recorder else None)
    prim.reset()

    plan = ([Step(target=t, destination=d) for t, d in cfg.explicit_plan]
            if cfg.explicit_plan else plan_from_instruction(cfg.instruction, cfg.planner))
    if not plan:
        failed = {"target": "-", "destination": "-", "success": False, "failure_mode": "F-1",
                  "elapsed_sec": 0.0, "lift_height_m": 0.0, "source": cfg.perception,
                  "score": 0.0, "identified_correctly": None}
        return {"episode": episode_index, "instruction": cfg.instruction, "success": False,
                "steps": [failed], "n_env_steps": prim.steps}

    step_results = []
    slots_used: dict[str, int] = {}
    for index, step in enumerate(plan):
        slot = slots_used.get(step.destination, 0)
        slots_used[step.destination] = slot + 1
        if recorder:
            recorder.set_overlay([
                f"[{episode_index + 1}] {cfg.planner}+{cfg.perception}",
                f"step {index + 1}/{len(plan)}: {step.target} -> {_ROMAJI.get(step.destination, step.destination)}",
            ])
        result = _run_one_step(env, prim, cfg, step, slot, recorder)
        step_results.append(result)
        if recorder:
            recorder.log_event(episode=episode_index, **result)
        if prim.steps > cfg.max_steps_per_episode:
            break
    if recorder:
        recorder.set_overlay([])

    return {
        "episode": episode_index,
        "instruction": cfg.instruction,
        "plan": [f"{s.target}→{s.destination}" for s in plan],
        "success": len(step_results) == len(plan) and all(r["success"] for r in step_results),
        "steps": step_results,
        "n_env_steps": prim.steps,
    }


def _run_one_step(env, prim: Primitives, cfg: RunConfig, step: Step, slot: int = 0,
                  recorder=None) -> dict:
    """1物体の pick & place。失敗モードは最初に起きた事象で1つだけ付ける。

    `cfg.recovery=True` のときは、**真値を見ずに**自分の失敗に気づいてやり直す:
      - 見つからない → 語を足し閾値を下げて探し直す（緩めた候補は妥当性検査を通す）
      - 掴めていない → 指の開き幅で検知し、知覚からやり直して掴み直す
    """
    started = time.perf_counter()
    trace: dict = {"events": [], "detect_level": 0, "grasp_attempts": 1}

    detection = perceive_with_escalation(env, prim, cfg, step.target, trace)
    if recorder is not None and detection.get("bbox") is not None:
        recorder.set_overlay(recorder._banner, detection["bbox"])

    base = {
        "target": step.target, "destination": step.destination,
        "source": detection["source"], "score": round(float(detection["score"]), 3),
        "identified_correctly": detection["identified_correctly"],
    }
    if detection["position_3d"] is None:
        return {**base, "success": False, "failure_mode": "F-3",  # 画面にあるのに検出できない
                "elapsed_sec": round(time.perf_counter() - started, 2), "lift_height_m": 0.0,
                **_trace_fields(trace)}

    start_pos = np.array(detection["position_3d"])
    truth_pos = object_position(env, step.target)

    failure_mode = None
    lift_height = 0.0
    prim.cfg.grasp_z_offset = GRASP_Z_OFFSET.get(step.target, 0.0)

    attempts = cfg.max_grasp_retries + 1 if cfg.recovery else 1
    base_z_offset = GRASP_Z_OFFSET.get(step.target, 0.0)
    for attempt in range(attempts):
        trace["grasp_attempts"] = attempt + 1
        if cfg.recovery:
            dz = cfg.grasp_retry_dz[min(attempt, len(cfg.grasp_retry_dz) - 1)]
            prim.cfg.grasp_z_offset = base_z_offset + dz
        reached = prim.pick(start_pos, object_yaw=detection.get("yaw"))

        if not cfg.recovery:
            failure_mode = None if reached else "F-4"
            break

        # --- ここから先は真値を使わない自己監視 ---
        # 掴めていれば指は物体の幅で止まる。空振りなら最後まで閉じる（実測で 36/36 分離）。
        if reached and prim.is_grasping():
            if attempt > 0:
                trace["events"].append({"kind": "regrasp_success", "attempt": attempt + 1})
            break
        if attempt == attempts - 1:
            failure_mode = "F-4" if not reached else None
            break

        trace["events"].append({
            "kind": "grasp_miss", "attempt": attempt + 1,
            "finger_gap_m": round(prim.finger_gap, 4),
            "next_dz": cfg.grasp_retry_dz[min(attempt + 1, len(cfg.grasp_retry_dz) - 1)],
        })
        if recorder:
            recorder.set_overlay(recorder._banner[:1] + [
                f"grasp miss (gap={prim.finger_gap * 1000:.1f}mm) -> re-perceive & retry",
            ])
        # 掴めていない物を指で払わないよう、開いてから退く。
        prim.retreat(start_pos)
        detection = perceive_with_escalation(env, prim, cfg, step.target, trace)
        if detection["position_3d"] is None:
            failure_mode = "F-3"
            break
        start_pos = np.array(detection["position_3d"])
        if recorder is not None and detection.get("bbox") is not None:
            recorder.set_overlay(recorder._banner, detection["bbox"])

    if failure_mode is None:
        # 持ち上げ直後に測る。配置後に測り直すと意味が変わるので値を使い回さない。
        # 基準は真値の初期位置（推定位置を基準にすると誤差が成否に混ざる）。
        # **これは採点であって制御の入力ではない。**
        lift_height = float(object_position(env, step.target)[2] - truth_pos[2])
        if lift_height < 0.05:
            failure_mode = "F-5"  # 持ち上がっていない＝把持失敗
        elif not prim.place(release_position(env, step.target, step.destination, slot)):
            failure_mode = "F-6"  # 運搬中に目的地へ行けなかった
        elif not placed_correctly(env, step.target, step.destination):
            failure_mode = "F-7"  # 箱の外に置いた

    return {
        **base,
        "success": failure_mode is None,
        "failure_mode": failure_mode,
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "lift_height_m": round(lift_height, 4),
        "position_error_m": round(float(np.linalg.norm(start_pos - truth_pos)), 4),
        **_trace_fields(trace),
    }


def _trace_fields(trace: dict) -> dict:
    """再試行の記録を手順の結果へ畳む。**何回やり直したかを隠さない。**"""
    return {
        "detect_level": trace["detect_level"],
        "grasp_attempts": trace["grasp_attempts"],
        "recovery_events": trace["events"],
    }
