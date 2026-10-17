"""評価: 指標の集計と失敗モードの分類。

定義の正は docs/20_設計/02_評価設計.md。ここはその実装。
**検算（成功数 + Σ失敗 == 総数）を必ず行い、合わなければ例外で止める。**
過去に内訳の1バケツを書き落として合計が母数と合わない報告を出した事例があるため。
"""

from __future__ import annotations

from collections import Counter

FAILURE_MODES: dict[str, str] = {
    "F-1": "計画失敗",
    "F-2": "対象取り違え",
    "F-3": "未検出",
    "F-4": "位置誤差",
    "F-5": "把持失敗",
    "F-6": "搬送失敗",
    "F-7": "配置失敗",
    "F-8": "タイムアウト",
    "F-9": "その他",
}


class LedgerMismatch(Exception):
    """内訳の合計が母数と合わない。結果を報告してはいけない状態。"""


def summarize_steps(step_results: list[dict]) -> dict:
    """手順（1物体の pick&place）単位の集計。

    step_results の各要素: {"success": bool, "failure_mode": str|None, "elapsed_sec": float, ...}
    """
    total = len(step_results)
    successes = [r for r in step_results if r["success"]]
    failures = Counter(r["failure_mode"] or "F-9" for r in step_results if not r["success"])

    checked = len(successes) + sum(failures.values())
    if checked != total:
        raise LedgerMismatch(f"内訳の合計 {checked} が母数 {total} と一致しない")

    unknown = set(failures) - set(FAILURE_MODES)
    if unknown:
        raise LedgerMismatch(f"未定義の失敗モード: {sorted(unknown)}")

    elapsed = [r["elapsed_sec"] for r in step_results]
    return {
        "n_steps": total,
        "n_success": len(successes),
        "step_success_rate": _rate(len(successes), total),
        "step_success_fraction": f"{len(successes)}/{total}",
        "failure_modes": {
            code: {"name": FAILURE_MODES[code], "count": failures[code]}
            for code in FAILURE_MODES if failures[code]
        },
        "cycle_time_sec_mean": round(sum(elapsed) / total, 2) if total else None,
        "ledger_check": f"{len(successes)} + {sum(failures.values())} = {total} ✔",
    }


def summarize_episodes(episodes: list[dict]) -> dict:
    """エピソード単位の集計。episodes の各要素は run_episode() の戻り値。"""
    total = len(episodes)
    if total == 0:
        raise LedgerMismatch("エピソードが0件。集計対象がない")

    n_success = sum(1 for ep in episodes if ep["success"])
    all_steps = [s for ep in episodes for s in ep["steps"]]

    correct_id = sum(1 for s in all_steps if s.get("identified_correctly") is True)
    n_identified = sum(1 for s in all_steps if s.get("identified_correctly") is not None)

    summary = {
        "n_episodes": total,
        "episode_success_fraction": f"{n_success}/{total}",
        "episode_success_rate": _rate(n_success, total),
        "step_level": summarize_steps(all_steps),
    }
    if n_identified:
        summary["identification_accuracy_fraction"] = f"{correct_id}/{n_identified}"
        summary["identification_accuracy"] = _rate(correct_id, n_identified)

    recovery = summarize_recovery(all_steps)
    if recovery:
        summary["recovery"] = recovery
    return summary


def summarize_recovery(step_results: list[dict]) -> dict | None:
    """再試行の集計。**何回やり直したかを隠さずに出す。**

    回復の定義: **自己監視が無ければその場で失敗していた手順が、やり直して成功した件数**。
    段0で見つかり1回目で掴めた手順は、再試行を通っていないので数えない。
    """
    tracked = [r for r in step_results if "grasp_attempts" in r]
    if not tracked:
        return None

    redetected = [r for r in tracked if r.get("detect_level", 0) > 0]
    regrasped = [r for r in tracked if r.get("grasp_attempts", 1) > 1]
    retried = [r for r in tracked if r.get("detect_level", 0) > 0 or r.get("grasp_attempts", 1) > 1]
    rescued = [r for r in retried if r["success"]]
    rejections = sum(1 for r in tracked
                     for e in r.get("recovery_events", []) if e.get("kind") == "reject")

    return {
        "n_steps": len(tracked),
        "n_redetect": len(redetected),
        "n_regrasp": len(regrasped),
        "n_retried": len(retried),
        "n_rescued": len(rescued),
        "rescued_fraction": f"{len(rescued)}/{len(retried)}",
        "n_plausibility_rejections": rejections,
        "max_grasp_attempts": max((r.get("grasp_attempts", 1) for r in tracked), default=1),
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
