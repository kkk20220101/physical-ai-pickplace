#!/usr/bin/env python
"""計画層の言語モデルが使い物になるかを実測する。

判定するのは4つ:
  1. 指示文から手順の JSON を安定して出せるか（10回中何回）
  2. **言い換え**でも正しく分解できるか（固定ルールでは扱えない文）
  3. **妨害物（シリアル）を正しく除外**できるか
  4. **決定的テスト**: 指示文が空・無関係のときに手順を出さないか
     （出すなら指示文を読んでいない）

使い方:
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt HF_HUB_DISABLE_XET=1
    CUDA_VISIBLE_DEVICES=1 python scripts/probe_planner.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.device import default_device  # noqa: E402
from src.planning.llm_planner import PlannerConfig, plan_with_llm  # noqa: E402

BASE = "飲み物を左の箱へ、食べ物を右の箱へ"
EXPECTED_BASE = {("Milk", "左の箱"), ("Can", "左の箱"), ("Bread", "右の箱")}

# 固定ルール（文字列一致）では**1つも扱えない**言い換え
PARAPHRASES = [
    ("ジュースとミルクを左の箱に寄せて", {("Can", "左の箱"), ("Milk", "左の箱")}),
    ("パンだけ右の箱に置いて", {("Bread", "右の箱")}),
    ("飲めるものを左、食べられるものを右", {("Milk", "左の箱"), ("Can", "左の箱"), ("Bread", "右の箱")}),
    ("牛乳と缶は左の箱、パンは右の箱にお願い", {("Milk", "左の箱"), ("Can", "左の箱"), ("Bread", "右の箱")}),
]

# 手順を出してはいけない入力（出すなら指示文を読んでいない）
DECISIVE = ["", "今日はいい天気ですね", "3 + 5 は何ですか"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--device", default=default_device())
    p.add_argument("--model", default=None, help="比較するモデルを差し替える")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = PlannerConfig(device=args.device)
    if args.model:
        cfg.model_id = args.model
    tag = cfg.model_id.split("/")[-1]
    out = Path(args.out or ROOT / "experiments" / "exp005_llm_planner" / f"{tag}.json")
    print(f"モデル: {cfg.model_id}", flush=True)

    # --- 1. 安定性（同じ指示を繰り返す。既定は決定的なので実質は形式の確認） ---
    parsed_ok = 0
    exact_ok = 0
    no_cereal = 0
    repeats = []
    for i in range(args.repeats):
        res = plan_with_llm(BASE, cfg, seed=i)
        steps = set(res["steps"])
        parsed_ok += int(res["parsed"])
        exact_ok += int(steps == EXPECTED_BASE)
        no_cereal += int(all(t != "Cereal" for t, _ in res["steps"]))
        repeats.append({"i": i, "parsed": res["parsed"], "steps": sorted(steps),
                        "dropped": res["dropped"]})
        print(f"[{i}] parsed={res['parsed']} steps={sorted(steps)}", flush=True)

    # --- 2. 言い換え ---
    paraphrase_rows = []
    paraphrase_ok = 0
    for text, expected in PARAPHRASES:
        res = plan_with_llm(text, cfg, seed=0)
        steps = set(res["steps"])
        ok = steps == expected
        paraphrase_ok += int(ok)
        cereal = any(t == "Cereal" for t, _ in res["steps"])
        paraphrase_rows.append({"instruction": text, "expected": sorted(expected),
                                "got": sorted(steps), "一致": ok, "妨害物を選んだ": cereal})
        print(f"[言い換え] {'✔' if ok else '✘'} {text} → {sorted(steps)}", flush=True)

    # --- 3. 決定的テスト ---
    decisive_rows = []
    decisive_ok = 0
    for text in DECISIVE:
        res = plan_with_llm(text, cfg, seed=0)
        ok = len(res["steps"]) == 0
        decisive_ok += int(ok)
        decisive_rows.append({"instruction": text or "(空文字)",
                              "steps": sorted(set(res["steps"])), "手順を出さない": ok})
        print(f"[決定的] {'✔' if ok else '✘'} 「{text or '(空)'}」 → {sorted(set(res['steps']))}",
              flush=True)

    summary = {
        "model": cfg.model_id,
        "基本指示": {
            "文": BASE,
            "JSONパース": f"{parsed_ok}/{args.repeats}",
            "期待どおりの手順": f"{exact_ok}/{args.repeats}",
            "妨害物を選ばなかった": f"{no_cereal}/{args.repeats}",
            "内訳": repeats,
        },
        "言い換え": {"正解": f"{paraphrase_ok}/{len(PARAPHRASES)}", "内訳": paraphrase_rows},
        "決定的テスト": {"手順を出さなかった": f"{decisive_ok}/{len(DECISIVE)}", "内訳": decisive_rows},
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n--- まとめ ---")
    print(f"JSONパース          {parsed_ok}/{args.repeats}")
    print(f"期待どおりの手順      {exact_ok}/{args.repeats}")
    print(f"妨害物を選ばなかった   {no_cereal}/{args.repeats}")
    print(f"言い換えの正解       {paraphrase_ok}/{len(PARAPHRASES)}")
    print(f"決定的テスト合格     {decisive_ok}/{len(DECISIVE)}")
    print(f"\n保存先: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
