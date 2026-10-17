#!/usr/bin/env python
"""MolmoAct が本課題の環境で「動くか・何Hz出るか」だけを測る実現可能性スパイク。

**採否を決める実験ではない。** 決めるのは次の2点のみ:

1. **この環境で MolmoAct-7B-D をロードして推論できるか**（できなければ以降は全て机上）
2. **1判断あたり何秒かかるか**＝サーベイ資料の判定基準③「計画層 10Hz 以上」に
   **届くのか届かないのか**（届かないなら、その数字自体が本課題の知見になる）

**既存の提出物には一切触れない。** 出力は `outputs/molmoact_spike/` のみ。
既存パイプライン（transformers 5.5.4）とは**別の conda 環境**で動かす
（MolmoAct は transformers 4.52 系を要求するため。DEC-59）。

使い方:
    conda run -n molmoact python \\
        experiments/exp008_molmoact_feasibility/probe_molmoact.py --trials 5
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "allenai/MolmoAct-7B-D-0812"
DEFAULT_OUT = ROOT / "outputs" / "molmoact_spike"

# サーベイ資料の判定基準③（計画層 10Hz 以上／動作生成層 30Hz 以上）。
# ここでは計画層に相当する1回の生成が何Hzかだけを見る。
TARGET_PLAN_HZ = 10.0

INSTRUCTION = "pick up the milk carton and place it in the left bin"


def observation_image(size: int = 512):
    """robosuite の `agentview` と同じ解像度の観測画像を1枚用意する。

    **実機のシーンをそのまま使う**のが理想だが、本スパイクの問いは
    「ロードできるか・何Hz出るか」であり、**絵の中身は速度に影響しない**。
    robosuite が使えるときは実シーンを、使えないときは同解像度の合成画像を使い、
    **どちらを使ったかを必ず記録する**（実測値の由来を曖昧にしない）。
    """
    try:
        sys.path.insert(0, str(ROOT))
        from src.envs.pickplace import EnvConfig, make_env

        env = make_env(EnvConfig(camera_height=size, camera_width=size))
        obs = env.reset()
        for key in ("agentview_image", "agentview"):
            if key in obs:
                from PIL import Image

                arr = obs[key]
                if arr.ndim == 3:
                    return Image.fromarray(arr[::-1]), "robosuite_agentview"
        raise KeyError("agentview 画像が観測に無い")
    except Exception as exc:  # robosuite が別環境にしか無い場合はここへ来る
        from PIL import Image
        import numpy as np

        rng = np.random.default_rng(0)
        arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(arr), f"synthetic_fallback({type(exc).__name__})"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record: dict = {
        "実施日時": datetime.now().isoformat(timespec="seconds"),
        "model_id": args.model,
        "python": platform.python_version(),
        "結論": None,
    }

    try:
        import torch
        import transformers

        record["torch"] = torch.__version__
        record["transformers"] = transformers.__version__
        record["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            record["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        record["結論"] = "不可: torch/transformers が読めない"
        record["エラー"] = f"{type(exc).__name__}: {exc}"
        (out / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 1

    image, image_source = observation_image()
    record["観測画像の由来"] = image_source

    # --- ロード ---
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        t0 = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            args.model, trust_remote_code=True, padding_side="left")
        # device_map="auto"/{"":0} のどちらでも、ロード後 RSS が約32.8GBに達し
        # CPU使用率5000%超が15分以上続いて完了しないことを2回実測で確認した。
        # 7B params * fp32(4byte) = 28GB がほぼ一致 → 実チェックポイントを読む前に
        # 全パラメータを fp32 でランダム初期化する低速パス(meta device 未対応)に
        # 落ちていると推測し、low_cpu_mem_usage=True を明示して meta 初期化を強制する。
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map={"": 0} if torch.cuda.is_available() else "cpu")
        record["ロード秒"] = round(time.perf_counter() - t0, 1)
        record["パラメータ数"] = int(sum(p.numel() for p in model.parameters()))
        record["実際のデバイス"] = str(next(model.parameters()).device)
    except Exception as exc:
        record["結論"] = "不可: モデルをロードできない"
        record["エラー"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        (out / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 1

    # --- 推論（1回目は暖機。計測から外す） ---
    latencies: list[float] = []
    text = None
    try:
        for i in range(args.trials + 1):
            t0 = time.perf_counter()
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": INSTRUCTION}]}],
                tokenize=True, return_dict=True, add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if i == 0:
                record["暖機秒"] = round(elapsed, 2)
                prompt_len = inputs["input_ids"].shape[1]
                text = processor.batch_decode(
                    generated[:, prompt_len:], skip_special_tokens=True)[0]
            else:
                latencies.append(elapsed)
    except Exception as exc:
        record["結論"] = "不可: 推論で例外"
        record["エラー"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        (out / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 1

    latencies.sort()
    median = latencies[len(latencies) // 2]
    record.update({
        "試行回数": len(latencies),
        "レイテンシ秒_中央値": round(median, 3),
        "レイテンシ秒_最小": round(latencies[0], 3),
        "レイテンシ秒_最大": round(latencies[-1], 3),
        "計画層Hz_実測": round(1.0 / median, 3),
        "判定基準3_計画10Hz": "未達" if 1.0 / median < TARGET_PLAN_HZ else "達成",
        "必要な高速化倍率": round(TARGET_PLAN_HZ * median, 1),
        "出力の先頭400字": (text or "")[:400],
        "結論": "可: ロードも推論も成功した",
    })
    (out / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))

    print(f"ロード: {record['ロード秒']} 秒 / パラメータ {record['パラメータ数']/1e9:.2f}B")
    print(f"レイテンシ中央値: {median:.3f} 秒  = {1.0/median:.2f} Hz")
    print(f"判定基準③(計画10Hz): **{record['判定基準3_計画10Hz']}**"
          f"（{record['必要な高速化倍率']} 倍の高速化が要る）")
    print(f"保存: {out / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
