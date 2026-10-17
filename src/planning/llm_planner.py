"""計画層: 日本語の指示文を手順に分解する（言語モデルの推論）。

固定ルール版（`src/pipeline.py` の `plan_from_instruction`）との違いは**汎化**である。
固定ルールは「飲み物」「食べ物」という語が文字列として含まれる場合しか動かないが、
言語モデルなら「ジュースとミルクをこっちに寄せて」のような言い換えも扱える。

守っていること:
  - 出力は **JSON に強制**し、パースできなければ**手順を出さない**（推測で埋めない）。
  - シーンに無い物体・未定義の配置先は**その場で落とす**（黙って別の物へ置き換えない）。
  - 妨害物（Cereal）は候補として渡すが、**指示に合わなければ選ばれないこと**を評価で測る。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

from src.device import default_device

# 社内プロキシ配下での取得に必要（未設定だと証明書検証と Xet CDN で落ちる）
os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# 物体の日本語名。モデルには日本語で見せ、戻り値で内部名へ直す。
OBJECT_LABELS: dict[str, str] = {
    "Milk": "牛乳パック",
    "Bread": "パン",
    "Cereal": "シリアルの箱",
    "Can": "ジュースの缶",
}
DESTINATIONS = ["左の箱", "右の箱"]

SYSTEM_PROMPT = """あなたは片付けロボットの計画係です。
卓上にある物の中から、指示に合うものだけを選び、指定された箱へ入れる手順を作ります。

卓上にある物: 牛乳パック / パン / シリアルの箱 / ジュースの缶
使える箱: 左の箱 / 右の箱

規則:
- 指示に合う物を**すべて**選ぶ。合わない物は選ばない。
- 出力は JSON 配列のみ。説明文や前置きは書かない。
- 各要素は {"target": "<物の名前>", "destination": "<箱の名前>"} の形にする。
- 指示から手順が作れないときは [] と答える。

例:
指示「シリアルとパンを右の箱へ」
出力 [{"target": "シリアルの箱", "destination": "右の箱"}, {"target": "パン", "destination": "右の箱"}]"""


@dataclass
class PlannerConfig:
    model_id: str = MODEL_ID
    device: str = field(default_factory=default_device)
    max_new_tokens: int = 200
    retries: int = 2          # JSON にならなかったときの作り直し回数
    temperature: float = 0.0  # 既定は決定的。ばらつきを測るときだけ上げる
    labels: dict[str, str] = field(default_factory=lambda: dict(OBJECT_LABELS))


@lru_cache(maxsize=2)
def _load(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16).to(device).eval()
    torch.set_grad_enabled(False)
    return tokenizer, model


def _generate(instruction: str, cfg: PlannerConfig, seed: int | None = None) -> str:
    import torch

    tokenizer, model = _load(cfg.model_id, cfg.device)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"指示「{instruction}」\n出力"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(cfg.device)
    if seed is not None:
        torch.manual_seed(seed)
    output = model.generate(
        **inputs, max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.temperature > 0, temperature=cfg.temperature or None,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _extract_json(raw: str) -> list | None:
    """本文から JSON 配列を取り出す。取り出せなければ None（推測で埋めない）。"""
    match = re.search(r"\[.*\]", raw, flags=re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def plan_with_llm(instruction: str, cfg: PlannerConfig | None = None,
                  seed: int | None = None) -> dict:
    """指示文 → 手順。戻り値には**採否の理由**も含める（黙って落とさない）。

    戻り値: {"steps": [(target, destination), ...], "raw": 生成文, "parsed": bool,
             "dropped": [落とした要素と理由]}
    """
    cfg = cfg or PlannerConfig()
    label_to_name = {label: name for name, label in cfg.labels.items()}

    raw = ""
    parsed = None
    for attempt in range(cfg.retries + 1):
        raw = _generate(instruction, cfg, seed=None if seed is None else seed + attempt)
        parsed = _extract_json(raw)
        if parsed is not None:
            break
        # 貪欲法（temperature=0）は何度生成しても同じ文になる。作り直しても無駄。
        if cfg.temperature <= 0:
            break

    if parsed is None:
        return {"steps": [], "raw": raw, "parsed": False, "dropped": []}

    steps, dropped = [], []
    for item in parsed:
        if not isinstance(item, dict):
            dropped.append({"item": item, "reason": "辞書でない"})
            continue
        target, destination = item.get("target"), item.get("destination")
        if target not in label_to_name:
            dropped.append({"item": item, "reason": f"シーンに無い物体: {target}"})
            continue
        if destination not in DESTINATIONS:
            dropped.append({"item": item, "reason": f"未定義の配置先: {destination}"})
            continue
        steps.append((label_to_name[target], destination))

    return {"steps": steps, "raw": raw, "parsed": True, "dropped": dropped}
