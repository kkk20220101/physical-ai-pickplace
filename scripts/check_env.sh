#!/usr/bin/env bash
# 実行環境の前提を1コマンドで確認する。
# 数値の正は docs/10_調査/01_実行環境インベントリ.md。ここは「今も成り立つか」の再確認用。
set -uo pipefail

SIM_PY="${SIM_PY:-python}"
# 既定は PATH 上の python。実際の絶対パスは人ごとに違うので公開物には書かない（C-35）。
# docs/10_調査/01_実行環境インベントリ.md §3 に実値を記載（社内限り）。
VLA_PY="${VLA_PY:-python}"

echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || echo "nvidia-smi なし"

echo "=== ディスク ==="
df -h "$PWD" | tail -1

echo "=== 描画 ==="
echo "DISPLAY=${DISPLAY:-(none)}  MUJOCO_GL=${MUJOCO_GL:-(未設定→egl を使う)}"
ldconfig -p | grep -q "libEGL.so.1" && echo "libEGL.so.1: あり" || echo "libEGL.so.1: なし（オフスクリーン描画不可）"

echo "=== シミュレータ環境 ($SIM_PY) ==="
MUJOCO_GL=egl "$SIM_PY" - <<'PY'
import importlib.util as u
for m in ("mujoco", "robosuite", "gymnasium", "cv2", "transformers", "torch"):
    spec = u.find_spec(m)
    if spec is None:
        print(f"  {m:<12} なし")
        continue
    try:
        mod = __import__(m)
        print(f"  {m:<12} {getattr(mod, '__version__', 'ver不明')}")
    except Exception as exc:
        print(f"  {m:<12} import 失敗: {type(exc).__name__}")
PY

echo "=== 学習/VLA 環境 ($VLA_PY) ==="
"$VLA_PY" -c "import lerobot,torch;print('  lerobot',getattr(lerobot,'__version__','?'),'/ torch',torch.__version__,'/ cuda',torch.cuda.is_available())" 2>&1 | tail -1

echo "=== 動画エンコーダ ==="
command -v ffmpeg >/dev/null && echo "  PATH の ffmpeg: $(command -v ffmpeg)" || echo "  PATH に ffmpeg なし（conda 環境のものを使う）"
# SIM_PY と同じ環境（同じ bin/）に ffmpeg が居るかを見る。
# 旧実装の `ls ffmpeg` はカレントディレクトリを見るだけで、検査になっていなかった。
SIM_PY_PATH="$(command -v "$SIM_PY" 2>/dev/null || true)"
if [ -n "$SIM_PY_PATH" ] && [ -x "$(dirname "$SIM_PY_PATH")/ffmpeg" ]; then
  echo "  SIM_PY と同じ環境の ffmpeg: あり"
else
  echo "  SIM_PY と同じ環境の ffmpeg: なし（PATH のものに頼る）"
fi

echo "=== HuggingFace 取得の前提（社内プロキシ対策）==="
echo "  SSL_CERT_FILE=${SSL_CERT_FILE:-(未設定)}   ※未設定だと httpx が証明書検証で落ちる"
echo "  HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-(未設定)}  ※未設定だと Xet CDN が 403 を返す"
echo "  http_proxy=${http_proxy:-(未設定)}"

echo
echo "詳細と実測値: docs/10_調査/01_実行環境インベントリ.md"
