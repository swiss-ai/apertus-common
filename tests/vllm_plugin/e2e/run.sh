#!/usr/bin/env bash
# End-to-end smoke test of the vLLM plugin on stock vLLM (official CPU image).
# Usage: tests/vllm_plugin/e2e/run.sh <tokenizer_dir> <model_dir>
# <model_dir> is created with make_tiny_model.py (needs torch and transformers)
# if it does not exist yet. Requires Docker; serves on localhost:8000.
set -euo pipefail

TOKENIZER=$(cd "$1" && pwd)
MODEL=$2
IMAGE=${VLLM_IMAGE:-vllm/vllm-openai-cpu:v0.28.0}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)

if [ ! -d "$MODEL" ]; then
    python "$HERE/make_tiny_model.py" "$TOKENIZER" "$MODEL"
fi
MODEL=$(cd "$MODEL" && pwd)

docker rm -f apertus2-e2e >/dev/null 2>&1 || true
# The image runs Python 3.12; the plugin code runs there, so the package's
# Python and tokenizers pins are bypassed for this test only.
docker run -d --name apertus2-e2e --shm-size=1g -p 8000:8000 \
    -v "$REPO:/src:ro" -v "$MODEL:/model:ro" -e VLLM_CPU_KVCACHE_SPACE=1 \
    --entrypoint bash "$IMAGE" -c '
        mkdir -p /tmp/ac && cp -r /src/src /src/pyproject.toml /src/README.md /tmp/ac/ &&
        pip install -q --no-deps --ignore-requires-python /tmp/ac &&
        exec vllm serve /model --served-model-name tiny --tokenizer-mode apertus2 \
            --reasoning-parser apertus2 --enable-auto-tool-choice \
            --tool-call-parser apertus2 --max-model-len 2048 --dtype bfloat16' >/dev/null

python "$HERE/smoke.py" --wait 600 || { docker logs apertus2-e2e | tail -30; exit 1; }
