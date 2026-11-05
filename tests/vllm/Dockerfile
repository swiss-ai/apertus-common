# CPU-only vLLM 0.28 (the minor the Apertus 2 serving image runs) for running
# tests/vllm without a GPU: docker build -t vllm-cpu-dev:0.28.0 -f tests/vllm/Dockerfile .
FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
# torchcodec needs ffmpeg shared libraries that the slim image lacks; vLLM only
# uses it for audio inputs and imports without it.
RUN pip install --no-cache-dir vllm==0.28.0 && pip uninstall -y -q torchcodec \
 && python -c "import vllm, vllm.parser.engine.parser_engine; print(vllm.__version__)"
