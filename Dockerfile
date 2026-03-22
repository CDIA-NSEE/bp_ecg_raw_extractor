FROM python:3.14-slim AS builder
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra ocr

FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd -g 1001 appgroup && useradd -u 1001 -g appgroup -s /bin/sh -m appuser
WORKDIR /app
COPY --from=builder --chown=appuser:appgroup /app/.venv .venv
COPY --chown=appuser:appgroup src/ src/
ENV PATH="/app/.venv/bin:$PATH"
# Pre-bake PaddleOCR models at image build time — eliminates 15-45s cold start per container.
# Run before USER switch so /app/models is writable by root, then chown to appuser.
RUN HOME=/app/models python -c \
    "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='pt', use_gpu=False)" \
    && chown -R appuser:appgroup /app/models
USER 1001
# HOME points to the baked model directory — no volume mount needed at runtime.
ENV HOME=/app/models
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "bp_ecg_raw_extractor.main"]
