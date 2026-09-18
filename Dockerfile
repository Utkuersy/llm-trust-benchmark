# syntax=docker/dockerfile:1
# AI Trustworthiness Benchmark Platform
# A multi-stage build: dependencies are compiled in a separate layer, keeping the runtime image thin.

# --------------------------------------------------------------------------- #
# Stage 1 — dependencies
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# --------------------------------------------------------------------------- #
# Stage 2 — runtime image
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    AITB__LOGGING__JSON_FORMAT=true

# An unprivileged user: the benchmark engine runs untrusted code,
# it must never be run as root.
RUN groupadd --gid 1000 aitb \
    && useradd --uid 1000 --gid aitb --create-home aitb

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=aitb:aitb . /app

RUN mkdir -p /app/db /app/results /app/mlruns \
    && chown -R aitb:aitb /app/db /app/results /app/mlruns

USER aitb

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).status==200 else 1)" \
    || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
