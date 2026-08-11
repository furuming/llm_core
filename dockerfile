FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home --shell /bin/bash app \
    && mkdir -p /models/huggingface \
    && chown -R app:app /models

ENV APP_PORT=9000 \
    HF_HOME=/models/huggingface \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# The application itself is mounted by Compose. Keeping only dependencies in
# the image means source edits made in WSL2 are immediately visible.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra cuda

EXPOSE 9000

USER app
