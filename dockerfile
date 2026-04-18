FROM python:3.11

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV APP_PORT=9000
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV UV_NO_CACHE=1

COPY . .

RUN uv venv "${VIRTUAL_ENV}" \
    && uv pip install -e ".[cuda]" \
    && cp .env.example .env

CMD ["python", "src/main.py"]

