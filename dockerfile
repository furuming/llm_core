FROM python:3.11

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV APP_PORT=9000
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV UV_NO_CACHE=1


# uvインストール
RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --system .

RUN uv venv "${VIRTUAL_ENV}" \
    && uv pip install -e ".[cuda]"

CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0"]
