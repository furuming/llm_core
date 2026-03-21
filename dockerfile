FROM python:3.11

WORKDIR /app

# uv 
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY . .
RUN cp .env.example .env

# RUN uv sync

