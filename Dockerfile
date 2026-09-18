FROM --platform=linux/amd64 zenmldocker/zenml:py3.12
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv venv && uv sync --frozen --all-extras --no-install-project

COPY src ./src
COPY data ./data
RUN uv pip install --python /app/.venv/bin/python --no-deps -e .

ENV PATH="/app/.venv/bin:$PATH"
