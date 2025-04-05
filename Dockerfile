FROM --platform=linux/amd64 zenmldocker/zenml:py3.12
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project into the image
ADD . /app

WORKDIR /app

RUN uv venv
RUN uv sync --frozen --extra k8s --extra azure --extra mlflow --extra bento
RUN pip install -e .

ENV PATH="/app/.venv/bin:$PATH"