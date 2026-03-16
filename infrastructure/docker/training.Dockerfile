FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY src/ ./src
COPY configs/ ./configs
COPY pyproject.toml ./pyproject.toml
COPY infrastructure/docker/entrypoint.sh ./entrypoint.sh
COPY .dvc/config .dvc/config
COPY data/processed.dvc data/processed.dvc
COPY data/interim.dvc data/interim.dvc

RUN git init && git config user.email "docker@build" && git config user.name "Docker"
RUN chmod +x ./entrypoint.sh
RUN apt update && apt install -y --no-install-recommends stockfish 
RUN PIP_EXTRA_INDEX_URL="" pip install --no-cache-dir ".[dev]"

ENTRYPOINT ["./entrypoint.sh"]