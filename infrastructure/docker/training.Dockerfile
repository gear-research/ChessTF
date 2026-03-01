FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

COPY src/ ./src
COPY configs/ ./configs
COPY pyproject.toml ./pyproject.toml
COPY infrastructure/docker/entrypoint.sh ./entrypoint.sh

RUN chmod +x ./entrypoint.sh
RUN apt update && apt install -y --no-install-recommends stockfish 
RUN pip install --no-cache-dir ".[dev]"

ENTRYPOINT ["./entrypoint.sh"]