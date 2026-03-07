FROM nvcr.io/nvidia/pytorch:25.01-py3

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY src/ ./src
COPY configs/ ./configs
COPY pyproject.toml ./pyproject.toml
COPY infrastructure/docker/entrypoint.sh ./entrypoint.sh
COPY data/processed ./data/processed

RUN chmod +x ./entrypoint.sh
RUN apt update && apt install -y --no-install-recommends stockfish 
RUN pip install --no-cache-dir ".[dev]"

ENTRYPOINT ["./entrypoint.sh"]