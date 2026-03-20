.PHONY: install lint format test download process filter-dry docker-train docker-sweep

YEAR ?= 2024
MONTH ?= 1

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

test:
	pytest tests/ -v

download:
	python -m chesstf.data.process download --year $(YEAR) --month $(MONTH)

process:
	python -m chesstf.data.process full --year $(YEAR) --month $(MONTH)

filter-dry:
	python -m chesstf.data.process filter --year $(YEAR) --month $(MONTH) --dry-run

docker-train:
	docker compose --profile train up --build

docker-sweep:
	docker compose --profile sweep up --build
