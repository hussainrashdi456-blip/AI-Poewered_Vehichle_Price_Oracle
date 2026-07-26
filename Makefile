.PHONY: help install run build docker-run docker-build clean test

help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  run          - Run Streamlit app"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  clean        - Clean cache files"
	@echo "  test         - Run tests"

install:
	pip install -r requirements.txt

run:
	cd app && streamlit run app.py

docker-build:
	docker build -t car-price-predictor ./app

docker-run:
	docker run -p 8501:8501 car-price-predictor

docker-compose:
	docker-compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	pytest tests/