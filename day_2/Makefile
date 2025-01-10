VERSION=2025-01-10
export VERSION

.PHONY: run
run:
	export MODEL_PATH="./models/2022-01.bin"; \
	pipenv run python duration_prediction_serve/serve.py
	
.PHONY: integration_test
integration_test:
	pipenv run python integration_tests/predict-test.py
	
.PHONY: tests
tests:
	pipenv run pytest
	
.PHONY: docker_build
docker_build:
	docker build -t duration-prediction .

.PHONY: docker_run 
docker_run: docker_build
	docker run -it --rm -p 127.0.0.1:9696:9696 duration-prediction:latest