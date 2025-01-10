VERSION=2025-01-10
export VERSION

.PHONY: run
run:
	export MODEL_PATH="./models/2022-01.bin"; \
	pipenv run python duration_prediction_serve/serve.py
	
.PHONY: integration_test
integration_test:
	pipenv run python integration_tests/predict-test.py