.PHONY: tests
tests:
	pipenv run pytest
	
.PHONY: train_2022_01_model
train_2022_01_model:
	pipenv run python -m duration_prediction.main --train-month 2022-01 --validation-month 2022-02 --model-save-path models/2022-01.bin