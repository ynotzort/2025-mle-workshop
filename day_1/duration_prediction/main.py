import logging
from datetime import datetime

import click
from duration_prediction.train import train

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@click.command()
@click.option(
    "--train-month", required=True, help="Training Month in the YYYY-MM format."
)
@click.option(
    "--validation-month", required=True, help="Validation Month in the YYYY-MM format."
)
@click.option(
    "--model-save-path", required=True, help="Path where we save the trained model"
)
def run(train_month: str, validation_month: str, model_save_path: str):
    train_year, train_month = train_month.split("-")
    train_year = int(train_year)
    train_month = int(train_month)

    validation_year, validation_month = validation_month.split("-")
    validation_year = int(validation_year)
    validation_month = int(validation_month)

    train_date = datetime(train_year, train_month, 1)
    validation_date = datetime(validation_year, validation_month, 1)
    train(
        train_month=train_date,
        validation_month=validation_date,
        output_filename=model_save_path,
    )


if __name__ == "__main__":
    run()
