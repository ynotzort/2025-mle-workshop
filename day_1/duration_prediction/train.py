#!/usr/bin/env python
# coding: utf-8

import click
import pickle
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


def read_dataframe(filename: str) -> pd.DataFrame:
    """Reads a dataframe and creates and prepares some features

    Args:
        filename (str): filepath to a parquet file

    Returns:
        pd.DataFrame: the parsed and processed dataframe
    """
    df = pd.read_parquet(filename)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


def train(train_month: datetime, validation_month: datetime, output_filename: str):
    url_template = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    train_url = url_template.format(year=train_month.year, month=train_month.month)
    val_url = url_template.format(
        year=validation_month.year, month=validation_month.month
    )
    print(f"{train_url=}")
    print(f"{val_url=}")
    df_train = read_dataframe(train_url)
    df_val = read_dataframe(val_url)

    # TODO: maybe show in logs?
    # len(df_train), len(df_val)

    categorical = ["PULocationID", "DOLocationID"]
    numerical = ["trip_distance"]
    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    val_dicts = df_val[categorical + numerical].to_dict(orient="records")

    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values

    pipeline = make_pipeline(DictVectorizer(), LinearRegression())
    pipeline.fit(train_dicts, y_train)
    y_pred = pipeline.predict(val_dicts)

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"mse: {mse}")

    with open(output_filename, "wb") as f_out:
        pickle.dump(pipeline, f_out)


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
