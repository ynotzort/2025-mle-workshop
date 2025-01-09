#!/usr/bin/env python
# coding: utf-8

from datetime import datetime, timedelta
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


def read_dataframe(filename):
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
    print(train_url, val_url)
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


if __name__ == "__main__":
    train(
        train_month=datetime(2022, 1, 1),
        validation_month=datetime(2022, 2, 1),
        output_filename="lin_reg.bin",
    )
