import numpy as np
import pandas as pd
from argparse import ArgumentParser
from boruta import BorutaPy
from catboost import CatBoostRegressor, Pool
from category_encoders import TargetEncoder
from clearml import Task
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def preprocess_data(df):
    # удаляем пропуски
    drop_rows = [
        "actual_delivery_time",
        "estimated_store_to_consumer_driving_duration",
        "market_id",
        "order_protocol",
    ]
    df.dropna(axis="index", subset=drop_rows, inplace=True)

    # заполняем оставшиеся пропуски
    market_features = {
        "total_onshift_dashers": 0,
        "total_busy_dashers": 0,
        "total_outstanding_orders": 0,
    }
    store_features = {"store_primary_category": "other"}
    fill_missing_values = market_features | store_features
    df.fillna(fill_missing_values, inplace=True)

    # рассчитаем целевой показатель
    df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"])
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["delivery_duration"] = (
        df["actual_delivery_time"] - df["created_at"]
    ).dt.total_seconds()

    # подправим типы данных
    df = df.astype({"market_id": "int64", "order_protocol": "int64"})
    col_types = {
        "market_id": "category",
        "store_primary_category": "category",
        "order_protocol": "category",
        "store_id": "category",
        "total_onshift_dashers": "int64",
        "total_busy_dashers": "int64",
        "total_outstanding_orders": "int64",
        "estimated_store_to_consumer_driving_duration": "int64",
        "delivery_duration": "int64",
    }
    df = df.astype(col_types)

    # разделим стоблцы по типу данных
    datetime_cols = df.select_dtypes("datetime64").columns.to_list()
    categorical_cols = df.select_dtypes("category").columns.to_list()
    numeric_cols = df.select_dtypes("number").columns.to_list()

    # удалим заказы, которые были созданы не в 2015 году
    time_mask = df["created_at"].dt.year == 2015
    df = df[time_mask]

    # удалим выбросы рассчитав показатель 1.5 IQR
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    mask_no_out = (
        (df[numeric_cols] >= 0)
        & (df[numeric_cols] >= lower)
        & (df[numeric_cols] <= upper)
    ).all(axis=1)
    df = df[mask_no_out]

    return df, (datetime_cols, categorical_cols, numeric_cols)


def create_pools(X_train, X_test, y_train, y_test, categorical_cols):
    train_data = Pool(X_train, y_train, cat_features=categorical_cols)
    eval_data = Pool(X_test, y_test, cat_features=categorical_cols)
    return (train_data, eval_data)


def train_model(train_data, eval_data, params):
    model = CatBoostRegressor(**params)
    model.fit(train_data, eval_set=eval_data)
    model.save_model("catboost_model")
    return model


def eval_model(model, eval_data, task):
    eval_metrics = model.eval_metrics(eval_data, ["RMSE", "MAPE"])
    RMSE, MAPE = eval_metrics["RMSE"][-1], eval_metrics["MAPE"][-1]
    task.get_logger().report_single_value("RMSE", round(RMSE, 2))
    task.get_logger().report_single_value("MAPE", round(MAPE, 2))
    task.get_logger().report_single_value("n_features", model.n_features_in_)
    return (eval_metrics["RMSE"][-1], eval_metrics["MAPE"][-1])


def build_features(df):
    df["month"] = df["created_at"].dt.month
    df["dayofweek"] = df["created_at"].dt.dayofweek + 1
    df["hour"] = df["created_at"].dt.hour
    col_types = {"month": "category", "dayofweek": "category", "hour": "category"}
    df = df.astype(col_types)
    dt_features = ["month", "dayofweek", "hour"]
    return df, dt_features


def select_features(X_fs, y_fs, categorical_cols, n_iters=20):
    X1, X2, y1, y2 = train_test_split(X_fs, y_fs, test_size=0.5)
    enc = TargetEncoder(cols=categorical_cols).fit(X1, y1)
    X2 = enc.transform(X2)
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators="auto", max_iter=n_iters, random_state=13)
    feat_selector.fit(np.array(X2), np.array(y2))
    feature_rank = pd.DataFrame(
        {
            "f_name": X_fs.columns.to_list(),
            "rnk": feat_selector.ranking_,
            "support": feat_selector.support_,
            "weak": feat_selector.support_weak_,
        }
    ).sort_values(by="rnk")
    return feature_rank


def main(params):
    file_path = "datasets/historical_data.csv"
    raw_data = pd.read_csv(file_path)

    preprocessed_dataset, coltypes = preprocess_data(raw_data)
    datetime_cols, categorical_cols, numeric_cols = coltypes
    processed_dataset, dt_features = build_features(preprocessed_dataset)
    categorical_cols += dt_features
    features_for_model = categorical_cols + numeric_cols
    ignored_features = {
        "total_items",
        "num_distinct_items",
        "min_item_price",
        "max_item_price",
        "store_primary_category",
        "estimated_order_place_duration",
    }
    task.upload_artifact("ignored_features", artifact_object=ignored_features)
    features_for_model = [f for f in features_for_model if f not in ignored_features]
    categorical_cols = [f for f in categorical_cols if f not in ignored_features]

    X = processed_dataset[features_for_model].drop("delivery_duration", axis=1)
    y = processed_dataset["delivery_duration"]

    splitted_data = train_test_split(X, y, test_size=0.15)
    X_train, X_test, y_train, y_test = splitted_data
    train_data, eval_data = create_pools(
        X_train, X_test, y_train, y_test, categorical_cols
    )

    cb_model = train_model(train_data, eval_data, params)
    eval_model(cb_model, eval_data, task)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iterations", default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.2)
    parser.add_argument("--verbose", default=100)
    parser.add_argument("--tags", nargs="+", default=[])
    args = parser.parse_args()

    task = Task.init(
        project_name="Delivery Duration Prediction",
        task_name="CatBoost",
        tags=args.tags,
    )

    params = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "verbose": args.verbose,
        "random_seed": 13,
    }
    main(params)
