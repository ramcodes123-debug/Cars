from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from .feature_engineering import (
    get_paths,
    identify_target_column
)


def load_training_data():
    """
    Load engineered training data.
    """

    paths = get_paths()

    feature_file = paths["features_file"]

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: "
            f"{feature_file}"
        )

    df = pd.read_csv(
        feature_file
    )

    return df


def prepare_data(df):
    """
    Separate features and target.
    """

    target = identify_target_column(df)

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    # Remove target-derived leakage if present
    leakage_columns = [
        "log_price"
    ]

    leakage_columns = [
        column
        for column in leakage_columns
        if column in X.columns
    ]

    if leakage_columns:

        X = X.drop(
            columns=leakage_columns
        )

        print(
            "Removed leakage columns:",
            leakage_columns
        )

    return X, y, target


def build_preprocessor(X):
    """
    Build preprocessing pipeline for
    numerical and categorical features.
    """

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


def build_models():
    """
    Return candidate regression models.
    """

    return {
        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
    }


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model.
    """

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def train_models(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train all candidate models.
    """

    preprocessor = build_preprocessor(
        X_train
    )

    models = build_models()

    trained_models = {}
    results = []

    for name, estimator in models.items():

        print(
            f"\nTraining {name}..."
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    estimator
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        trained_models[name] = pipeline

        results.append({
            "model": name,
            **metrics
        })

        print(
            f"MAE : {metrics['MAE']:,.2f}"
        )

        print(
            f"RMSE: {metrics['RMSE']:,.2f}"
        )

        print(
            f"R2  : {metrics['R2']:.4f}"
        )

    results_df = pd.DataFrame(
        results
    )

    return trained_models, results_df


def save_models(
    trained_models,
    results_df
):
    """
    Save all trained models and
    the best model.
    """

    paths = get_paths()

    models_dir = paths["models_dir"]

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for name, model in trained_models.items():

        filename = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "")
            + ".joblib"
        )

        filepath = (
            models_dir /
            filename
        )

        joblib.dump(
            model,
            filepath
        )

        print(
            f"Saved: {filepath}"
        )

    # Select model with lowest RMSE
    best_row = (
        results_df
        .sort_values(
            "RMSE"
        )
        .iloc[0]
    )

    best_model_name = (
        best_row["model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    best_model_path = (
        models_dir /
        "car_price_model.joblib"
    )

    joblib.dump(
        best_model,
        best_model_path
    )

    print(
        "\nBest model:",
        best_model_name
    )

    print(
        "Saved final model:",
        best_model_path
    )

    return best_model_name


def main():
    """
    Complete model training pipeline.
    """

    print(
        "========== CAR PRICE MODEL TRAINING =========="
    )

    df = load_training_data()

    print(
        f"Training dataset shape: {df.shape}"
    )

    X, y, target = prepare_data(
        df
    )

    print(
        f"Target column: {target}"
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows: {len(X_test)}"
    )

    trained_models, results_df = (
        train_models(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    print(
        "\n========== MODEL COMPARISON =========="
    )

    print(
        results_df.sort_values(
            "RMSE"
        ).to_string(
            index=False
        )
    )

    best_model_name = save_models(
        trained_models,
        results_df
    )

    print(
        "\nTraining completed successfully."
    )

    print(
        "Selected model:",
        best_model_name
    )


if __name__ == "__main__":
    main()