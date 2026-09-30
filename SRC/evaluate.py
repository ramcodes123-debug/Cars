from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from .feature_engineering import (
    get_paths,
    identify_target_column
)


def load_data():
    """
    Load engineered feature data.
    """

    paths = get_paths()

    feature_file = paths["features_file"]

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature file not found: "
            f"{feature_file}"
        )

    return pd.read_csv(
        feature_file
    )


def load_model():
    """
    Load saved production model.
    """

    paths = get_paths()

    model_file = paths["model_file"]

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found: {model_file}"
        )

    return joblib.load(
        model_file
    )


def prepare_test_data(df):
    """
    Recreate the same train/test split
    used during model training.
    """

    target = identify_target_column(
        df
    )

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    leakage_columns = [
        "log_price"
    ]

    leakage_columns = [
        col
        for col in leakage_columns
        if col in X.columns
    ]

    if leakage_columns:

        X = X.drop(
            columns=leakage_columns
        )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    return (
        X_test,
        y_test,
        target
    )


def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Calculate regression metrics.
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

    non_zero = y_test != 0

    if non_zero.sum() > 0:

        mape = (
            np.mean(
                np.abs(
                    (
                        y_test[non_zero]
                        -
                        predictions[non_zero]
                    )
                    /
                    y_test[non_zero]
                )
            )
            * 100
        )

    else:
        mape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "predictions": predictions
    }


def create_comparison(
    y_test,
    predictions
):
    """
    Create actual vs predicted table.
    """

    comparison = pd.DataFrame({
        "actual_price":
            y_test.values,

        "predicted_price":
            predictions
    })

    comparison["error"] = (
        comparison["actual_price"]
        -
        comparison["predicted_price"]
    )

    comparison["absolute_error"] = (
        comparison["error"].abs()
    )

    return comparison


def main():
    """
    Run model evaluation.
    """

    print(
        "========== MODEL EVALUATION =========="
    )

    df = load_data()

    model = load_model()

    X_test, y_test, target = (
        prepare_test_data(df)
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    print(
        f"\nTarget: {target}"
    )

    print(
        f"MAE : {metrics['MAE']:,.2f}"
    )

    print(
        f"RMSE: {metrics['RMSE']:,.2f}"
    )

    print(
        f"R2  : {metrics['R2']:.4f}"
    )

    print(
        f"MAPE: {metrics['MAPE']:.2f}%"
    )

    comparison = create_comparison(
        y_test,
        metrics["predictions"]
    )

    print(
        "\n========== SAMPLE PREDICTIONS =========="
    )

    print(
        comparison.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()