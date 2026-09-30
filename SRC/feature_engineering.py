from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from .preprocessing import get_paths


def create_car_age(df):
    """
    Create car_age from year when year exists.
    """

    df = df.copy()

    if "year" in df.columns:

        current_year = datetime.now().year

        df["car_age"] = (
            current_year - df["year"]
        )

        df["car_age"] = (
            df["car_age"]
            .clip(lower=0)
        )

    return df


def identify_target_column(df):
    """
    Identify the car price target column.
    """

    target_candidates = [
        "selling_price",
        "price",
        "resale_price"
    ]

    for column in target_candidates:

        if column in df.columns:
            return column

    raise ValueError(
        "No price target column found. "
        "Expected one of: "
        f"{target_candidates}"
    )


def create_features(df):
    """
    Complete feature-engineering pipeline.
    """

    df = df.copy()

    target = identify_target_column(df)

    print(
        f"Target column identified: {target}"
    )

    # Create car age
    df = create_car_age(df)

    # Convert numeric columns where possible
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Create log price temporarily for analysis
    # but remove it before model training because
    # it is derived directly from the target.
    df["log_price"] = np.log1p(
        df[target]
    )

    # Remove target-derived feature
    df = df.drop(
        columns=["log_price"]
    )

    # Handle missing numerical values
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        if df[column].isna().sum() > 0:

            df[column] = df[column].fillna(
                df[column].median()
            )

    # Handle categorical missing values
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        if df[column].isna().sum() > 0:

            mode = df[column].mode()

            if len(mode) > 0:
                df[column] = df[column].fillna(
                    mode.iloc[0]
                )
            else:
                df[column] = df[column].fillna(
                    "Unknown"
                )

    return df


def save_features(df, file_path=None):
    """
    Save engineered features.
    """

    paths = get_paths()

    if file_path is None:
        file_path = paths["features_file"]

    file_path = Path(file_path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        file_path,
        index=False
    )

    print(
        f"Feature dataset saved to: {file_path}"
    )


def main():
    """
    Run feature engineering from command line.
    """

    paths = get_paths()

    clean_file = paths["clean_file"]

    if not clean_file.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {clean_file}"
        )

    df = pd.read_csv(
        clean_file
    )

    print(
        f"Loaded clean dataset: {df.shape}"
    )

    df = create_features(df)

    print(
        f"Feature dataset shape: {df.shape}"
    )

    print("\nFeatures:")
    print(df.columns.tolist())

    save_features(df)

    print(
        "\nFeature engineering completed."
    )


if __name__ == "__main__":
    main()