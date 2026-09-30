from pathlib import Path
import pandas as pd
import numpy as np


def get_project_root():
    """
    Return the project root directory.

    Assumes this file is located at:
    CarMarket-AI/src/preprocessing.py
    """
    return Path(__file__).resolve().parent.parent


def get_paths():
    """
    Return commonly used project paths.
    """

    project_root = get_project_root()

    paths = {
        "project_root": project_root,
        "data_dir": project_root / "DATA",
        "raw_dir": project_root / "DATA" / "RAW",
        "processed_dir": project_root / "DATA" / "PROCESSED",
        "models_dir": project_root / "MODELS",
        "raw_file": project_root / "DATA" / "RAW" / "cars.csv",
        "clean_file": project_root / "DATA" / "PROCESSED" / "cars_clean.csv",
        "features_file": project_root / "DATA" / "PROCESSED" / "cars_features.csv",
        "model_file": project_root / "MODELS" / "car_price_model.joblib",
    }

    return paths


def load_raw_data(file_path=None):
    """
    Load the raw car dataset.

    Parameters
    ----------
    file_path : str or Path, optional
        Path to the raw CSV file.

    Returns
    -------
    pandas.DataFrame
    """

    paths = get_paths()

    if file_path is None:
        file_path = paths["raw_file"]

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    return df


def standardize_column_names(df):
    """
    Standardize dataframe column names.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


def clean_string_values(df):
    """
    Remove unnecessary whitespace from string columns.
    """

    df = df.copy()

    object_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in object_columns:
        df[column] = df[column].apply(
            lambda value:
            value.strip()
            if isinstance(value, str)
            else value
        )

    return df


def standardize_missing_values(df):
    """
    Convert common missing-value representations to NaN.
    """

    df = df.copy()

    missing_tokens = [
        "",
        " ",
        "NA",
        "N/A",
        "na",
        "n/a",
        "null",
        "NULL",
        "None",
        "none",
        "-"
    ]

    df = df.replace(
        missing_tokens,
        np.nan
    )

    return df


def remove_duplicates(df):
    """
    Remove duplicate rows.
    """

    df = df.copy()

    before = len(df)

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    removed = before - len(df)

    print(
        f"Removed {removed} duplicate rows."
    )

    return df


def remove_invalid_values(df):
    """
    Remove obviously invalid numeric values.

    Rules are applied only when the corresponding
    columns exist in the dataset.
    """

    df = df.copy()

    rules = {
        "year": lambda x: x > 0,
        "selling_price": lambda x: x > 0,
        "price": lambda x: x > 0,
        "resale_price": lambda x: x > 0,
        "km_driven": lambda x: x >= 0,
        "kilometers_driven": lambda x: x >= 0,
        "distance_travelled": lambda x: x >= 0,
    }

    for column, rule in rules.items():

        if column not in df.columns:
            continue

        before = len(df)

        df = df[
            df[column].isna() |
            rule(df[column])
        ]

        removed = before - len(df)

        if removed > 0:
            print(
                f"{column}: removed "
                f"{removed} invalid rows."
            )

    return df.reset_index(drop=True)


def fill_missing_values(df):
    """
    Fill missing numerical values with median
    and categorical values with mode.
    """

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in numeric_columns:

        if df[column].isna().sum() > 0:
            df[column] = df[column].fillna(
                df[column].median()
            )

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


def clean_data(df):
    """
    Complete preprocessing pipeline.
    """

    print("Starting preprocessing...")

    df = standardize_column_names(df)

    df = standardize_missing_values(df)

    df = clean_string_values(df)

    df = remove_duplicates(df)

    df = remove_invalid_values(df)

    df = fill_missing_values(df)

    print(
        f"Final cleaned shape: {df.shape}"
    )

    return df


def save_clean_data(df, file_path=None):
    """
    Save cleaned dataset.
    """

    paths = get_paths()

    if file_path is None:
        file_path = paths["clean_file"]

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
        f"Clean dataset saved to: {file_path}"
    )


def main():
    """
    Run preprocessing from command line.
    """

    df = load_raw_data()

    print(
        f"Raw dataset shape: {df.shape}"
    )

    df = clean_data(df)

    save_clean_data(df)

    print("Preprocessing completed.")


if __name__ == "__main__":
    main()