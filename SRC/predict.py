from pathlib import Path

import joblib
import pandas as pd

from .feature_engineering import (
    get_paths,
    identify_target_column
)


class CarPricePredictor:
    """
    Reusable car price prediction service.
    """

    def __init__(
        self,
        model_path=None
    ):

        paths = get_paths()

        if model_path is None:
            model_path = (
                paths["model_file"]
            )

        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Model not found: "
                f"{self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        self.feature_columns = None

        self._load_feature_columns()

    def _load_feature_columns(self):
        """
        Load the feature names used during
        training from the processed dataset.
        """

        paths = get_paths()

        feature_file = (
            paths["features_file"]
        )

        if not feature_file.exists():

            raise FileNotFoundError(
                f"Feature dataset not found: "
                f"{feature_file}"
            )

        df = pd.read_csv(
            feature_file
        )

        target = identify_target_column(
            df
        )

        self.feature_columns = [
            column
            for column in df.columns
            if column != target
        ]

        # Remove target-derived leakage
        self.feature_columns = [
            column
            for column in self.feature_columns
            if column != "log_price"
        ]

    def prepare_input(
        self,
        car_data
    ):
        """
        Convert dictionary input into a
        DataFrame matching training features.
        """

        input_data = {}

        for column in self.feature_columns:

            if column in car_data:
                input_data[column] = (
                    car_data[column]
                )

            else:
                raise ValueError(
                    f"Missing required feature: "
                    f"{column}"
                )

        input_df = pd.DataFrame(
            [input_data]
        )

        return input_df

    def predict(
        self,
        car_data
    ):
        """
        Predict car market price.

        Parameters
        ----------
        car_data : dict

        Returns
        -------
        float
            Predicted price.
        """

        input_df = self.prepare_input(
            car_data
        )

        prediction = self.model.predict(
            input_df
        )

        return float(
            prediction[0]
        )


def load_predictor():
    """
    Convenience function for FastAPI.
    """

    return CarPricePredictor()


if __name__ == "__main__":

    predictor = load_predictor()

    paths = get_paths()

    feature_file = (
        paths["features_file"]
    )

    training_data = pd.read_csv(
        feature_file
    )

    target = identify_target_column(
        training_data
    )

    # Create a generic test case using
    # median/mode values from the dataset.
    test_car = {}

    for column in predictor.feature_columns:

        if pd.api.types.is_numeric_dtype(
            training_data[column]
        ):

            test_car[column] = (
                training_data[column]
                .median()
            )

        else:

            mode = (
                training_data[column]
                .mode()
            )

            if len(mode) > 0:
                test_car[column] = (
                    mode.iloc[0]
                )
            else:
                test_car[column] = "Unknown"

    predicted_price = predictor.predict(
        test_car
    )

    print(
        "========== PREDICTION TEST =========="
    )

    print(
        f"Predicted price: "
        f"₹{predicted_price:,.2f}"
    )