from pathlib import Path

import pandas as pd
import requests
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_FILE = PROJECT_ROOT / "DATA" / "PROCESSED" / "cars_features.csv"

API_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CarMarket AI",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    if not FEATURES_FILE.exists():
        st.error(
            f"Feature dataset not found:\n{FEATURES_FILE}"
        )
        st.stop()

    return pd.read_csv(FEATURES_FILE)


cars = load_data()


# ============================================================
# IDENTIFY TARGET COLUMN
# ============================================================

possible_targets = [
    "selling_price",
    "price",
    "resale_price"
]

PRICE_COLUMN = None

for column in possible_targets:
    if column in cars.columns:
        PRICE_COLUMN = column
        break

if PRICE_COLUMN is None:
    st.error(
        "Could not identify the price column in the dataset."
    )
    st.stop()


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    column
    for column in cars.columns
    if column != PRICE_COLUMN
    and column != "log_price"
]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚗 CarMarket AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Explore Cars",
        "Compare Cars",
        "Price Prediction"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("🚗 CarMarket AI Dashboard")

    st.markdown(
        """
        ### Used Car Price Prediction System

        CarMarket AI uses machine learning to estimate the
        market price of used cars based on vehicle and
        market-related features.
        """
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Cars",
            f"{len(cars):,}"
        )

    with col2:
        st.metric(
            "Average Price",
            f"₹{cars[PRICE_COLUMN].mean():,.0f}"
        )

    with col3:
        st.metric(
            "Median Price",
            f"₹{cars[PRICE_COLUMN].median():,.0f}"
        )

    with col4:
        st.metric(
            "Features",
            len(FEATURE_COLUMNS)
        )

    st.divider()

    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        cars.head(10),
        use_container_width=True
    )

    # --------------------------------------------------------
    # PRICE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("💰 Price Distribution")

    st.bar_chart(
        cars[PRICE_COLUMN].value_counts().sort_index().head(30)
    )


# ============================================================
# EXPLORE CARS
# ============================================================

elif page == "Explore Cars":

    st.title("🔎 Explore Cars")

    filtered_cars = cars.copy()

    # --------------------------------------------------------
    # CITY FILTER
    # --------------------------------------------------------

    if "city" in cars.columns:

        st.subheader("📍 Filter by City")

        cities = sorted(
            cars["city"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_city = st.multiselect(
            "Select City",
            cities,
            default=[]
        )

        if selected_city:
            filtered_cars = filtered_cars[
                filtered_cars["city"].astype(str).isin(selected_city)
            ]

    # --------------------------------------------------------
    # OTHER CATEGORICAL FILTERS
    # --------------------------------------------------------

    categorical_columns = cars.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in categorical_columns
        if column != "city"
    ]

    for column in categorical_columns:

        values = sorted(
            cars[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(values) <= 50:

            selected_values = st.multiselect(
                f"Filter by {column.replace('_', ' ').title()}",
                values
            )

            if selected_values:
                filtered_cars = filtered_cars[
                    filtered_cars[column]
                    .astype(str)
                    .isin(selected_values)
                ]

    # --------------------------------------------------------
    # PRICE FILTER
    # --------------------------------------------------------

    if PRICE_COLUMN in cars.columns:

        min_price = float(cars[PRICE_COLUMN].min())
        max_price = float(cars[PRICE_COLUMN].max())

        if min_price < max_price:

            price_range = st.slider(
                "💰 Price Range",
                min_value=min_price,
                max_value=max_price,
                value=(min_price, max_price)
            )

            filtered_cars = filtered_cars[
                filtered_cars[PRICE_COLUMN].between(
                    price_range[0],
                    price_range[1]
                )
            ]

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.subheader(
        f"🚘 {len(filtered_cars):,} Cars Found"
    )

    st.dataframe(
        filtered_cars,
        use_container_width=True
    )


# ============================================================
# COMPARE CARS
# ============================================================

elif page == "Compare Cars":

    st.title("⚖️ Compare Cars")

    identity_columns = [
        "full_name",
        "name",
        "model_name",
        "model",
        "car_name"
    ]

    identity_column = None

    for column in identity_columns:
        if column in cars.columns:
            identity_column = column
            break

    if identity_column is None:

        st.warning(
            "A suitable car name/model column was not found "
            "for comparison."
        )

    else:

        car_names = (
            cars[identity_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_cars = st.multiselect(
            "Select up to 5 cars",
            car_names,
            max_selections=5
        )

        if selected_cars:

            comparison = cars[
                cars[identity_column]
                .astype(str)
                .isin(selected_cars)
            ]

            st.subheader("🚗 Selected Cars")

            st.dataframe(
                comparison,
                use_container_width=True
            )

            if PRICE_COLUMN in comparison.columns:

                st.subheader("💰 Price Comparison")

                price_comparison = (
                    comparison
                    .groupby(identity_column)[PRICE_COLUMN]
                    .mean()
                    .sort_values(ascending=False)
                )

                st.bar_chart(price_comparison)


# ============================================================
# PRICE PREDICTION
# ============================================================

elif page == "Price Prediction":

    st.title("💰 Car Price Prediction")

    st.markdown(
        """
        Enter the details of a used car below and CarMarket AI
        will estimate its market price.
        """
    )

    st.divider()

    user_features = {}

    # ========================================================
    # CITY - DEDICATED DROPDOWN
    # ========================================================

    if "city" in FEATURE_COLUMNS:

        st.subheader("📍 Location")

        cities = sorted(
            cars["city"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if cities:

            selected_city = st.selectbox(
                "City",
                cities,
                index=0
            )

            user_features["city"] = selected_city

    # ========================================================
    # OTHER FEATURES
    # ========================================================

    remaining_features = [
        column
        for column in FEATURE_COLUMNS
        if column != "city"
    ]

    # Create two columns for cleaner UI

    left_column, right_column = st.columns(2)

    for index, column in enumerate(remaining_features):

        if column not in cars.columns:
            continue

        series = cars[column]

        # ----------------------------------------------------
        # NUMERIC FEATURE
        # ----------------------------------------------------

        if pd.api.types.is_numeric_dtype(series):

            median_value = float(series.median())

            if index % 2 == 0:

                with left_column:

                    user_features[column] = st.number_input(
                        column.replace("_", " ").title(),
                        value=median_value
                    )

            else:

                with right_column:

                    user_features[column] = st.number_input(
                        column.replace("_", " ").title(),
                        value=median_value
                    )

        # ----------------------------------------------------
        # CATEGORICAL FEATURE
        # ----------------------------------------------------

        else:

            values = sorted(
                series
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if not values:
                continue

            if index % 2 == 0:

                with left_column:

                    user_features[column] = st.selectbox(
                        column.replace("_", " ").title(),
                        values
                    )

            else:

                with right_column:

                    user_features[column] = st.selectbox(
                        column.replace("_", " ").title(),
                        values
                    )

    st.divider()

    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    predict_button = st.button(
        "🚀 Predict Car Price",
        use_container_width=True
    )

    if predict_button:

        try:

            with st.spinner("Calculating estimated market price..."):

                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "features": user_features
                    },
                    timeout=30
                )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if response.status_code == 200:

                result = response.json()

                predicted_price = result["predicted_price"]

                st.success("Prediction completed successfully!")

                st.metric(
                    "Estimated Car Price",
                    f"₹{predicted_price:,.2f}"
                )

                st.info(
                    "This is a machine-learning estimate based "
                    "on the features provided."
                )

            # ------------------------------------------------
            # VALIDATION ERROR
            # ------------------------------------------------

            elif response.status_code == 422:

                st.error(
                    f"Invalid input:\n\n{response.text}"
                )

            # ------------------------------------------------
            # SERVER ERROR
            # ------------------------------------------------

            else:

                st.error(
                    f"API Error {response.status_code}:\n\n"
                    f"{response.text}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the FastAPI server.\n\n"
                "Please start FastAPI first using:\n\n"
                "`uvicorn api.main:app --reload`"
            )

        except Exception as exc:

            st.error(
                f"Prediction failed:\n\n{str(exc)}"
            )