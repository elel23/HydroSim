from prophet import Prophet
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st


def prepare_data(df):
    df_prophet = df[
        [
            "datetime",
            "LeafCount",
            "hole",
            "temperature",
            "humidity",
            "light",
            "pH",
            "EC",
            "TDS",
            "WaterTemp",
        ]
    ].copy()

    df_prophet.rename(columns={"datetime": "ds", "LeafCount": "y"}, inplace=True)

    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    return df_prophet


def prepare_data_with_target(df, target_column):
    df_prophet = df[
        [
            "datetime",
            "LeafCount",
            "hole",
            "temperature",
            "humidity",
            "light",
            "pH",
            "EC",
            "TDS",
            "WaterTemp",
        ]
    ].copy()

    df_prophet.rename(columns={"datetime": "ds", target_column: "y"}, inplace=True)

    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    return df_prophet
    df_prophet = df[
        [
            "datetime",
            "LeafCount",
            "hole",
            "temperature",
            "humidity",
            "light",
            "pH",
            "EC",
            "TDS",
            "WaterTemp",
        ]
    ].copy()

    df_prophet.rename(columns={"datetime": "ds", "WaterTemp": "y"}, inplace=True)

    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    return df_prophet


def load_model(model_path):
    model_loaded = joblib.load(model_path)

    return model_loaded


def create_future_dataframe(df_test, periods):
    future_dates = pd.date_range(start=df_test["ds"].max(), periods=periods, freq="D")
    last_row = df_test.iloc[-1]

    future = pd.DataFrame({"ds": future_dates})
    for col in [
        "hole",
        "temperature",
        "humidity",
        "light",
        "pH",
        "EC",
        "TDS",
        "WaterTemp",
    ]:
        future[col] = last_row[col]
    return future


def create_future_dataframe_temperature(df_test, periods):
    future_dates = pd.date_range(start=df_test["ds"].max(), periods=periods, freq="D")
    last_row = df_test.iloc[-1]

    future = pd.DataFrame({"ds": future_dates})
    for col in [
        "hole",
        "humidity",
        "light",
        "pH",
        "EC",
        "TDS",
        "WaterTemp",
        "LeafCount",
    ]:
        future[col] = last_row[col]
    return future


def make_predictions(model, future):
    forecast = model.predict(future)
    forecast[["yhat", "yhat_lower", "yhat_upper"]] = forecast[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].clip(lower=0)
    return forecast


def quality_model():
    # Load the dataset
    url = "https://raw.githubusercontent.com/Vinzzztty/Forecasting-Hidroponik/refs/heads/V2/dataset/dataset_model_kualitas.csv"
    data = pd.read_csv(url)

    # Define feature columns and target column
    feature_columns = [
        "temperature",
        "humidity",
        "light",
        "pH",
        "EC",
        "TDS",
        "WaterTemp",
    ]
    target_column = "Pattern"

    # Extract features and target
    X = data[feature_columns]  # Features
    y = data[target_column]  # Target variable

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the model
    model = GradientBoostingClassifier(learning_rate=0.1, max_depth=10)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


def predict_pattern(model, input_data):
    # Define the mapping from pattern values to descriptive labels and images
    pattern_mapping = {
        1: (
            "Pattern 1: Normal",
            "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/normal.png?raw=true",
        ),
        2: (
            "Pattern 2: Ideal",
            "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/optimal.png?raw=true",
        ),
        3: (
            "Pattern 3: Over",
            "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/over.png?raw=true",
        ),
    }

    # Ensure the input_data is a DataFrame
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    # Make the prediction
    prediction = model.predict(input_data)[0]

    # Map the prediction to its descriptive label and image URL
    prediction_label, image_url = pattern_mapping.get(
        prediction, ("Unknown Pattern", None)
    )

    # Display the corresponding image and label centered
    if image_url:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <h3>{prediction_label}</h3>
                <img src="{image_url}" alt="{prediction_label}" style="max-width: 100%;">
            </div>
            """,
            unsafe_allow_html=True,
        )

    return prediction_label


def use_saved_model_streamlit(df_test, load_model, periods):
    """
    Use a saved Prophet model to make predictions on test data and future periods in a Streamlit app.

    Parameters:
    - df_test: Test dataset (Pandas DataFrame with 'datetime' and 'y' columns)
    - load_model: Path to the saved model
    - periods: Number of periods for forecasting into the future

    Returns:
    - forecast: Forecast results as a DataFrame
    """
    # Load the saved Prophet model
    model = joblib.load(load_model)
    st.success(f"Loaded model from: {load_model}")

    # Prepare the test data
    df_test = df_test.copy()
    df_test_resampled = df_test.resample("D").mean().reset_index()
    df_test_resampled = df_test_resampled.rename(columns={"datetime": "ds"})

    # Predict on the test dataset
    test_forecast = model.predict(df_test_resampled[["ds"]])

    # Future forecasting
    future = model.make_future_dataframe(periods=periods, freq="D")
    future_forecast = model.predict(future)

    # Combine results
    forecast = pd.concat([test_forecast, future_forecast]).reset_index(drop=True)

    # Visualization in Streamlit
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add actual test data
    fig.add_trace(
        go.Scatter(
            x=df_test_resampled["ds"],
            y=df_test_resampled["y"],
            mode="lines+markers",
            name="Test Data (Actual)",
            line=dict(color="green"),
        )
    )

    # Add test forecast
    fig.add_trace(
        go.Scatter(
            x=test_forecast["ds"],
            y=test_forecast["yhat"],
            mode="lines+markers",
            name="Test Forecast",
            line=dict(color="orange"),
        )
    )

    # Add future forecast
    fig.add_trace(
        go.Scatter(
            x=future_forecast["ds"],
            y=future_forecast["yhat"],
            mode="lines+markers",
            name="Future Forecast",
            line=dict(color="blue"),
        )
    )

    # Add prediction intervals for the future forecast
    fig.add_trace(
        go.Scatter(
            x=future_forecast["ds"],
            y=future_forecast["yhat_upper"],
            mode="lines",
            name="Upper Bound",
            line=dict(color="gray", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_forecast["ds"],
            y=future_forecast["yhat_lower"],
            mode="lines",
            name="Lower Bound",
            line=dict(color="gray", dash="dot"),
            fill="tonexty",  # Fill between yhat_lower and yhat_upper
            fillcolor="rgba(128,128,128,0.2)",
        )
    )

    # Customize layout
    fig.update_layout(
        title=f"Forecast Using Model: {load_model}",
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Return the forecast data
    return forecast
