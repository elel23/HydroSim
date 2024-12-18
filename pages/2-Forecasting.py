import streamlit as st
import pandas as pd
from utils import model, visualization, cek_optimization
import matplotlib.pyplot as plt
import time
import warnings

# GLOBAL VARIABLE
MAX_DAY = 40


def set_page_config():
    """Set the initial page configuration."""
    st.set_page_config(
        page_icon="https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/logo_hijau.png?raw=true",
        page_title="Hydrosim - Forecasting",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_custom_css():
    """Inject custom CSS for styling."""
    st.markdown(
        """
        <style>
        /* Styling the header image */
        .header-image {
            width: 100%;
            height: auto;
        }
        
        /* Change the background color of the sidebar */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar with navigation."""
    with st.sidebar:
        st.markdown(
            "![Logo](https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/new_hijau.png?raw=true)"
        )


def handle_file_upload(option):
    """Handle CSV file upload or use example CSV."""
    if option == "Unggah file CSV":
        uploaded_file = st.file_uploader(
            "Unggah file CSV untuk dilakukan prediksi", type=["csv"]
        )
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
    elif option == "Gunakan contoh file CSV":
        url_example = "https://raw.githubusercontent.com/Vinzzztty/Forecasting-Hidroponik/main/dataset/dummy_data_test.csv"
        st.write("Menggunakan contoh file CSV dari URL")
        return pd.read_csv(url_example)
    return None


def preprocess_data(df):
    """Preprocess the input data to ensure required columns are available and properly formatted."""
    # check if 'datetime' column is not present
    if "datetime" not in df.columns:
        st.info(
            "Kolom 'datetime' tidak ditemukan, akan membuat kolom 'datetime' dari kolom 'day' dan 'time' secara otomatis!."
        )

        # Inisialisasi start_date
        start_date = pd.to_datetime("2024-07-01")

        # Ensure 'day' column exists and is of integer type
        if "day" in df.columns:
            df["day"] = df["day"].astype(int)
        else:
            st.error("Kolom 'day' tidak ditemukan pada file CSV.")

        # Ensure 'time' column exists and format it properly
        if "time" in df.columns:
            df["time"] = df["time"].apply(lambda x: "{:.2f}".format(x))
            df["time"] = pd.to_datetime(df["time"], format="%H.%M").dt.time

            # Replace dots with colons for time formatting
            df["time"] = df["time"].astype(str)

            df["datetime"] = df.apply(
                lambda row: start_date
                + pd.Timedelta(days=row["day"] - 1)
                + pd.to_timedelta(row["time"]),
                axis=1,
            )

            df = df.drop_duplicates(subset=["day", "time", "LeafCount"])

            df.set_index("datetime", inplace=True)
            df = df.sort_index()

        else:
            st.error("Kolom 'time' tidak ditemukan pada file CSV.")
            return None

        df["datetime"] = df.apply(
            lambda row: start_date
            + pd.Timedelta(days=row["day"] - 1)
            + pd.to_timedelta(row["time"]),
            axis=1,
        )

    # Convert 'datetime'column to datetime format if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        if df["datetime"].isnull().any():
            st.error("⚠️ Ada nilai yang tidak bisa dikonversi ke format datetime.")
            return None

    important_columns = [
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
    return df[important_columns]


def forecast_temperature(df, periods):
    """Forecast temperature using the saved temperature model."""
    df_temperature = model.prepare_data_with_target(df, "temperature")
    model_temperature = model.load_model("./model/temperature_prophet_model.pkl")

    # Dynamically set the cap based on domain knowledge or historical max
    df_temperature["cap"] = 18

    # Create a future dataframe for predictions
    future_temperature = model_temperature.make_future_dataframe(
        periods=periods, freq="D"
    )
    future_temperature["cap"] = 18
    for feature in [
        "hole",
        "humidity",
        "light",
        "pH",
        "EC",
        "TDS",
        "WaterTemp",
        "LeafCount",
    ]:
        if feature in df_temperature:
            future_temperature[feature] = df_temperature[feature].iloc[-1]

    # Predict future temperature
    forecast_temperature = model_temperature.predict(future_temperature)
    st.subheader("📈 Temperature Forecast")
    # st.dataframe(forecast_temperature)

    # Visualize the forecast
    fig_temperature = visualization.visualize_forecast_varible(
        forecast_temperature, "temperature", periods
    )
    st.plotly_chart(fig_temperature)

    return forecast_temperature


def forecast_humidity(df, periods):
    """Forecast humidity using the saved humidity model."""
    df_humidity = model.prepare_data_with_target(df, "humidity")
    model_humidity = model.load_model("./model/humidity_prophet_model.pkl")

    # Dynamically set the cap based on domain knowledge or historical max
    df_humidity["cap"] = 18

    future_humidity = model_humidity.make_future_dataframe(periods=periods, freq="D")

    future_humidity["cap"] = 18
    for feature in [
        "hole",
        "temperature",
        "light",
        "pH",
        "EC",
        "TDS",
        "WaterTemp",
        "LeafCount",
    ]:
        future_humidity[feature] = df_humidity[feature].iloc[-1]

    forecast_humidity = model_humidity.predict(future_humidity)
    st.subheader("📈 Humidity Forecast")

    fig_humidity = visualization.visualize_forecast_varible(
        forecast_humidity, "humidity", periods
    )
    st.plotly_chart(fig_humidity)
    return forecast_humidity


def forecast_water_temp(df, periods):
    """Forecast water temperature using the saved WaterTemp model."""
    df_water_temp = model.prepare_data_with_target(df, "WaterTemp")
    model_water_temp = model.load_model("./model/WaterTemp_prophet_model.pkl")

    df_water_temp["cap"] = 18

    future_water_temp = model_water_temp.make_future_dataframe(
        periods=periods, freq="D"
    )
    future_water_temp["cap"] = 18

    for feature in [
        "hole",
        "temperature",
        "humidity",
        "light",
        "pH",
        "EC",
        "TDS",
        "LeafCount",
    ]:
        future_water_temp[feature] = df_water_temp[feature].iloc[-1]

    forecast_water_temp = model_water_temp.predict(future_water_temp)
    st.subheader("📈 Water Temperature Forecast")

    # st.dataframe(forecast_water_temp)

    fig_water_temp = visualization.visualize_forecast_varible(
        forecast_water_temp, "WaterTemp", periods
    )
    st.plotly_chart(fig_water_temp)
    return forecast_water_temp


def forecast_growth(df):
    """Forecast the growth of leaves and temperature based on the model and user input."""
    df_prophet = model.prepare_data(df)
    models = model.load_model("./model/prophet_model.pkl")

    unique_days = df["datetime"].dt.date.nunique()
    st.info(f"🗓️ Total hari setelah di Tanam: {unique_days} hari")

    with st.spinner(text="⏳ Sedang menganalisis..."):
        time.sleep(2)

    # LeafCount forecast (requires 'cap' for logistic growth)
    future = models.make_future_dataframe(periods=unique_days, freq="D")
    for feature in [
        "hole",
        "temperature",
        "humidity",
        "light",
        "pH",
        "EC",
        "TDS",
        "WaterTemp",
    ]:
        future[feature] = df_prophet[feature].iloc[-1]
    future["cap"] = 18  # Required for logistic growth
    forecast = models.predict(future)

    # Slider for forecasting periods
    max_periods = MAX_DAY - unique_days
    periods = st.slider(
        "⏳ Pilih hari untuk Forecasting pertumbuhan daun",
        min_value=unique_days,
        max_value=max_periods,
        step=1,
    )

    # Create future dataframe for LeafCount prediction
    future = model.create_future_dataframe(df_prophet, periods=periods)
    future["cap"] = 18  # Required for logistic growth
    forecast = model.make_predictions(models, future)

    # Visualize LeafCount forecast
    st.markdown(""" --- """)
    st.markdown(f"### 📈 Hasil Forecasting untuk {periods} Hari Ke Depan")
    fig = visualization.plot_forecast(forecast, periods)
    st.plotly_chart(fig)

    # Display LeafCount forecast results
    col1, col2 = st.columns([6, 4])
    with col1:
        periods = forecast["day"].max()
        image_path = select_image_path(periods)
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="{image_path}" alt="Prediksi Jumlah Daun Selada" style="width: 50%;">
                <p>Prediksi Jumlah Daun Selada</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.write(f"📋 Tabel Prediksi")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])

    forecast_temp = forecast_temperature(df, periods)
    forecast_hum = forecast_humidity(df, periods)
    forecast_waterTemp = forecast_water_temp(df, periods)

    return df_prophet, forecast, forecast_temp, forecast_hum, forecast_waterTemp


def select_image_path(periods):
    """Select the appropriate image based on the predicted leaf count."""
    if periods <= 10:
        return "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/early_leaf.png?raw=true"
    elif periods <= 14:
        return "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/over.png?raw=true"
    elif periods <= 18:
        return "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/mid_leaf.png?raw=true"
    elif periods <= 24:
        return "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/normal.png?raw=true"
    else:
        return "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/high_leaf.png?raw=true"


def display_summary(df, df_prophet, forecast, periods):
    """Display summary of the forecasting results."""
    st.markdown(f"#### 📝 Kesimpulan")
    conclusion = cek_optimization.summarize_forecast(df, forecast, periods)
    st.info(f"\n{conclusion}")

    growth_percentage, last_leaf_count, max_forecasted_leaf_count = (
        visualization.calculate_growth_percentage(df, forecast)
    )
    fig = visualization.plot_growth_bar(
        growth_percentage, last_leaf_count, max_forecasted_leaf_count
    )
    st.plotly_chart(fig)

    st.markdown("##### 🔍 Kesimpulan Masing Masing Variabel")
    suggestions = cek_optimization.check_optimization(
        pd.merge(df_prophet, forecast, on="ds")
    )

    if suggestions:
        # Extract the variable names from suggestions
        variable_names = [suggestion.split()[2] for suggestion in suggestions]

        # Filter to include only '_x' columns and remove '_x' for display
        important_columns = [name for name in variable_names if name.endswith("_x")]
        display_names = [name.replace("_x", "") for name in important_columns]

        # Display selectbox for variable selection with cleaned names
        selected_display_name = st.selectbox(
            "Pilih variabel untuk melihat kesimpulan:", display_names
        )

        # Add '_x' back to match the original variable names in `suggestions`
        selected_variable = selected_display_name + "_x"

        # Find and display the conclusion for the selected variable
        selected_conclusion = next(
            (
                suggestion
                for suggestion in suggestions
                if selected_variable in suggestion
            ),
            "Kesimpulan tidak ditemukan.",
        )
        st.write(selected_conclusion)
    else:
        st.subheader(
            "✅ Semua variabel berada dalam kondisi optimal untuk pertumbuhan tanaman selada."
        )


def main():
    set_page_config()
    inject_custom_css()
    render_sidebar()

    st.title("Welcome to Forecasting Page")
    option = st.radio(
        "Pilih metode input data:", ("Unggah file CSV", "Gunakan contoh file CSV")
    )
    df = handle_file_upload(option)

    if df is not None:
        df = preprocess_data(df)
        if df is not None:
            st.markdown("### 📊 Data tanaman yang di Upload")
            st.dataframe(df)
            df_prophet, forecast, forecast_temp, forecast_hum, forecast_waterTemp = (
                forecast_growth(df)
            )

            forecasts = {
                "temperature": forecast_temp,
                "humidity": forecast_hum,
                "WaterTemp": forecast_waterTemp,
            }

            display_summary(df, df_prophet, forecast, periods=MAX_DAY)

            ##
            suggestions = cek_optimization.check_optimization_forecast_with_suggestions(
                forecasts
            )

            # Display suggestions in Streamlit
            st.markdown("### Saran Variabel Lingkungan")
            if suggestions:
                for suggestion in suggestions:
                    st.write(suggestion)
            else:
                st.write("Tiidak ada Saran. Semua Nilai Variabel Sudah Optimal.")
            ##

            st.markdown("### 🔎 Detail Variabel")
            selected_feature = st.selectbox(
                "🎯 Pilih fitur untuk divisualisasikan:", df.columns[1:]
            )
            visualization.visualize_feature(df, selected_feature)

            st.markdown("#### 🆚 Visualisasi Perbandingan Fitur")
            feature_a = st.selectbox("Pilih Fitur A", df.columns[1:])
            feature_b = st.selectbox("Pilih Fitur B", df.columns[2:])
            if feature_a and feature_b:
                visualization.visualize_comparison(df, feature_a, feature_b)

            # Add Quality Prediction Section
            st.markdown(f"#### Pola Pertumbuhan Tanaman Selada")

            # Display loading spinner while the model is being loaded
            with st.spinner("Loading model..."):
                # Load Model Pola Pertumbuhan Tanaman Selada
                model_quality, accuracy = model.quality_model()

            st.write("Enter the values for prediction")
            # Create two columns for inputs
            col5, col6 = st.columns(2)

            with col5:
                temperature_2 = st.number_input(
                    "Temperature", format="%.2f", value=25.9, step=0.01
                )
                humidity_2 = st.number_input("Humidity", value=84, step=1)
                light_2 = st.number_input("Light", value=10870, step=1)

            with col6:
                pH_2 = st.number_input("pH", format="%.2f", value=6.6, step=0.01)
                EC_2 = st.number_input("EC", value=983, step=1)
                TDS_2 = st.number_input("TDS", value=493, step=1)
                WaterTemp_2 = st.number_input(
                    "Water Temperature", format="%.2f", value=26.3, step=0.01
                )

            # Create input data for prediction
            input_data = {
                "temperature": temperature_2,
                "humidity": humidity_2,
                "light": light_2,
                "pH": pH_2,
                "EC": EC_2,
                "TDS": TDS_2,
                "WaterTemp": WaterTemp_2,
            }

            # Make prediction
            if st.button("Predict"):
                prediction_result = model.predict_pattern(model_quality, input_data)
                st.write(f"Predicted Quality: {prediction_result}")
        else:
            st.write("Silakan unggah file CSV terlebih dahulu.")


if __name__ == "__main__":
    main()
