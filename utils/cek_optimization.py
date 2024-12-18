import pandas as pd


def check_optimization(df):
    # Calculate the mean of each feature
    means = df.mean().round(2)

    # Define optimal ranges for each feature
    optimal_conditions = {
        "temperature_x": (25, 28),
        "humidity_x": (50, 70),
        "light_x": (1000, 4000),
        "pH_x": (6.0, 7.0),
        "EC_x": (1200, 1800),
        "TDS_x": (560, 840),
        "WaterTemp_x": (25, 28),
    }

    # Determine if each feature is within optimal range
    def check_optimal(feature, value):
        if feature in optimal_conditions:
            lower, upper = optimal_conditions[feature]
            return lower <= value <= upper
        return True  # Always optimal if no specific range

    # Generate conclusions for each feature
    conclusions = []
    for feature, mean_value in means.items():
        is_optimal = check_optimal(feature, mean_value)

        if is_optimal:
            # Positive statement if the feature is within optimal range
            conclusions.append(
                f"✔️ Rata-rata {feature} dalam kondisi ideal pada nilai {mean_value}. Kondisi ini mendukung pertumbuhan optimal 🌱."
            )
        else:
            # Normative statement without suggesting specific ranges
            conclusions.append(
                f"⚠️ Rata-rata {feature} tercatat pada {mean_value}. Memerlukan perhatian untuk mencapai kondisi yang lebih mendukung."
            )

    return conclusions


def check_optimization_forecast_with_suggestions(forecasts):
    """
    Evaluate the forecasted means of temperature, humidity, and WaterTemp for optimal conditions
    and provide actionable suggestions if they are not optimal.

    Parameters:
    - forecasts: Dictionary containing forecast DataFrames for 'temperature', 'humidity', and 'WaterTemp'.

    Returns:
    - conclusions: List of conclusions and suggestions for each variable.
    """
    # Define optimal ranges and suggestions for each variable
    optimal_conditions = {
        "temperature": {
            "range": (25, 28),
            "suggestion_low": "Tanaman mendapatkan suhu yang cukup rendah. Pastikan tanaman ditempatkan di area dengan pencahayaan yang cukup atau gunakan pemanas ruangan untuk menjaga suhu ideal.",
            "suggestion_high": "Suhu terlalu tinggi untuk tanaman. Pertimbangkan untuk meningkatkan ventilasi atau menggunakan pendingin ruangan untuk menjaga suhu optimal.",
        },
        "humidity": {
            "range": (50, 70),
            "suggestion_low": "Kelembaban terlalu rendah. Gunakan humidifier atau penyiraman tambahan untuk meningkatkan kelembaban udara.",
            "suggestion_high": "Kelembaban terlalu tinggi. Pastikan ventilasi udara yang baik untuk menghindari pertumbuhan jamur atau bakteri.",
        },
        "WaterTemp": {
            "range": (25, 28),
            "suggestion_low": "Suhu air terlalu rendah. Gunakan alat pemanas air untuk menjaga suhu air pada tingkat yang ideal.",
            "suggestion_high": "Suhu air terlalu tinggi. Pastikan sistem pendingin air bekerja dengan baik atau gunakan air yang lebih dingin.",
        },
    }

    # Generate conclusions and suggestions
    conclusions = []
    for variable, forecast in forecasts.items():
        # Debugging: Check the forecast DataFrame structure
        if "yhat" not in forecast.columns or "ds" not in forecast.columns:
            raise ValueError(
                f"Forecast for {variable} must contain 'yhat' and 'ds' columns."
            )

        # Calculate mean of forecasted values ('yhat')
        mean_value = forecast["yhat"].mean().round(2)

        # Get optimal range and suggestions
        if variable in optimal_conditions:
            lower, upper = optimal_conditions[variable]["range"]
            suggestion_low = optimal_conditions[variable]["suggestion_low"]
            suggestion_high = optimal_conditions[variable]["suggestion_high"]

            # Check if the mean is within the optimal range
            if lower <= mean_value <= upper:
                conclusions.append(
                    f"✔️ Rata-rata {variable} dalam kondisi ideal pada nilai {mean_value}. Kondisi ini mendukung pertumbuhan optimal 🌱."
                )
            else:
                if mean_value < lower:
                    conclusions.append(
                        f"⚠️ Rata-rata {variable} terlalu rendah pada nilai {mean_value}. {suggestion_low}"
                    )
                elif mean_value > upper:
                    conclusions.append(
                        f"⚠️ Rata-rata {variable} terlalu tinggi pada nilai {mean_value}. {suggestion_high}"
                    )
        else:
            conclusions.append(
                f"❓ Tidak ada rentang optimal yang ditentukan untuk {variable}."
            )

    return conclusions


def summarize_forecast(df, forecast, periods):
    # Nilai LeafCount terakhir pada data input
    last_leaf_count = df["LeafCount"].iloc[-1]

    # Nilai tertinggi dari hasil forecasting
    max_forecasted_leaf_count = forecast["yhat"].max()

    # Hitung persentase peningkatan
    growth_percentage = (
        (max_forecasted_leaf_count - last_leaf_count) / last_leaf_count
    ) * 100

    conclusion = (
        f"🌿 **Prediksi Pertumbuhan Daun Selada** 🌿\n\n"
        f"📈 Berdasarkan simulasi pertumbuhan daun selada, diperkirakan terjadi peningkatan sebesar "
        f"**{growth_percentage:.2f}%** dari jumlah daun awal 🌱.\n\n"
        f"📅 Pada hari ke-**{periods}**, banyaknya daun diprediksi akan mencapai **{max_forecasted_leaf_count:.0f}** daun 🥬.\n\n"
        f"✨ Tetap jaga kondisi lingkungan agar prediksi pertumbuhan ini dapat tercapai! 💧☀️"
    )

    return conclusion
