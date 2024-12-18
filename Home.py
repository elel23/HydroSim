import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def set_page_config():
    """Set the initial page configuration."""
    st.set_page_config(
        page_icon="https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/logo_hijau.png?raw=true",
        page_title="Hydrosim - Home",
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
        <img src='https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/new_banner_800.png?raw=true' class='header-image'/>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar with navigation."""
    with st.sidebar:
        st.markdown(
            "![Logo](https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/new_hijau.png?raw=true)"
        )


def generate_kfold_summary(mean_rmse):
    """
    Generate conclusions based on K-Fold RMSE results for each variable.

    Parameters:
    - mean_rmse: Dictionary of mean RMSE values for each variable.

    Returns:
    - List of conclusions for each variable.
    """
    conclusions = []
    for variable, rmse in mean_rmse.items():
        conclusions.append(
            f"- **{variable}**: Rata-rata RMSE dari K-Fold adalah **{rmse:.4f}**."
        )
        if rmse < 1.5:
            conclusions.append(
                f"  ✔️ Model memiliki akurasi yang sangat baik dalam memprediksi {variable}."
            )
        elif 1.5 <= rmse < 3.0:
            conclusions.append(
                f"  ⚠️ Model memiliki akurasi yang cukup baik untuk {variable}, tetapi masih bisa ditingkatkan."
            )
        else:
            conclusions.append(
                f"  ❌ Model memiliki akurasi yang kurang baik untuk {variable}. Perlu dilakukan tuning untuk meningkatkan performa."
            )
    return conclusions


def main():
    set_page_config()
    inject_custom_css()

    render_sidebar()

    st.title("🌿 Welcome to the Home Page")
    st.header("🎯 Tujuan")
    st.write(
        """
    HydroSim bertujuan untuk **meramalkan pertumbuhan tanaman hidroponik** dengan menggunakan data historis yang meliputi jumlah daun serta variabel lingkungan seperti suhu, kelembapan, cahaya, pH, dan lainnya. Dengan peramalan ini, pengguna dapat mengoptimalkan kondisi pertumbuhan dan meningkatkan hasil panen.
    """
    )

    st.header("✨ Manfaat")
    st.write(
        """
    - **Prediksi Pertumbuhan**: Memberikan gambaran mengenai perkembangan tanaman hidroponik di masa depan berdasarkan data historis.
    - **Optimalisasi Kondisi**: Membantu dalam menyesuaikan variabel lingkungan untuk mencapai pertumbuhan tanaman yang maksimal.
    - **Pengambilan Keputusan yang Lebih Baik**: Memungkinkan pengguna untuk membuat keputusan yang lebih tepat dalam mengelola kebun hidroponik.
    - **Efisiensi Waktu dan Sumber Daya**: Mengurangi risiko kegagalan tanam dan menghemat sumber daya melalui pemantauan yang lebih efektif.
    """
    )

    # Set up the two columns layout with different widths
    st.header("🔍 Kesimpulan")
    # col1, col2 = st.columns([3, 7])  # 70% and 30%

    # Content for the first column
    # with col1:
    st.markdown(
        "Model forecasting menggunakan algoritma Prophet menghasilkan metrik evaluasi sebagai berikut:"
    )

    st.markdown("- **RMSE (Root Mean Square Error)**: 1.66")
    st.markdown("- **MAE (Mean Absolute Error)**: 1.26")

    st.markdown(
        "Hasil menunjukkan bahwa model memiliki akurasi yang baik dengan kesalahan prediksi yang relatif rendah."
    )

    # Content for the second column
    # with col2:
    st.image(
        "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/evaluasi_model.png?raw=true",
        caption="Evaluasi Model",
    )

    # Set up the two columns layout with different widths
    st.header("🔍 Perbandingan Model ARIMA dan Prophet")

    # Data evaluasi model
    data = {
        "Model": ["ARIMA", "Prophet"],
        "RMSE": [3.19, 1.66],
        "MAE": [2.14, 1.26],
    }

    # Membuat DataFrame
    df = pd.DataFrame(data)

    col1, col2 = st.columns([3, 7])

    with col1:
        st.write("### Tabel Evaluasi Model")
        st.dataframe(df)

    with col2:
        st.image(
            "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/perbandingan_model.png?raw=true",
            caption="Evaluasi Model",
        )

    st.header("🔍 Forecasting dengan Target Variable Lingkungan")
    st.image(
        "https://github.com/Vinzzztty/V2_Forecasting_Hidroponik/blob/main/assets/forecasting_variabel_lingkungan.png?raw=true"
    )
    # Data for K-Fold Results
    k_fold_data = {
        "fold": [1, 2, 3, 4, 5],
        "WaterTemp": [0.905299, 2.465845, 1.854033, 2.756737, 2.890689],
        "Temperature": [1.089587, 1.217267, 0.930451, 1.043412, 1.157338],
        "Humidity": [4.854804, 9.410359, 7.386358, 5.284181, 5.840736],
    }

    # Create DataFrame
    df_k_fold = pd.DataFrame(k_fold_data)

    # Calculate Mean RMSE for each variable
    mean_rmse = {
        "WaterTemp": df_k_fold["WaterTemp"].mean(),
        "Temperature": df_k_fold["Temperature"].mean(),
        "Humidity": df_k_fold["Humidity"].mean(),
    }

    col3, col4 = st.columns([3, 7])

    with col3:

        # Display Unified Table
        st.subheader("K-Fold RMSE Results")
        df_display = df_k_fold.copy()
        df_display.loc["Mean"] = df_display.mean()
        st.dataframe(
            df_display.style.format(
                {"WaterTemp": "{:.4f}", "Temperature": "{:.4f}", "Humidity": "{:.4f}"}
            )
        )

    with col4:

        # Visualizations
        st.markdown("### RMSE Visualizations Across Folds")

        # Melt data for unified visualization
        df_melted = df_k_fold.melt(
            id_vars=["fold"], var_name="Variable", value_name="RMSE"
        )

        # Plotly Visualization
        fig = px.bar(
            df_melted,
            x="fold",
            y="RMSE",
            color="Variable",
            barmode="group",
            title="RMSE Across Folds for WaterTemp, Temperature, and Humidity",
            labels={"fold": "Fold", "RMSE": "Root Mean Square Error (RMSE)"},
            height=500,
        )
        fig.update_layout(legend_title_text="Variable")
        st.plotly_chart(fig)

    # Display Mean RMSE as Summary
    st.subheader("Kesimpulan Forecasting Variabel Lingkungan")
    col5, col6 = st.columns([5, 5])

    with col5:
        for variable, rmse in mean_rmse.items():
            st.write(f"- **{variable}**: Mean RMSE = {rmse:.4f}")

    with col6:
        # Generate summary
        kfold_conclusions = generate_kfold_summary(mean_rmse)

        # Display conclusions
        for conclusion in kfold_conclusions:
            st.write(conclusion)

    # Kesimpulan
    st.write("### Kesimpulan")
    st.markdown(
        """
        Hasil evaluasi menunjukkan bahwa model Prophet mengungguli model ARIMA dalam metrik RMSE dan MAE.
        Prophet memiliki RMSE (1.66) dan MAE (1.26) yang lebih rendah dibandingkan dengan ARIMA, yang memiliki RMSE (3.19) dan MAE (2.14).
        Hal ini menunjukkan bahwa model Prophet memberikan prediksi yang lebih akurat.
        """
    )


if __name__ == "__main__":
    main()
