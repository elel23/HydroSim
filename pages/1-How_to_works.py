import streamlit as st
import pandas as pd


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


def download_template_csv():
    """Generate and provide a template CSV file for download."""
    # Create a DataFrame for the CSV template
    template_data = {
        "datetime": ["2024-07-22 14:30:00", "2024-07-23 14:30:00"],
        "LeafCount": [20, 25],
        "hole": [1, 1],
        "temperature": [25.3, 26.1],
        "humidity": [60.5, 61.0],
        "light": [500, 510],
        "pH": [6.5, 6.4],
        "EC": [1.5, 1.6],
        "TDS": [700, 720],
        "WaterTemp": [23.0, 23.2],
    }

    template_df = pd.DataFrame(template_data)

    # Convert DataFrame to CSV
    csv_data = template_df.to_csv(index=False)

    # Create a download button for the template CSV
    st.download_button(
        label="📄 Download Template CSV",
        data=csv_data,
        file_name="template_hydroponic.csv",
        mime="text/csv",
    )


def main():
    set_page_config()
    inject_custom_css()

    render_sidebar()

    st.title("Welcome to How to Works Page 🛠️")

    # Tools used
    with st.expander("🔨 Tools Yang Digunakan"):
        st.markdown(
            """
        - `Python` (^3.11)
        - `Streamlit`: Framework dan UI
        - `Prophet`: Algoritma machine learning untuk Forecasting
        """
        )

    st.header("📊 Petunjuk Sebelum Memasukkan File CSV:")

    st.markdown(
        """
        - **Pastikan file CSV Anda memiliki kolom yang diperlukan**:
          - `datetime`: Tanggal dan waktu pengambilan data (format: `YYYY-MM-DD HH:MM:SS`)
          - `LeafCount`: Jumlah daun tanaman
          - `hole`: Jumlah lubang dalam sistem hidroponik
          - `temperature`: Suhu lingkungan (dalam °C)
          - `humidity`: Kelembaban udara (dalam %)
          - `light`: Intensitas cahaya (dalam lumen)
          - `pH`: Tingkat keasaman atau kebasaan
          - `EC`: Konduktivitas listrik, mengukur konsentrasi nutrisi
          - `TDS`: Total padatan terlarut dalam air
          - `WaterTemp`: Suhu air dalam sistem hidroponik (dalam °C)
        """
    )
    st.markdown(
        "- **Format kolom 'datetime' harus sesuai dengan format datetime standar**, yaitu `YYYY-MM-DD HH:MM:SS`. Contoh: `2024-07-22 14:30:00` 🕒."
    )
    st.markdown(
        "- **Pastikan dataset Anda mencakup data yang cukup untuk akurasi ramalan yang optimal**. Data yang dimasukkan harus mencakup **minimal `5` hari dan maksimal `40` hari** 📅."
    )
    st.markdown(
        "- Jika kolom 'datetime' tidak ada, sistem akan mencoba membuatnya dari kolom `day` dan `time` 🛠️."
    )

    st.image(
        "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/pre-processing-data.png?raw=true",
        caption="Contoh Format CSV 🗂️",
        use_column_width=True,
    )

    st.header("📄 Download Template CSV")
    download_template_csv()

    st.header("🛠️ Langkah-Langkah Penggunaan:")
    st.write("Ikuti langkah-langkah berikut untuk memulai:")
    st.markdown("1. **Unggah file CSV** yang sesuai dengan petunjuk di atas.")
    st.markdown(
        "2. **Tinjau data yang diunggah** untuk memastikan bahwa semua kolom telah terbaca dengan benar."
    )
    st.markdown("3. **Pilih parameter** yang ingin Anda analisis atau ramalkan.")
    st.markdown(
        "4. **Jalankan simulasi** untuk mendapatkan prediksi pertumbuhan tanaman hidroponik Anda."
    )
    st.markdown(
        "5. **Tinjau hasil prediksi** dan sesuaikan variabel lingkungan jika diperlukan."
    )

    st.header("💡 Tips Penggunaan:")
    st.markdown(
        "- **Simpan salinan dataset** sebelum melakukan perubahan, agar Anda memiliki data asli untuk referensi."
    )
    st.markdown(
        "- **Gunakan data terbaru** untuk mendapatkan hasil prediksi yang lebih akurat."
    )
    st.markdown(
        "- Jika Anda menemui masalah dengan format atau kolom yang hilang, **periksa kembali file CSV Anda** dan pastikan sesuai dengan panduan di atas."
    )


if __name__ == "__main__":
    main()
