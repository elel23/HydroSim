import streamlit as st


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
        <img src='https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/banner_800.png?raw=true' class='header-image'/>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar with navigation."""
    with st.sidebar:
        st.markdown(
            "![Logo](https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/hijau.png?raw=true)"
        )


def main():
    set_page_config()
    inject_custom_css()

    render_sidebar()

    st.header("Forecasting")


if __name__ == "__main__":
    main()


## Backup
unique_days = df["datetime"].dt.date.nunique()

        df_prophet = model.prepare_data(df_train)
        df_test = model.prepare_data(df)

        st.info(f"Total hari setelah di Tanam: {unique_days} hari")

        # Add select box for the height of the tanaman
        height_option = st.selectbox("Pilih berat tanaman (gram):", options=[100, 150])

        # Set cap value based on the selected height option
        cap = 18 if height_option == 100 else 23

        st.write(f"Cap value set to: {cap}")

        df_prophet["cap"] = cap

        # Initialize Prophet model with regressors
        models = Prophet(growth="logistic")
        models.add_regressor("hole")
        models.add_regressor("temperature")
        models.add_regressor("humidity")
        models.add_regressor("light")
        models.add_regressor("pH")
        models.add_regressor("EC")
        models.add_regressor("TDS")
        models.add_regressor("WaterTemp")

        # Fit the model
        models.fit(df_prophet)

        new_periods = MAX_DAY - unique_days

        print(len(df))
        print(new_periods)

        future = models.make_future_dataframe(
            periods=new_periods, freq="D"
        )  # Match the test data period
        future["cap"] = 18

        future["hole"] = df_test["hole"].iloc[-1]
        future["temperature"] = df_test["temperature"].iloc[-1]
        future["humidity"] = df_test["humidity"].iloc[-1]
        future["light"] = df_test["light"].iloc[-1]
        future["pH"] = df_test["pH"].iloc[-1]
        future["EC"] = df_test["EC"].iloc[-1]
        future["TDS"] = df_test["TDS"].iloc[-1]
        future["WaterTemp"] = df_test["WaterTemp"].iloc[-1]

        # Make predictions
        forecast = models.predict(future)

        # Ensure forecast includes only test dates
        forecast_test = forecast[forecast["ds"].isin(df_test["ds"])]

        # Merge forecast with actuals for evaluation using suffixes to avoid overlap
        merged = pd.merge(
            df_test, forecast_test[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds"
        )

        # Plot the results
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot actual values
        ax.plot(
            merged["ds"],
            merged["y"],
            marker="o",
            linestyle="-",
            color="blue",
            label="Actual",
        )

        # Plot forecasted values
        ax.plot(
            merged["ds"],
            merged["yhat"],
            marker="o",
            linestyle="--",
            color="red",
            label="Forecast",
        )

        # Plot the uncertainty intervals
        ax.fill_between(
            merged["ds"],
            merged["yhat_lower"],
            merged["yhat_upper"],
            color="red",
            alpha=0.2,
        )

        # Customize the plot
        ax.set_xlabel("Date")
        ax.set_ylabel("Leaf Count")
        ax.set_title("Actual vs Forecasted Leaf Count")
        ax.legend()
        ax.grid(True)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)