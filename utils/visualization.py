import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import pandas as pd
import plotly.graph_objs as go


def plot_forecast(forecast, periods):
    # Calculate the number of days since the first date in the forecast
    forecast["day"] = (forecast["ds"] - forecast["ds"].min()).dt.days + 1

    # Create a figure
    fig = go.Figure()

    # Add the forecasted values
    fig.add_trace(
        go.Scatter(
            x=forecast["day"],
            y=forecast["yhat"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="red", dash="dash"),
        )
    )

    # Add the uncertainty intervals
    fig.add_trace(
        go.Scatter(
            x=forecast["day"].tolist() + forecast["day"][::-1].tolist(),
            y=forecast["yhat_upper"].tolist() + forecast["yhat_lower"][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(color="rgba(255, 255, 255, 0)"),
            showlegend=False,
            name="Uncertainty Interval",
        )
    )

    # Highlight the maximum forecast point
    max_y = forecast["yhat"].max()
    max_date = forecast.loc[forecast["yhat"].idxmax(), "day"]
    fig.add_trace(
        go.Scatter(
            x=[max_date],
            y=[max_y],
            mode="markers+text",
            name="Puncak Perkiraan",
            text=["Puncak Perkiraan"],
            textposition="top center",
            marker=dict(color="red", size=10),
        )
    )

    # Add animation frames
    frames = [
        go.Frame(
            data=[
                go.Scatter(
                    x=forecast["day"][:i],
                    y=forecast["yhat"][:i],
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="red", dash="dash"),
                )
            ],
            name=str(i),
        )
        for i in range(1, len(forecast) + 1)
    ]

    fig.update(frames=frames)

    # Customize the layout
    fig.update_layout(
        title=f"Perkiraan Jumlah Daun untuk {periods} Hari ke Depan",
        xaxis_title="Hari",
        yaxis_title="Jumlah Daun",
        legend=dict(font=dict(size=12)),
        sliders=[
            {
                "active": 0,
                "steps": [
                    {
                        "label": str(i),
                        "method": "animate",
                        "args": [
                            [str(i)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 500, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for i in range(1, len(forecast) + 1)
                ],
                "transition": {"duration": 0},
            }
        ],
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
    )

    # Automatically start the animation by simulating a button click on render
    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ]
    )

    return fig


def calculate_growth_percentage(df, forecast):
    # Last actual leaf count from the input data
    last_leaf_count = df["LeafCount"].iloc[-1]

    # Max forecasted leaf count
    max_forecasted_leaf_count = forecast["yhat"].max()

    # Calculate percentage increase
    growth_percentage = (
        (max_forecasted_leaf_count - last_leaf_count) / last_leaf_count
    ) * 100

    return growth_percentage, last_leaf_count, max_forecasted_leaf_count


def plot_growth_bar(
    growth_percentage, last_leaf_count, max_forecasted_leaf_count, days=40
):
    fig = go.Figure()

    # Add bars for initial and forecasted leaf count
    fig.add_trace(
        go.Bar(
            x=["Hari Terakhir", f"Hari ke-{days} (Forecast)"],
            y=[last_leaf_count, max_forecasted_leaf_count],
            text=[
                f"{last_leaf_count:.0f} Daun",
                f"{max_forecasted_leaf_count:.0f} Daun (+{growth_percentage:.2f}%)",
            ],
            textposition="auto",
            marker=dict(color=["blue", "red"]),
            name="Jumlah Daun",
        )
    )

    # Customize layout
    fig.update_layout(
        title=f"Kenaikan Persentase Jumlah Daun Selama {days} Hari ke Depan",
        xaxis_title="Hari",
        yaxis_title="Jumlah Daun",
        template="plotly_white",
        yaxis=dict(
            range=[0, max(max_forecasted_leaf_count * 1.2, last_leaf_count * 1.2)]
        ),
    )

    return fig


def visaulize_all_features(df):
    # Convert datetime to 'day' since the start of data collection
    df["day"] = (df["datetime"] - df["datetime"].min()).dt.days + 1

    # List of features to visualize
    features = [
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

    # Group by 'day' and calculate the mean of each feature
    daily_means = df.groupby("day")[features].mean().reset_index()

    # Loop through each feature and create a separate plot
    for feature in features:
        fig = go.Figure()

        # Add the mean feature data as a trace
        fig.add_trace(
            go.Scatter(
                x=daily_means["day"],
                y=daily_means[feature],
                mode="lines+markers",
                name=feature,
                line=dict(width=2),
            )
        )

        # Customize layout
        fig.update_layout(
            title=f"Rata-rata '{feature}' Terhadap Hari",
            xaxis_title="Hari",
            yaxis_title=f"Rata-rata {feature}",
            legend=dict(title="Fitur", orientation="h"),
            hovermode="x unified",
            xaxis=dict(
                tickmode="linear", tick0=0, dtick=1
            ),  # Ensure x-axis shows each day
        )

        # Display each figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)


def visualize_feature(df, selected_feature):
    # Convert datetime to 'day' since the start of data collection
    df["day"] = (df["datetime"] - df["datetime"].min()).dt.days + 1

    if selected_feature:
        # Group by 'day' and calculate the mean of the selected feature
        daily_means = df.groupby("day")[selected_feature].mean().reset_index()

        # Calculate the total average of the selected feature
        total_average = daily_means[selected_feature].mean()

        # Create a figure for the selected feature
        fig = go.Figure()

        # Add the mean feature data as a trace
        fig.add_trace(
            go.Scatter(
                x=daily_means["day"],
                y=daily_means[selected_feature],
                mode="lines+markers",
                name=selected_feature,
                line=dict(width=2),
            )
        )

        # Customize layout
        fig.update_layout(
            title=f"📈 Rata-rata '{selected_feature}' Terhadap Hari",
            xaxis_title="Hari",
            yaxis_title=f"Rata-rata {selected_feature}",
            legend=dict(title="Fitur", orientation="h"),
            hovermode="x unified",
            xaxis=dict(
                tickmode="linear", tick0=0, dtick=1  # Ensure x-axis shows each day
            ),
        )

        # Display the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Display additional information
        st.info(
            f"✨ **Informasi Fitur `{selected_feature}`:**\n"
            f"- **Rata-rata harian**: {total_average:.2f} 🌟\n"
            f"\n**Keterangan:**\n"
            f"Data ini memberikan wawasan berharga tentang bagaimana `{selected_feature}` berubah seiring waktu. "
            f"Analisis ini membantu dalam memahami pola dan tren yang dapat digunakan untuk keputusan yang lebih baik. 🚀"
        )
    else:
        st.write("🔍 Pilih fitur untuk divisualisasikan.")


def visualize_comparison(df, feature_a, feature_b):

    # Menghitung rata-rata dari setiap fitur
    mean_feature_a = df[feature_a].mean()
    mean_feature_b = df[feature_b].mean()

    # Menghitung jumlah hari sejak tanggal pertama
    df["day"] = (df["datetime"] - df["datetime"].min()).dt.days + 1  # Hari pertama = 1

    # Buat figure untuk line chart
    fig = go.Figure()

    # Tambahkan trace untuk feature_a
    fig.add_trace(
        go.Scatter(
            x=df["day"],
            y=df[feature_a],
            mode="lines",
            name=f"Rata-rata {feature_a} ({mean_feature_a:.2f})",
            line=dict(color="blue"),
        )
    )

    # Tambahkan trace untuk feature_b
    fig.add_trace(
        go.Scatter(
            x=df["day"],
            y=df[feature_b],
            mode="lines",
            name=f"Rata-rata {feature_b} ({mean_feature_b:.2f})",
            line=dict(color="green"),
        )
    )

    # Sesuaikan layout
    fig.update_layout(
        title=f"Perbandingan '{feature_a}' vs '{feature_b}' (Rata-Rata)",
        xaxis_title="Day",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
    )

    # Tampilkan plot di Streamlit
    st.plotly_chart(fig, use_container_width=True)


def visualize_forecast_varible(forecast, target_variable, periods):
    """
    Visualize forecast results using Plotly.

    Parameters:
    - forecast: DataFrame with 'ds', 'yhat', 'yhat_lower', 'yhat_upper'.
    - target_variable: Name of the variable being forecasted.
    - periods: Number of forecast periods.

    Returns:
    - fig: A Plotly Figure object.
    """
    # Ensure required columns are present
    required_columns = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    if not all(col in forecast.columns for col in required_columns):
        raise ValueError(f"Forecast DataFrame must contain {required_columns}")

    # Create a Plotly figure
    fig = go.Figure()

    # Add forecasted values
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="blue"),
        )
    )

    # Add prediction intervals
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode="lines",
            name="Upper Bound",
            line=dict(color="gray", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            mode="lines",
            name="Lower Bound",
            line=dict(color="gray", dash="dot"),
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.2)",
        )
    )

    # Customize the layout
    fig.update_layout(
        title=f"Forecast for {target_variable.capitalize()} ({periods} Days)",
        xaxis_title="Date",
        yaxis_title=target_variable.capitalize(),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )

    return fig

    # # Display the plot in Streamlit
    # st.plotly_chart(fig, use_container_width=True)

    # # Display the forecasted data in a table
    # st.write(f"### Forecasted Data for {target_variable.capitalize()}")
    # st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])
