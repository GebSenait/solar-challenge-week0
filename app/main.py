import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import (
    load_data,
    prepare_dataframe,
    plot_daily_profile,
    apply_savgol_filter,
    create_forecasting_model,
)


def main():
    # --- PAGE CONFIGURATION ---
    st.set_page_config(
        page_title="Solar Challenge Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- SIDEBAR: DATA SOURCE SELECTION ---
    st.sidebar.header("Data Configuration")
    st.sidebar.markdown("Choose a data source or upload your own CSV file.")

    data_mode = st.sidebar.radio(
        "Select Data Source",
        ("Sample Dataset", "Upload CSV"),
        key="data_mode_selector",
    )

    df = None
    dataset_label = None

    if data_mode == "Sample Dataset":
        dataset_label = st.sidebar.selectbox(
            "Select Country Data",
            ("Benin", "Sierra Leone", "Togo"),
            key="country_selector",
        )
        df = load_data(dataset_label)
        if df is not None:
            st.sidebar.success(f"Sample dataset for {dataset_label} loaded successfully.")
    else:
        st.sidebar.markdown("Upload a CSV file that matches the solar dataset structure.")

        upload_col1, upload_col2 = st.sidebar.columns(2)
        with upload_col1:
            uploaded_file = st.file_uploader(
                "Upload Solar CSV",
                type="csv",
                key="uploaded_solar_csv",
            )
        with upload_col2:
            st.markdown(
                """
                - Required columns: `Timestamp`, `GHI`, `DHI`, `DNI`
                - Timestamp format: `YYYY-MM-DD HH:MM[:SS]`
                """
            )

        if uploaded_file is not None:
            dataset_label = uploaded_file.name or "Uploaded Dataset"
            try:
                raw_df = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"Error reading uploaded file: {exc}")
                df = None
            else:
                df = prepare_dataframe(raw_df, source_name=dataset_label)
                if df is not None:
                    st.sidebar.success(f"File '{dataset_label}' uploaded successfully.")
        else:
            dataset_label = "Uploaded Dataset"

    # --- MAIN CONTENT ---
    if df is not None:
        try:
            # --- MAIN PAGE TITLE AND OVERVIEW ---
            st.title(f"📊 Solar Irradiance Dashboard: {dataset_label}")
            st.markdown(
                "A comprehensive analysis of solar data including daily profiles, smoothing, forecasting, "
                "and optional custom data uploads."
            )

            # --- SECTION 1: DATA OVERVIEW (KPIs) ---
            st.header("Data Overview")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Records", f"{len(df):,}")

            with col2:
                st.metric("Start Date", df.index.min().strftime("%Y-%m-%d"))

            with col3:
                st.metric("End Date", df.index.max().strftime("%Y-%m-%d"))

            with col4:
                if "GHI" in df.columns:
                    avg_ghi = df["GHI"].mean()
                    st.metric("Average GHI", f"{avg_ghi:.2f} W/m²")
                else:
                    st.metric("Average GHI", "N/A")

            # Option to show raw data
            if st.checkbox("Show Raw Data", False, key="show_raw"):
                st.subheader("Raw Data Preview")
                st.dataframe(df.head(100), use_container_width=True)

            st.markdown("---")

            # --- SECTION 2: DAILY PROFILE (Bar Chart) ---
            st.header("1. Average Daily Solar Profile")
            st.markdown(
                "This chart shows the average hourly GHI (Global Horizontal Irradiance) profile throughout the day."
            )

            if "GHI" in df.columns:
                fig_daily = plot_daily_profile(df, dataset_label)
                st.plotly_chart(fig_daily, use_container_width=True)
            else:
                st.error("Error: GHI column not found in data. Cannot display daily profile.")

            st.markdown("---")

            # --- SECTION 3: TIME SERIES AND SMOOTHING ---
            st.header("2. GHI Time Series Analysis")
            st.markdown(
                "Time series visualization of raw and smoothed GHI data using the Savitzky-Golay filter."
            )

            if "GHI" in df.columns:
                # Calculate smoothed GHI
                df["GHI_Smoothed"] = apply_savgol_filter(df["GHI"])

                # Plotting the raw and smoothed data
                fig_ts = px.line(
                    df,
                    y=["GHI", "GHI_Smoothed"],
                    title=f"GHI (Global Horizontal Irradiance) Time Series in {dataset_label}",
                    height=500,
                    labels={"value": "GHI (W/m²)", "index": "Timestamp"},
                )
                fig_ts.update_yaxes(title="GHI (W/m²)")
                fig_ts.update_xaxes(title="Timestamp")
                fig_ts.update_layout(
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    ),
                )
                st.plotly_chart(fig_ts, use_container_width=True)

                # Show smoothing statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Raw GHI Mean", f"{df['GHI'].mean():.2f} W/m²")
                with col2:
                    st.metric("Smoothed GHI Mean", f"{df['GHI_Smoothed'].mean():.2f} W/m²")
            else:
                st.error("Error: GHI column not found in data. Cannot display time series.")

            st.markdown("---")

            # --- SECTION 4: FORECASTING MODEL ---
            st.header("3. DHI Forecasting Model (Linear Regression)")
            st.markdown(
                "A Linear Regression model trained to predict DHI (Diffuse Horizontal Irradiance) based on GHI, "
                "DNI, and time-based features."
            )

            required_cols = ["GHI", "DNI", "DHI"]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if not missing_cols:
                model, mse, r2, test_index, y_pred = create_forecasting_model(df)

                if model is not None:
                    model_col1, model_col2 = st.columns(2)

                    with model_col1:
                        st.metric("Model R² Score (Test Set)", f"{r2:.4f}")

                    with model_col2:
                        st.metric("Mean Squared Error (Test Set)", f"{mse:.2f}")

                    if len(test_index) > 0 and len(y_pred) > 0:
                        df_results = pd.DataFrame(
                            {
                                "Actual DHI": df["DHI"].loc[test_index],
                                "Predicted DHI": y_pred,
                            },
                            index=test_index,
                        )

                        plot_limit = min(500, len(df_results))
                        df_plot = df_results.head(plot_limit)

                        fig_pred = px.line(
                            df_plot,
                            title=f"Actual vs. Predicted DHI (First {plot_limit} Test Points)",
                            height=500,
                            labels={"value": "DHI (W/m²)", "index": "Timestamp"},
                        )
                        fig_pred.update_yaxes(title="DHI (W/m²)")
                        fig_pred.update_xaxes(title="Timestamp")
                        fig_pred.update_layout(
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1,
                            ),
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Actual DHI Mean", f"{df_results['Actual DHI'].mean():.2f} W/m²")
                        with col2:
                            st.metric(
                                "Predicted DHI Mean",
                                f"{df_results['Predicted DHI'].mean():.2f} W/m²",
                            )
                else:
                    st.error("Error: Could not create forecasting model. Please check the data.")
            else:
                st.error(
                    f"Error: Missing required columns for forecasting model: {', '.join(missing_cols)}"
                )

            # --- FOOTER ---
            st.sidebar.markdown("---")
            st.sidebar.info("Dashboard built using Streamlit, Pandas, and Plotly.")

        except Exception as exc:
            st.error(f"An error occurred while processing the data: {exc}")
            st.stop()

    else:
        if data_mode == "Sample Dataset":
            st.info("👈 Select a country from the sidebar to load a sample dataset.")
        else:
            st.info("👈 Upload a CSV file in the sidebar to analyze your own solar dataset.")

        st.markdown(
            """
            ### About This Dashboard

            This dashboard provides:
            - **Daily Profile Analysis**: Average hourly GHI patterns
            - **Time Series Analysis**: Raw and smoothed GHI visualization
            - **Forecasting Model**: DHI prediction using Linear Regression

            Choose a data source from the sidebar to get started!
            """
        )


if __name__ == "__main__":
    main()
