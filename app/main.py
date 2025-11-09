import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import load_data, plot_daily_profile, apply_savgol_filter, create_forecasting_model

# --- Configuration ---
st.set_page_config(
    page_title="Solar Challenge Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Setup ---
st.sidebar.title("Dashboard Controls")
country_selection = st.sidebar.selectbox(
    "Select Country Data",
    ("Benin", "Sierra Leone", "Togo")
)

# --- Load Data and Handle Errors ---
# This calls the load_data function from utils.py (which has the corrected path)
df = load_data(country_selection)

if df is not None:
    # --- Main Page Title and Overview ---
    st.title(f"Solar Irradiance Dashboard: {country_selection}")
    st.markdown("A preliminary analysis of solar data including smoothing and forecasting.")

    # --- Data Filtering and Display ---
    st.subheader("Data Overview")
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Start Date", df.index.min().strftime('%Y-%m-%d'))
    col3.metric("End Date", df.index.max().strftime('%Y-%m-%d'))
    
    if st.checkbox('Show Raw Data', False):
        st.dataframe(df.head(), use_container_width=True)

    # --- Section 1: Daily Profile (Bar Chart) ---
    st.header("1. Average Daily Solar Profile")
    st.plotly_chart(plot_daily_profile(df, country_selection), use_container_width=True)

    # --- Section 2: Time Series and Smoothing ---
    st.header("2. GHI Time Series Analysis")
    
    # Calculate smoothed GHI
    df['GHI_Smoothed'] = apply_savgol_filter(df['GHI'])
    
    # Plotting the raw and smoothed data
    fig_ts = px.line(
        df,
        y=['GHI', 'GHI_Smoothed'],
        title=f'GHI (Global Horizontal Irradiance) Time Series in {country_selection}',
        height=500
    )
    fig_ts.update_yaxes(title='GHI (W/m²)')
    fig_ts.update_layout(hovermode="x unified")
    st.plotly_chart(fig_ts, use_container_width=True)

    # --- Section 3: Simple Forecasting Model ---
    st.header("3. DHI Forecasting Model (Linear Regression)")
    
    # Run the model
    model, mse, r2, test_index, y_pred = create_forecasting_model(df)
    
    st.markdown(f"A simple Linear Regression model was trained to predict DHI based on GHI and DNI.")
    
    model_col1, model_col2 = st.columns(2)
    model_col1.metric("Model R² Score (Test Set)", f"{r2:.4f}")
    model_col2.metric("Mean Squared Error (Test Set)", f"{mse:.2f}")
    
    # Plotting the forecast vs actuals
    df_results = pd.DataFrame({
        'Actual DHI': df['DHI'].loc[test_index],
        'Predicted DHI': y_pred
    }, index=test_index)
    
    fig_pred = px.line(
        df_results.head(500), # Plotting only the first 500 points for clarity
        title='Actual vs. Predicted DHI (First 500 Test Points)',
        height=500
    )
    fig_pred.update_yaxes(title='DHI (W/m²)')
    st.plotly_chart(fig_pred, use_container_width=True)

    # --- Footer ---
    st.sidebar.markdown("---")
    st.sidebar.info("Dashboard built using Streamlit, Pandas, and Plotly.")