import pandas as pd
import numpy as np
import streamlit as st
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# --- CRITICAL FIX: CORRECTED DATA PATHS ---
# Using '../data/' to access files one level up from the 'app' folder
file_names = {
    "Benin": "../data/benin-malamville.csv",
    "Sierra Leone": "../data/sierraleone-bumbuna.csv",
    "Togo": "../data/togo-dapaong-qc.csv"
}

@st.cache_data
def load_data(country):
    """Loads and preprocesses the data for the given country."""
    try:
        # Load the data using the corrected path
        df = pd.read_csv(
            file_names[country],
            index_col='Timestamp (UTC)',
            parse_dates=True
        )

        # Basic Preprocessing (as done in notebooks)
        df.replace(-9999, np.nan, inplace=True)
        df.dropna(subset=['GHI', 'DHI', 'DNI'], inplace=True)
        
        # Filter daytime zeros (simple filter for dashboard)
        df_filtered = df[(df['GHI'] > 0) | (df.index.hour < 6) | (df.index.hour > 18)].copy()
        
        # Feature Engineering (Month, DayOfWeek for plotting)
        df_filtered['Month'] = df_filtered.index.to_period('M')
        df_filtered['Month_Name'] = df_filtered.index.strftime('%Y-%m')
        
        return df_filtered
        
    except FileNotFoundError:
        # Added error handling to show up in Streamlit if the file is still missing
        st.error(f"Error: Data file for {country} not found at {file_names[country]}.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        return None

def apply_savgol_filter(series, window_length=51, polyorder=3):
    """Applies Savitzky-Golay filter for smoothing."""
    # Ensure data is numeric and handle NaNs for filtering
    data = series.fillna(series.median()).values
    return savgol_filter(data, window_length, polyorder)

def plot_daily_profile(df, country):
    """Generates a Plotly chart for the average daily profile of GHI."""
    df['Hour'] = df.index.hour
    daily_avg = df.groupby('Hour')['GHI'].mean().reset_index()

    fig = go.Figure(data=[
        go.Bar(
            x=daily_avg['Hour'], 
            y=daily_avg['GHI'], 
            marker_color='rgb(255,165,0)'
        )
    ])
    fig.update_layout(
        title=f'Average Hourly GHI Profile in {country}',
        xaxis_title='Hour of Day (24h)',
        yaxis_title='Average GHI (W/m²)',
        xaxis=dict(tickmode='linear', dtick=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_forecasting_model(df):
    """Creates and evaluates a simple linear regression model for DHI prediction."""
    
    # 1. Define Features (X) and Target (y)
    # Simple model using GHI, DNI, and Time features (Hour, Month, Day of week)
    df_model = df.copy()
    df_model['Hour_sin'] = np.sin(2 * np.pi * df_model.index.hour / 24)
    df_model['Month_sin'] = np.sin(2 * np.pi * df_model.index.month / 12)

    features = ['GHI', 'DNI', 'Hour_sin', 'Month_sin']
    target = 'DHI'

    # Filter out NaNs if any were introduced during feature creation (should be few)
    df_model.dropna(subset=features + [target], inplace=True)

    X = df_model[features]
    y = df_model[target]

    # 2. Time Series Split (chronological split - use the last 20% for testing)
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    # 3. Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Predict and Evaluate
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2, y_test.index, y_pred