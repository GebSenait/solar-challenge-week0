import pandas as pd
import numpy as np
import streamlit as st
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from pathlib import Path

# --- PATH CONFIGURATION: Use pathlib for robust path handling ---
# Get the path to the directory containing the current script (app/utils.py)
CURRENT_DIR = Path(__file__).parent
# Define the data directory path by going up one level and into the 'data' folder
DATA_DIR = CURRENT_DIR.parent / "data"

# --- FILE MAPPING: Country names to their data files ---
# FIXED: Corrected benin-malamville.csv to benin-malanville.csv
file_names = {
    "Benin": DATA_DIR / "benin-malanville.csv",
    "Sierra Leone": DATA_DIR / "sierraleone-bumbuna.csv",
    "Togo": DATA_DIR / "togo-dapaong_qc.csv"
}

@st.cache_data
def load_data(country):
    """
    Loads and preprocesses solar irradiance data for the given country.
    
    Args:
        country: Name of the country (must be in file_names)
        
    Returns:
        pd.DataFrame: Processed DataFrame with Timestamp as index, or None if error
    """
    # Validate country selection
    if country not in file_names:
        st.error(f"Error: Unknown country '{country}'. Available countries: {list(file_names.keys())}")
        return None
    
    file_path = file_names[country]
    
    if not file_path.exists():
        st.error(f"Error: Data file for {country} not found at {file_path}. Please check file existence.")
        return None
    
    try:
        raw_df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: Data file for {country} not found at {file_path}.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while reading data for {country}: {str(e)}")
        return None
    
    return prepare_dataframe(raw_df, source_name=country)


def prepare_dataframe(raw_df, source_name="Uploaded Dataset"):
    """
    Prepares a raw solar irradiance DataFrame for analysis by handling timestamps,
    cleaning, and feature engineering.
    
    Args:
        raw_df: Raw pandas DataFrame loaded from CSV or upload
        source_name: Name of the dataset (for logging/messages)
        
    Returns:
        pd.DataFrame: Processed DataFrame with Timestamp index, or None if error
    """
    try:
        df = raw_df.copy()
        
        # --- TIMESTAMP HANDLING: Robust detection ---
        timestamp_candidates = [
            'Timestamp (UTC)', 'Timestamp', 'timestamp (utc)', 'timestamp',
            'Datetime', 'Date', 'Time (UTC)', 'time', 'date_time', 'Date/Time'
        ]
        
        timestamp_col = next(
            (col for col in df.columns if col in timestamp_candidates),
            None
        )
        
        if timestamp_col is None:
            timestamp_col = next(
                (col for col in df.columns if 'timestamp' in col.lower()),
                None
            )
        
        # Attempt to combine separate date/time columns if present
        if timestamp_col is None:
            date_col = next(
                (col for col in df.columns if 'date' in col.lower() and 'time' not in col.lower()),
                None
            )
            time_col = next(
                (col for col in df.columns if 'time' in col.lower() and 'date' not in col.lower()),
                None
            )
            
            if date_col and time_col:
                df['Timestamp'] = df[date_col].astype(str) + ' ' + df[time_col].astype(str)
                timestamp_col = 'Timestamp'
                df.drop(columns=[date_col, time_col], inplace=True)
        
        if timestamp_col is None:
            st.error(f"Error: Could not find a timestamp column in {source_name} data. Available columns: {list(df.columns)}")
            return None
        
        # --- DATETIME CONVERSION: Try multiple formats for robustness ---
        datetime_formats = [
            '%Y-%m-%d %H:%M:%S',      # Standard format: '2021-08-09 00:01:00'
            '%Y-%m-%d %H:%M',         # Without seconds: '2021-08-09 00:01'
            '%m/%d/%Y %H:%M:%S',      # Excel format: '8/9/2021 00:01:00'
            '%m/%d/%Y %I:%M:%S %p',   # Excel with AM/PM: '8/9/2021 12:01:00 AM'
        ]
        
        df['dt_temp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        if df['dt_temp'].isna().sum() / len(df) > 0.3:
            for fmt in datetime_formats:
                df['dt_temp'] = pd.to_datetime(df[timestamp_col], format=fmt, errors='coerce')
                if df['dt_temp'].notna().sum() / len(df) > 0.7:
                    break
        
        if df['dt_temp'].isna().sum() / len(df) > 0.5:
            st.warning(f"Warning: Explicit timestamp parsing failed for {source_name}. Falling back to inference.")
            df['dt_temp'] = pd.to_datetime(df[timestamp_col], errors='coerce', infer_datetime_format=True)
        
        initial_len = len(df)
        df = df.dropna(subset=['dt_temp']).copy()
        dropped = initial_len - len(df)
        if dropped > 0:
            st.warning(f"Warning: Dropped {dropped} rows with invalid timestamps in {source_name}.")
        
        if len(df) == 0:
            st.error(f"Error: No valid timestamps found in {source_name} data.")
            return None
        
        df.set_index('dt_temp', inplace=True)
        df.index.name = 'Timestamp'
        
        try:
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
        except Exception as e:
            st.warning(f"Warning: Could not set UTC timezone for {source_name}. Error: {e}")
        
        if timestamp_col in df.columns and timestamp_col != 'Timestamp':
            df.drop(columns=[timestamp_col], inplace=True)
        
        return _preprocess_data(df, source_name)
    
    except ValueError as ve:
        st.error(f"Data error while preparing {source_name}: {ve}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while preparing {source_name}: {str(e)}")
        return None


def _preprocess_data(df, source_name):
    """
    Applies common data preprocessing steps: cleaning, filtering, and feature engineering.
    
    Args:
        df: DataFrame with Timestamp index
        country: Country name (for logging)
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Replace sentinel values with NaN
    df.replace(-9999, np.nan, inplace=True)
    
    # Check for required columns
    required_cols = ['GHI', 'DHI', 'DNI']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Warning: Missing required columns in {source_name} dataset: {', '.join(missing_cols)}. Some features may be limited.")
    
    # Drop rows with NaN in required columns (if they exist)
    available_cols = [col for col in required_cols if col in df.columns]
    if available_cols:
        initial_len = len(df)
        df.dropna(subset=available_cols, inplace=True)
        dropped = initial_len - len(df)
        if dropped > 0:
            st.info(f"Info: Removed {dropped} rows with missing values in required columns.")
    
    # Validate DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        st.error(f"Error: Index is not a DatetimeIndex after processing {source_name} data.")
        return df
    
    # Filter daytime zeros (keep nighttime data even if GHI is 0)
    # Only filter if GHI column exists
    if 'GHI' in df.columns:
        df = df[(df['GHI'] > 0) | (df.index.hour < 6) | (df.index.hour > 18)].copy()
    
    # Feature Engineering: Add time-based features for analysis
    df['Month'] = df.index.to_period('M')
    df['Month_Name'] = df.index.strftime('%Y-%m')
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfYear'] = df.index.dayofyear
    
    return df


def apply_savgol_filter(series, window_length=51, polyorder=3):
    """
    Applies Savitzky-Golay filter for smoothing time series data.
    
    Args:
        series: pandas Series with numeric values
        window_length: Length of the filter window (must be odd)
        polyorder: Order of the polynomial used for fitting
        
    Returns:
        numpy.ndarray: Smoothed values
    """
    # Ensure data is numeric
    if not pd.api.types.is_numeric_dtype(series):
        st.error("Error: Series must be numeric for smoothing.")
        return series.values
    
    # Handle NaNs: fill with median
    data = series.fillna(series.median()).values
    
    # Ensure window_length is odd and less than data length
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(data):
        window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
    
    # Ensure polyorder is less than window_length
    if polyorder >= window_length:
        polyorder = max(1, window_length - 1)
    
    try:
        return savgol_filter(data, window_length, polyorder)
    except Exception as e:
        st.warning(f"Warning: Could not apply Savitzky-Golay filter. Error: {e}")
        return data


def plot_daily_profile(df, country):
    """
    Generates a Plotly bar chart for the average daily profile of GHI.
    
    Args:
        df: DataFrame with Timestamp index and GHI column
        country: Country name for title
        
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    # Validate inputs
    if not isinstance(df.index, pd.DatetimeIndex):
        st.error("Error: Cannot plot daily profile - data index is not a DatetimeIndex.")
        return go.Figure()
    
    if 'GHI' not in df.columns:
        st.error("Error: Cannot plot daily profile - GHI column not found.")
        return go.Figure()
    
    # Calculate average hourly GHI
    if 'Hour' not in df.columns:
        df['Hour'] = df.index.hour
    
    daily_avg = df.groupby('Hour')['GHI'].mean().reset_index()
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=daily_avg['Hour'],
            y=daily_avg['GHI'],
            marker_color='rgb(255,165,0)',
            name='Average GHI'
        )
    ])
    
    fig.update_layout(
        title=f'Average Hourly GHI Profile in {country}',
        xaxis_title='Hour of Day (24h)',
        yaxis_title='Average GHI (W/m²)',
        xaxis=dict(tickmode='linear', dtick=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    
    return fig


def create_forecasting_model(df):
    """
    Creates and evaluates a simple linear regression model for DHI prediction.
    
    Args:
        df: DataFrame with Timestamp index and required columns (GHI, DNI, DHI)
        
    Returns:
        tuple: (model, mse, r2, test_index, y_pred) or (None, 0, 0, empty_index, empty_array) if error
    """
    # Validate inputs
    if not isinstance(df.index, pd.DatetimeIndex):
        st.error("Error: Cannot create forecasting model - data index is not a DatetimeIndex.")
        return None, 0, 0, pd.DatetimeIndex([]), np.array([])
    
    required_cols = ['GHI', 'DNI', 'DHI']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Error: Cannot create forecasting model - missing required columns: {', '.join(missing_cols)}")
        return None, 0, 0, pd.DatetimeIndex([]), np.array([])
    
    # Prepare features
    df_model = df.copy()
    
    # Create cyclical time features
    df_model['Hour_sin'] = np.sin(2 * np.pi * df_model.index.hour / 24)
    df_model['Hour_cos'] = np.cos(2 * np.pi * df_model.index.hour / 24)
    df_model['Month_sin'] = np.sin(2 * np.pi * df_model.index.month / 12)
    df_model['Month_cos'] = np.cos(2 * np.pi * df_model.index.month / 12)
    
    # Define features and target
    features = ['GHI', 'DNI', 'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos']
    target = 'DHI'
    
    # Remove rows with NaN in features or target
    df_model = df_model.dropna(subset=features + [target]).copy()
    
    if len(df_model) < 10:
        st.error("Error: Not enough data points remaining after filtering to train the model.")
        return None, 0, 0, pd.DatetimeIndex([]), np.array([])
    
    # Prepare X and y
    X = df_model[features]
    y = df_model[target]
    
    # Time series split: use last 20% for testing (chronological split)
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    test_index = y_test.index
    
    # Train model
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return model, mse, r2, test_index, y_pred
        
    except Exception as e:
        st.error(f"Error: Could not train forecasting model. {str(e)}")
        return None, 0, 0, pd.DatetimeIndex([]), np.array([])
