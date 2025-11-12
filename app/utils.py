import pandas as pd
import numpy as np
import streamlit as st
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from pathlib import Path

# --- FIX: USE PATHLIB TO ANCHOR PATHS RELATIVE TO THIS FILE ---

# Get the path to the directory containing the current script (app/utils.py)
CURRENT_DIR = Path(__file__).parent
# Now, define the data directory path by going up one level and into the 'data' folder
DATA_DIR = CURRENT_DIR.parent / "data"

file_names = {
    # CRITICAL FILE PATH FIXES:
    # 1. Benin: Ensuring file name is correct (may fix cloud case-sensitivity)
    "Benin": DATA_DIR / "benin-malamville.csv",
    "Sierra Leone": DATA_DIR / "sierraleone-bumbuna.csv",
    # 2. Togo: Fixing the hyphen to an UNDERSCORE (togo-dapaong_qc.csv)
    "Togo": DATA_DIR / "togo-dapaong_qc.csv" 
}

@st.cache_data
def load_data(country):
    """Loads and preprocesses the data for the given country."""
    
    file_path = str(file_names[country])
    
    try:
        # Load the data using the corrected path
        # Note: We defer date parsing here, as we will handle it explicitly below.
        df = pd.read_csv(file_path)
        
        # --- CRITICAL FIX FOR TIMESTAMP COLUMN (ISSUE 2) ---
        timestamp_candidates = [
            'Timestamp (UTC)', 'Timestamp', 'timestamp (utc)', 'timestamp',
            'Datetime', 'Date', 'Time (UTC)', 'time', 'date_time', 'Date/Time' # Expanded candidates
        ]
        
        # 1. Find the first matching column name
        timestamp_col = next(
            (col for col in df.columns if col in timestamp_candidates),
            None # If no candidate is found, set to None
        )
        
        if timestamp_col is None:
            # 2. Check for a single column with 'time' or 'date' (case-insensitive fallback)
            fuzzy_col = next(
                (col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()),
                None
            )
            
            if fuzzy_col:
                timestamp_col = fuzzy_col

        # --- SPECIAL HANDLING FOR SIERRA LEONE (Combine Date/Time columns) ---
        # This combines separate Date and Time columns into a single 'Timestamp' column.
        if country == "Sierra Leone":
            # Look for date and time columns, ignoring case
            date_col = next((col for col in df.columns if 'date' in col.lower() and 'time' not in col.lower()), None)
            time_col = next((col for col in df.columns if 'time' in col.lower() and 'date' not in col.lower()), None)
            
            if date_col and time_col:
                # Combine them into a new 'Timestamp' column string
                df['Timestamp'] = df[date_col].astype(str) + ' ' + df[time_col].astype(str)
                timestamp_col = 'Timestamp'
                
                # Drop the original date/time columns
                df.drop(columns=[date_col, time_col], inplace=True)

        if timestamp_col is None:
            # If after all checks, no column is found, raise the error.
            raise ValueError(f"Could not find a valid timestamp column in the data. Looked for: {timestamp_candidates}. Fallback check failed.")


        # --- CRITICAL FIX: ROBUST DATETIME CONVERSION ---
        # Define formats to try, handling the common Excel format (M/D/Y) and the standard one (Y-M-D)
        datetime_formats_to_try = [
            # 1. Excel format (M/D/Y H:M:S AM/PM) -> E.g., '8/9/2021 12:01:00 AM'
            '%m/%d/%Y %I:%M:%S %p', 
            # 2. Excel format (M/D/Y H:M:S 24hr) -> E.g., '8/9/2021 00:01:00'
            '%m/%d/%Y %H:%M:%S',
            # 3. Standard format (Y-M-D H:M:S 24hr) -> E.g., '2021-08-09 00:01:00'
            '%Y-%m-%d %H:%M:%S',
        ]
        
        # Convert the timestamp column string values to datetime objects
        # We try multiple formats until a majority of the data is parsed.
        df['dt_temp'] = df[timestamp_col]

        for fmt in datetime_formats_to_try:
            # Attempt conversion with the explicit format
            df['dt_temp'] = pd.to_datetime(df['dt_temp'], format=fmt, errors='coerce')
            
            # Check how many conversions succeeded. If > 50% succeeded, stop trying.
            if df['dt_temp'].notna().sum() / len(df) > 0.5:
                break
            
            # If still many NaT values, try next format on the original string column
            df['dt_temp'] = df[timestamp_col].copy()

        # If the loop finished and most values are still NaT, we fall back to inference
        if df['dt_temp'].isna().sum() / len(df) > 0.5:
            st.warning("Explicit format parsing failed, falling back to inference. If errors persist, a new format is needed.")
            df['dt_temp'] = pd.to_datetime(df[timestamp_col], errors='coerce')

        # Drop the original column now that we have the datetime column
        df.drop(columns=[timestamp_col], inplace=True)

        # --- FINAL INDEX SETTING (Applies to ALL countries) ---
        
        # Set the successfully converted column as the DataFrame index
        df.set_index('dt_temp', inplace=True)
        df.index.name = 'Timestamp' # Rename the index
        
        # Ensure the index is UTC localized for consistency
        try:
             df.index = df.index.tz_localize('UTC', errors='coerce')
        except:
             # If index already has TZ info, remove it before localizing
             df.index = df.index.tz_convert(None).tz_localize('UTC', errors='coerce')
        
        # Helper function to apply common filtering and feature engineering
        return df_filtered_after_special_handling(df)
        
    except FileNotFoundError:
        st.error(f"Error: Data file for {country} not found at {file_path}. Please check file existence and path.")
        return None
    except ValueError as ve:
        st.error(f"Data Column Error: {ve}")
        return None
    except Exception as e:
        # Catching the generic error and trying to provide more context
        st.error(f"An unexpected error occurred during data loading: {e}. This may be caused by an issue in setting the DatetimeIndex.")
        return None

def df_filtered_after_special_handling(df):
    """Applies common filtering and feature engineering steps."""
    
    # Basic Preprocessing (as done in notebooks)
    df.replace(-9999, np.nan, inplace=True)
    
    # Ensure GHI, DHI, DNI exist before dropping NaNs
    required_cols = ['GHI', 'DHI', 'DNI']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Warning: Missing required columns in dataset: {', '.join(missing_cols)}. Dashboard functionality may be limited.")
        
    df.dropna(subset=[col for col in required_cols if col in df.columns], inplace=True)

    # Filter daytime zeros (simple filter for dashboard)
    # Ensure index has .hour property (i.e., it's a DatetimeIndex)
    if not isinstance(df.index, pd.DatetimeIndex):
         st.warning("Index is not a DatetimeIndex after loading/conversion. Skipping time-based filtering.")
         df_filtered = df.copy()
    else:
        # We need to drop NaT entries first to prevent errors in .hour access
        # Note: df.index.name might be None if it's an unnamed index
        df_filtered = df.dropna(axis=0, subset=[df.index.name] if df.index.name else None).copy()

        # The GHI filtering logic
        df_filtered = df_filtered[(df_filtered['GHI'] > 0) | (df_filtered.index.hour < 6) | (df_filtered.index.hour > 18)].copy()
    
    # Feature Engineering (Month, DayOfWeek for plotting)
    # This must be done on a DatetimeIndex
    if isinstance(df_filtered.index, pd.DatetimeIndex):
        df_filtered['Month'] = df_filtered.index.to_period('M')
        df_filtered['Month_Name'] = df_filtered.index.strftime('%Y-%m')
    else:
        st.warning("Cannot create time-based features; Index is not a DatetimeIndex.")
        
    return df_filtered


def apply_savgol_filter(series, window_length=51, polyorder=3):
    """Applies Savitzky-Golay filter for smoothing."""
    # Ensure data is numeric and handle NaNs for filtering
    data = series.fillna(series.median()).values
    return savgol_filter(data, window_length, polyorder)

def plot_daily_profile(df, country):
    """Generates a Plotly chart for the average daily profile of GHI."""
    # Check if index is DatetimeIndex before accessing .hour
    if not isinstance(df.index, pd.DatetimeIndex):
        st.error("Cannot plot daily profile: Data index is not recognized as date/time.")
        return go.Figure()

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
    
    # Check if index is DatetimeIndex before accessing .hour and .month
    if not isinstance(df.index, pd.DatetimeIndex):
        st.error("Cannot create forecasting model: Data index is not recognized as date/time.")
        return None, 0, 0, pd.DatetimeIndex([]), np.array([])
        
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
    
    # Check if we have enough data after filtering
    if len(X) < 2:
        st.error("Not enough data points remaining after filtering to train the model.")
        return None, 0, 0, pd.DatetimeIndex([]), np.array([])

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