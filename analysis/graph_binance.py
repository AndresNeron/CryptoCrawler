#!/usr/bin/env python3

import os
import sys
import psycopg2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from binance.client import Client

# Add the project root (../) to sys.path before any custom imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now safe to import personal packages
import globals
from utils.colors import Colors

# Load environment variables from .env file
load_dotenv()

# Database connection configuration from environment variables
api_key = os.getenv('API_KEY')
api_secret = os.getenv('SECRET_KEY')
client = Client(api_key, api_secret)


# Example strategy
def  strategy(df):
        buy = df['SMA'] > df['SMA'].shift(1)
        buy &= df['RSI'] < 30
        sell = df['SMA'] < df['SMA'].shift(1)
        sell |= df['RSI'] > 70
        return buy, sell

# Calculate Technical Indicators
def  calculate_indicators(df):
        # Simple Moving Average (SMA)
        sma = df['close'].rolling(window=50).mean()
        df['SMA'] = sma

        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    #print(f"[RSI] {rsi}")
        df['RSI'] = rsi

# Download Binance historical data
def  download_data(symbol, interval, start_str):
        print(Colors.ORANGE + f'Downloading data for {symbol}. Interval {interval}. Starting from {start_str}' + Colors.R)
        klines = client.get_historical_klines(symbol, interval, start_str)
        data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        data['close'] = data['close'].astype(float)
        return data

# Backtesting strategy
def  backtest(data):
        calculate_indicators(data)
        buy, sell = strategy(data)
        data['buy'] = buy
        data['sell'] = sell
        data['position'] = np.nan
        data.loc[buy, 'position'] = 1
        data.loc[sell, 'position'] = 0
        data['position'].fillna(method='ffill', inplace=True)
        data['position'].fillna(0, inplace=True)
        data['returns'] = np.log(data['close'] / data['close'].shift(1))
        data['strategy'] = data['position'].shift(1) * data['returns']
        data['cumulative_returns'] = data['strategy'].cumsum().apply(np.exp)
        return data



# Plot graph
def plot_results(data, symbol, interval, start_date):

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))  # Single subplot

    # Plot cumulative returns on the primary y-axis (left)
    ax1.plot(data.index, data['cumulative_returns'], label='Cumulative Returns', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Returns', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Set title
    fig.suptitle(f'Cumulative Returns', fontsize=14)

    # Legends for better distinction
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.1), fontsize=10)

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Save plot
    plot_filename = os.path.join(globals.graph_dir, f"backtest_results_{symbol}_{interval}_{start_date}_{globals.timestamp}.png")
    plt.savefig(plot_filename)
    print(Colors.GREEN + f"[!] Plot saved as {plot_filename}" + Colors.R)
    plt.close()


def plot_results_with_rsi(data, symbol, interval, start_date):

    start_date = start_date.replace(' ', '_').lower()

    calculate_indicators(data)

    # Create a figure with two subplots: one for cumulative returns and SMA, and another for RSI
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot cumulative returns and SMA in the first subplot
    ax1.plot(data.index, data['cumulative_returns'], label='Cumulative Returns', color='blue', linewidth=1.5)
    #ax1.plot(data.index, data['SMA'], label='SMA (50)', color='red', linestyle='--')
    ax1.set_ylabel('Cumulative Returns / SMA')
    ax1.set_title(f'Cumulative Returns and SMA for {symbol}')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Plot RSI in the second subplot
    ax2.plot(data.index, data['RSI'], label='RSI (14)', color='blue', linewidth=1.5)
    ax2.axhline(30, color='red', linestyle='--', linewidth=0.8, label='RSI 30 (Oversold)')
    ax2.axhline(70, color='red', linestyle='--', linewidth=0.8, label='RSI 70 (Overbought)')
    ax2.fill_between(data.index, 30, 70, color='yellow', alpha=0.1)  # Highlight neutral RSI zone
    ax2.set_ylabel('RSI')
    ax2.set_title(f'RSI for {symbol}')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    # Set the x-axis label for the bottom subplot
    ax2.set_xlabel('Date')

    # Adjust layout
    fig.tight_layout()

    # Save plot
    plot_filename = os.path.join(globals.graph_dir, f'backtest_with_rsi_{symbol}_{interval}_{start_date}_{globals.timestamp}.png')
    plt.savefig(plot_filename)
    print(Colors.GREEN + f"[!] Plot saved as {plot_filename}" + Colors.R)
    plt.close()


def simple_moving_average(data, alpha=0.1):
    """
    Calculate the simple moving average for the midpoint of price differences between consecutive return values.
    Also, calculate the midpoint of the corresponding timestamps.
    This function returns two lists:
        - smoothed_medians (exponentially smoothed medians)
        - timestamp_midpoints (midpoints of the corresponding timestamps)
    """
    medians = []
    timestamp_midpoints = []
    smoothed_medians = []  # List to store the exponentially smoothed medians

    # Loop through the data, calculate midpoints between consecutive points
    for i in range(1, len(data)):
        # Calculate the midpoint between the i-th and (i-1)-th return values
        median = (data[i] + data[i - 1]) / 2
        medians.append(median)

        # Apply exponential smoothing on the medians
        if i == 1:  # First point, no smoothing to apply
            smoothed_median = median
        else:
            smoothed_median = alpha * median + (1 - alpha) * smoothed_medians[-1]

        smoothed_medians.append(smoothed_median)

        # Convert timestamps to numeric values (Unix timestamp), calculate the midpoint, then convert back to Timestamp
        timestamp1 = data.index[i].timestamp()  # Convert to Unix timestamp (float)
        timestamp2 = data.index[i - 1].timestamp()  # Convert to Unix timestamp (float)
        timestamp_midpoint = (timestamp1 + timestamp2) / 2  # Average the timestamps
        timestamp_midpoint = pd.to_datetime(timestamp_midpoint, unit='s')  # Convert back to Timestamp
        timestamp_midpoints.append(timestamp_midpoint)

    return medians, smoothed_medians, timestamp_midpoints


def plot_simple_moving_average(data, symbol, alpha=0.1):
    """
    Plot both cumulative returns, medians, and the exponentially smoothed moving average medians on the same graph.
    """
    if 'cumulative_returns' not in data.columns:
        print("Calculating cumulative returns.")
        data['cumulative_returns'] = (data['close'] / data['close'].iloc[0]) - 1
    else:
        print("Cumulative returns already calculated.")

    # Calculate medians, smoothed medians, and timestamp midpoints
    medians, smoothed_medians, timestamp_midpoints = simple_moving_average(data['cumulative_returns'], alpha)

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))  # Single subplot for both plots

    # Plot cumulative returns on the primary y-axis (left) with a thicker line
    ax1.plot(data.index, data['cumulative_returns'], label='Cumulative Returns', color='blue', linewidth=3)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Returns', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Plot the raw medians (before smoothing) in green with a thinner line
    ax1.plot(timestamp_midpoints, medians, label='Medians (Before Smoothing)', color='black', linewidth=2)

    # Plot the exponentially smoothed medians (simple moving average midpoints) in red with a thinner line
    ax1.plot(timestamp_midpoints, smoothed_medians, label=f'Smoothed Medians (SMA Midpoints) α={alpha}', color='red', linewidth=1)

    # Set title and legends
    ax1.set_title(f'Cumulative Returns and Smoothed Medians (SMA Midpoints) for {symbol}')
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.1), fontsize=10)

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Save plot
    plot_filename = os.path.join(globals.graph_dir, f'backtest_results_{symbol}.png')
    plt.savefig(plot_filename)
    print(Colors.GREEN + f"[!] Plot saved as {plot_filename}" + Colors.R)
    plt.close()


def plot_multiple_smooth_averages(data, symbol, alpha_values=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Plot cumulative returns and multiple exponentially smoothed moving averages with different alphas on the same graph.
    """
    if 'cumulative_returns' not in data.columns:
        print("Calculating cumulative returns.")
        data['cumulative_returns'] = (data['close'] / data['close'].iloc[0]) - 1
    else:
        print("Cumulative returns already calculated.")

    # Create figure for the plot
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))  # Single subplot for both plots

    # Plot cumulative returns on the primary y-axis (left) with a thicker line
    ax1.plot(data.index, data['cumulative_returns'], label='Cumulative Returns', color='blue', linewidth=3)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Returns', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Loop over each alpha value to calculate and plot the smoothed medians
    for alpha in alpha_values:
        # Calculate medians, smoothed medians, and timestamp midpoints for the current alpha
        medians, smoothed_medians, timestamp_midpoints = simple_moving_average(data['cumulative_returns'], alpha)

        # Plot the medians (before smoothing) for this alpha
        ax1.plot(timestamp_midpoints, medians, label=f'Medians (Before Smoothing) α={alpha}', linestyle='--', linewidth=1)

        # Plot the exponentially smoothed medians (simple moving average midpoints) for this alpha
        ax1.plot(timestamp_midpoints, smoothed_medians, label=f'Smoothed Medians (SMA Midpoints) α={alpha}', linewidth=2)

    # Set title and legends
    ax1.set_title(f'Cumulative Returns and Smoothed Medians (SMA Midpoints) for {symbol}')
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.1), fontsize=10)

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Save plot
    plot_filename = os.path.join(globals.graph_dir, f'backtest_results_{symbol}.png')
    plt.savefig(plot_filename)
    print(Colors.GREEN + f"[!] Plot saved as {plot_filename}" + Colors.R)
    plt.close()


def find_max_min_points(data, column='close'):
    """
    Identify the maximum and minimum points in the specified column of the dataset.

    Parameters:
        data (pd.DataFrame): The dataset containing the time series.
        column (str): The column in the dataset to analyze for maxima and minima.

    Returns:
        max_points (pd.DataFrame): DataFrame containing local maxima.
        min_points (pd.DataFrame): DataFrame containing local minima.
    """
    # Shift data to find local maxima and minima
    previous = data[column].shift(1)
    next_ = data[column].shift(-1)

    # Logical conditions for maxima and minima
    maxima = (data[column] > previous) & (data[column] > next_)
    minima = (data[column] < previous) & (data[column] < next_)

    # Extract maxima and minima points
    max_points = data[maxima]
    min_points = data[minima]

    return max_points, min_points


def analyze_extreme_points(data, symbol, interval, start_date):

    output_file = f"{globals.parent_path}/graphs/{globals.timestamp}/backtesting/{symbol}_{interval}_{start_date.replace(' ', '_')}_extreme_points.png"
    print(Colors.BOLD_WHITE + f"[+] Analyzing data for {interval} interval starting {start_date} and ending {globals.timestamp}" + Colors.R)

    # Find maximum and minimum points
    max_points, min_points = find_max_min_points(data, column='close')

    print("Maximum Points:")
    print(max_points)

    print("Minimum Points:")
    print(min_points)

    # Plot the results
    plt.figure(figsize=(12, 7))
    plt.plot(data.index, data['close'], label='Close Price', color='blue')
    plt.scatter(max_points.index, max_points['close'], color='green', label='Maxima', s=10)
    plt.scatter(min_points.index, min_points['close'], color='red', label='Minima', s=10)

    # Add lines connecting the maxima and minima
    plt.plot(max_points.index, max_points['close'], color='green', linestyle='--', label='Maxima Line')
    plt.plot(min_points.index, min_points['close'], color='red', linestyle='--', label='Minima Line')

    # Add title, legend, and grid
    plt.title(f"Maximum and Minimum Points for {symbol} ({interval} - {globals.timestamp})")
    plt.legend()
    plt.grid()

    # Save the plot to a file
    plt.savefig(output_file, format="png", dpi=300)
    plt.close()  # Close the figure to free up memory

    print(Colors.GREEN + f"[!] Plot saved to {output_file}" + Colors.R)


def exponential_smoothing(series, alpha=0.9):
    smoothed_series = series.copy()
    for t in range(1, len(series)):
        smoothed_series[t] = alpha * series[t] + (1 - alpha) * smoothed_series[t-1]
    return smoothed_series


def plot_analyze_extreme_points_and_rsi(data, symbol, interval, start_date, alpha=0.9):
    """
    Plot analyze_extreme_points graph and RSI as subplots in the same figure.
    """
    start_date = start_date.replace(' ', '_').lower()

    # Ensure necessary indicators are calculated
    calculate_indicators(data)

    # Identify extreme points in RSI (example: overbought and oversold)
    overbought = data[data['RSI'] > 70]
    oversold = data[data['RSI'] < 30]

    # Exponentially smooth the overbought and oversold RSI points
    smoothed_overbought = exponential_smoothing(overbought['RSI'], alpha)
    smoothed_oversold = exponential_smoothing(oversold['RSI'], alpha)

    # Reindex the smoothed values to match the original data
    smoothed_overbought = smoothed_overbought.reindex(data.index, method='ffill')
    smoothed_oversold = smoothed_oversold.reindex(data.index, method='ffill')

    # Identify maxima and minima for the price (close)
    max_points, min_points = find_max_min_points(data, column='close')

    # Create a figure with two subplots: one for analyze_extreme_points and another for RSI
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot analyze_extreme_points data in the first subplot
    ax1.plot(data.index, data['close'], label='Price', color='blue', linewidth=1.5)
    ax1.plot(data.index, data['SMA'], label='SMA (50)', color='red', linestyle='--')
    ax1.set_title(f'Price and SMA for {symbol}')
    ax1.set_ylabel('Price / SMA')
    ax1.legend()

    # We can change this alpha for testing purposes
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Highlight extreme points (local minima/maxima for price)
    ax1.scatter(max_points.index, max_points['close'], color='green', label='Maxima', marker='o', s=25)
    ax1.scatter(min_points.index, min_points['close'], color='red', label='Minima', marker='o', s=25)

    # Connect maxima points with a green line
    ax1.plot(max_points.index, max_points['close'], color='green', linestyle='-', linewidth=1)

    # Connect minima points with a red line
    ax1.plot(min_points.index, min_points['close'], color='red', linestyle='-', linewidth=1)

    # Plot RSI in the second subplot
    ax2.plot(data.index, data['RSI'], label='RSI (14)', color='blue', linewidth=1.5)
    ax2.axhline(30, color='red', linestyle='--', linewidth=0.8, label='RSI 30 (Oversold)')
    ax2.axhline(70, color='red', linestyle='--', linewidth=0.8, label='RSI 70 (Overbought)')
    ax2.fill_between(data.index, 30, 70, color='yellow', alpha=0.1)  # Highlight neutral RSI zone
    ax2.set_ylabel('RSI')
    ax2.set_title('Relative Strength Index (RSI)', loc='left', pad=40)

    # Plot extreme RSI points (overbought and oversold)
    ax2.scatter(overbought.index, overbought['RSI'], color='green', label='Overbought', marker='o')
    ax2.scatter(oversold.index, oversold['RSI'], color='red', label='Oversold', marker='o')

    # Connect overbought points with a red line (smoothed)
    #ax2.plot(data.index, smoothed_overbought, color='red', linestyle='-', linewidth=1)

    # Connect oversold points with a blue line (smoothed)
    #ax2.plot(data.index, smoothed_oversold, color='blue', linestyle='-', linewidth=1)

    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    # Set the x-axis label for the bottom subplot
    ax2.set_xlabel('Date')

    # Adjust layout for better spacing
    fig.tight_layout()

    # Save the plot
    plot_filename = os.path.join(globals.graph_dir, f'analyze_extreme_points_and_rsi_{symbol}_{interval}_{globals.timestamp}.png')
    plt.savefig(plot_filename)
    print(Colors.GREEN + f"[!] Plot saved as {plot_filename}" + Colors.R)
    plt.close()



# Run Test Trading mode
def test_trading(data, symbol, interval, start_date):
    #data = download_data(symbol, interval, start_date)
    #plot_simple_moving_average(data, symbol)
    #plot_multiple_smooth_averages(data, symbol, alpha_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    #plot_multiple_smooth_averages(data, symbol, alpha_values=[0.5, 0.9])
    #plot_multiple_smooth_averages(data, symbol, alpha_values=[0.9])

    data = backtest(data)
    plot_results(data, symbol, interval, start_date)
    return data

# Live Trading mode
def live_trading(data, symbol, interval, start_date):
    client = Client(api_key, api_secret)
    prev_buy_signal = False
    prev_sell_signal = False
    while True:
        calculate_indicators(data)
        buy_signal, sell_signal = strategy(data)
        buy_signal = buy_signal[-1]
        sell_signal = sell_signal[-1]
        if buy_signal and not prev_buy_signal:
            order = client.create_test_order(
                symbol=symbol,
                side='BUY',
                type='MARKET',
                quantity=0.001,
            )
            print('Buy signal generated. Placing market buy order.')

        elif sell_signal and not prev_sell_signal:
            order = client.create_test_order(
                symbol=symbol,
                side='SELL',
                type='MARKET',
                quantity=0.001,
            )
            print('Sell signal generated. Placing market sell order.')
        prev_buy_signal = buy_signal
        prev_sell_signal = sell_signal
        time.sleep(60)


def fetch_current_assets():
    try:
        account_info = client.get_account()
        balances = account_info['balances']

        # Convert to a readable DataFrame
        assets = []
        for balance in balances:
            asset = balance['asset']
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                assets.append({'Asset': asset, 'Free': free, 'Locked': locked})

        assets_df = pd.DataFrame(assets)
        return assets_df
    except Exception as e:
        print(f"Error fetching current assets: {e}")
        return None

# Fetch a list of supported trading pairs from the API
def get_supported_symbols():
    exchange_info = client.get_exchange_info()
    return {symbol['symbol'] for symbol in exchange_info['symbols']}

def insert_symbols(symbols):
    # Database connection configuration from environment variables
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')

    # Insert symbols into the PostgreSQL database
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO binance_symbols (symbol)
            VALUES (%s)
        """

        cont=1
        for symbol in symbols:
            print(Colors.GREEN + f"[{cont}] Inserting {symbol}" + Colors.R)
            cursor.execute(insert_query, (symbol,))
            cont+=1

        conn.commit()
        print(Colors.GREEN + "\n[!] Symbols inserted into database successfully." + Colors.R)

    except psycopg2.Error as e:
        print(f"Database error: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":

    # Insert supported symbols into database
    #supported_symbols = get_supported_symbols()
    #insert_symbols(supported_symbols)
    #sys.exit(0)

    # Example: Fetch and display current assets
    assets_df = fetch_current_assets()

    if assets_df is not None:
            print(Colors.ORANGE + "[!] Looping through assets:\n" + Colors.R)
            for idx, row in assets_df.iterrows():
                asset = row['Asset']
                free = row['Free']
                locked = row['Locked']
                print(Colors.BOLD_WHITE + f"\n[{idx+1}] Asset: {asset}, Free: {free}, Locked: {locked}" + Colors.R)

                if asset == "ETHW":
                    print(Colors.ORANGE + "[!] Skipping ETHW!" + Colors.R)
                    continue

                if free > 0:
                    symbol = f"{asset}USDT"
                    try:
                        print(Colors.GREEN + f"Running trading test for {symbol}..." + Colors.R)

                        for interval, start_date in globals.date_combinations:
                            print(Colors.BOLD_WHITE + f"[+] Analyzing data for {interval} interval starting {globals.timestamp}" + Colors.R)

                            # Download the data
                            data = download_data(symbol, interval, start_date)

                            # Fetching and preparing data
                            data['daily_returns'] = data['close'].pct_change()  # Calculate daily returns
                            data['cumulative_returns'] = (1 + data['daily_returns']).cumprod()  # Calculate cumulative returns

                            # Graph RSI
                            plot_results_with_rsi(data, symbol, interval, start_date)
                            plot_analyze_extreme_points_and_rsi(data, symbol, interval, start_date)

                            test_trading(data, symbol, interval, start_date)
                            analyze_extreme_points(data, symbol, interval, start_date)


                    except Exception as e:
                        print(Colors.RED + f"Could not run test trading for {symbol}: {e}" + Colors.R)
