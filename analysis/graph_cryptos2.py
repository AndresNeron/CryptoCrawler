#!/usr/bin/env python3

# This script works for interpreting and graphing data stored on cryptocurrency database
# Some data analysis pipelines are applied here.

import re
import os
import csv
import sys
import time
import shutil
import random
import argparse
import psycopg2
import warnings
import ipaddress
import matplotlib
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from binance.client import Client
from datetime import datetime, timedelta
from matplotlib.ticker import MaxNLocator
from psycopg2.extras import execute_values
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the project root (../) to sys.path before any custom imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Personal packages
import globals
from utils.colors import Colors

# Suppress pandas FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Optionally, for pandas-specific behavior in future versions:
pd.set_option('mode.chained_assignment', None)

# Load environment variables from .env file
load_dotenv()

# Database connection configuration from environment variables
api_key = os.getenv('API_KEY')
api_secret = os.getenv('SECRET_KEY')
client = Client(api_key, api_secret)

cont_global = 1

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="This is script works for performing analysis and graphing data from cryptocurrency database.")
    parser.add_argument("-a", "--analysis", help="\t\tAnalysis number")
    parser.add_argument("-d", "--db_host", help="\t\tIPv4 address of database")
    parser.add_argument("-v", "--volatility", action='store_true', help="\t\tCalculate the volatility of each crypto")
    parser.add_argument("-b", "--backfill", help="\t\tBack fill missing parameters")
    parser.add_argument("-f", "--full", action='store_true', help="\t\tPerform all the graphs.")
    parser.add_argument("-t", "--time", help="\t\tReceive period of time, per example, '1_month', '1_year', etc")
    parser.add_argument("-e", "--examples", action='store_true', help="\t\tShow execution examples")
    return parser.parse_args()


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
    timestamp = datetime.now().replace(microsecond=0)
    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    print(Colors.ORANGE + f"[!] Downloading data for {symbol}. Interval {interval}. Starting from {start_str} at {timestamp}" + Colors.R)
    klines = client.get_historical_klines(symbol, interval, start_str)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    data['close'] = data['close'].astype(float)
    return data

# Function to update historical data incrementally
def update_data(data, symbol, interval, verbose=False):

    if verbose:
        print(f"Type of data: {type(data)}")

        # Print the current structure of the DataFrame
        if isinstance(data, pd.DataFrame):
            print(f"Columns in the DataFrame: {list(data.columns)}")
            print(f"First few rows of data:\n{data.head()}")

    # Ensure `data` is a DataFrame
    if data is None or not isinstance(data, pd.DataFrame):
        print(Colors.RED + f"[!] Warning: Data is invalid or None. Initializing empty DataFrame for {symbol}." + Colors.R)
        data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Check if `timestamp` is the index
    if data.index.name != 'timestamp':
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
        elif data.empty:
            print(Colors.YELLOW + "[!] No data available to update." + Colors.R)
            return data
        else:
            raise ValueError("[!] Data is missing a 'timestamp' column for indexing.")

    # Get the last timestamp from the existing data
    if not data.empty:
        end_time = data.index[-1] + pd.Timedelta(seconds=1)  # Increment by 1 second to avoid overlap
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")  # Convert to string for Binance API
    else:
        end_time_str = "1 day ago"  # Default to fetching from a day ago if data is empty
        print(Colors.ORANGE + f"[!] No existing data. Fetching from default start time: {end_time_str}" + Colors.R)
    
    print(f"[!] Updating data for {symbol}. Interval {interval}. Starting from {end_time_str}")
    
    # Download new data starting from the calculated timestamp
    new_data = download_data(symbol, interval, end_time_str)

    # Combine existing and new data, dropping duplicates
    updated_data = pd.concat([data, new_data]).drop_duplicates().sort_index()
    
    return updated_data


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



def get_graphs_directory(time_span):
    """
    Generate a directory path for saving graphs based on the execution date and time span.
    """
    time_span = time_span.replace(' ', '_')
    execution_date = datetime.now().strftime("%Y-%m-%d")
    graphs_dir = f"graphs/{globals.timestamp}/{time_span}"
    os.makedirs(graphs_dir, exist_ok=True)
    return graphs_dir


def calculate_volatility(cryptocurrency, timestamps, prices):
    """
    Calculate a single volatility value for a cryptocurrency.

    Args:
        cryptocurrency (str): The name of the cryptocurrency.
        timestamps (list): List of timestamps.
        prices (list): List of prices.

    Returns:
        tuple: The latest timestamp and the volatility value.
    """

    global cont_global
    print(Colors.BOLD_WHITE + f"[{cont_global}] Calculating volatility of {cryptocurrency}" + Colors.R)
    cont_global += 1

    # Ensure there is enough data for calculation
    if len(prices) < 2:
        return None, 0

    # Calculate percentage returns
    returns = np.diff(prices) / prices[:-1]  # Percentage changes

    # Calculate the standard deviation (volatility)
    volatility = np.std(returns)

    # Use the latest timestamp for the result
    latest_timestamp = timestamps[-1]

    return latest_timestamp, volatility


def get_limit_points(time_span):
    # Map time spans to number of points to plot
    points_map = {
        "1_day": 50,
        "1_week": 70,
        "2_weeks": 100,
        "1_month": 150,
        "2_months": 200,
        "3_months": 300,
        "6_months": 400,
        "1_year": 500,
        "2_years": 700,
        "5_years": 1000
    }
    # Default fallback if time_span not found
    return points_map.get(time_span, 150)

def analysis1(cursor, conn, time_span="1_month", volatility=False):
    """
    Generate a graph of price trends for each tracked cryptocurrency in the database,
    and calculate volatility of each one.
    """

    # Create a directory name based on the current date and time span
    graphs_dir = get_graphs_directory(time_span)

    # Then inside analysis1 before cursor.execute:
    limit_points = get_limit_points(time_span)

    cursor.execute(
        """
        SELECT cryptocurrency
        FROM crawled_cryptos;
        """
    )

    assets = cursor.fetchall()

    crypto_cont = 1
    for row in assets:
        asset = row[0]

#        cursor.execute(
#            """
#            SELECT cryptocurrency, price, timestamp
#            FROM crypto_data
#            WHERE cryptocurrency = %s
#              AND timestamp >= CURRENT_DATE - INTERVAL %s
#            ORDER BY timestamp DESC;
#            """, (asset,time_span)
#        )

        cursor.execute(
            """
            WITH ranked_data AS (
                SELECT *,
                       row_number() OVER (ORDER BY timestamp) AS rn,
                       count(*) OVER () AS total
                FROM crypto_data
                WHERE cryptocurrency = %s
                  AND timestamp >= CURRENT_DATE - INTERVAL %s
            )
            SELECT cryptocurrency, price, timestamp
            FROM ranked_data
            WHERE rn %% CEIL(total::float / %s)::int = 0
            ORDER BY timestamp DESC;
            """, (asset, time_span, limit_points)
        )


        results = cursor.fetchall()

        # Skip cryptocurrencies with no data in the specified time span
        if not results:
            print(f"[{crypto_cont}] Skipping {asset}: No data found for the specified time span.")
            crypto_cont += 1
            continue

        # Data preparation for the graphing
        timestamps = [result[2] for result in results]
        prices = [float(result[1]) for result in results]

        if volatility:
            # Calculate a single volatility value
            latest_timestamp, single_volatility = calculate_volatility(asset, timestamps, prices)

            # Save the volatility to the database
            cursor.execute(
                """
                INSERT INTO crypto_volatility (cryptocurrency, timestamp, volatility)
                VALUES (%s, %s, %s)
                """,
                (asset, latest_timestamp, float(single_volatility)),
            )
            conn.commit()

        # Plot the data
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, prices, marker='o', markersize=1, label=asset)
        plt.title(f"{asset} Price Trends")
        plt.xlabel("Timestamp")
        plt.ylabel("Price (USD)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot in the execution directory
        graph_name = f"{asset}_price_trend.png"
        graph_path = os.path.join(graphs_dir, graph_name)
        # Check if the file already exists and delete it
        if os.path.exists(graph_path):
            os.remove(graph_path)

        plt.savefig(graph_path)
        print(Colors.ORANGE + f"[{crypto_cont}]\tGraph saved as " + Colors.BOLD_WHITE +f"{graph_path}" + Colors.R)
        crypto_cont += 1
        plt.close()


def analysis2(cursor, conn, time_span="2_week"):
    """
    Plot portfolio trends of my personal assets on a single graph.
    """
    graphs_dir = get_graphs_directory(time_span)
    if os.path.exists(graphs_dir):
        shutil.rmtree(graphs_dir)
    os.makedirs(graphs_dir, exist_ok=True)

    # Example: Fetch and display current assets
    assets_df = fetch_current_assets()

    if assets_df is not None:
        print(Colors.ORANGE + "[!] Your personal assets are:" + Colors.R)

        # Initialize the plot outside the loop
        plt.figure(figsize=(10, 5))
        crypto_cont = 1
        data_added = False

        for idx, row in assets_df.iterrows():
            asset = row['Asset']
            free = row['Free']
            locked = row['Locked']
            print(Colors.BOLD_WHITE + f"\n[{idx+1}] Asset: {asset}, Free: {free}, Locked: {locked}" + Colors.R)

            cursor.execute("""
                SELECT timestamp, total_value_usd 
                FROM assets_per_crypto
                WHERE cryptocurrency = %s
                  AND timestamp >= CURRENT_DATE - INTERVAL %s
                ORDER BY timestamp DESC;
            """, (asset, time_span))
            results = cursor.fetchall()

            # Skip cryptocurrencies with no data in the specified time span
            if not results:
                print(f"[!] Skipping {asset}: No data found for the specified time span.\n\n")
                crypto_cont += 1
                continue

            # Prepare data for plotting
            timestamps = [result[0] for result in results]
            prices = [float(result[1]) for result in results]

            # Add the data to the plot
            plt.plot(timestamps, prices, marker='o', markersize=1, label=asset)
            crypto_cont += 1
            data_added = True

        if data_added:
            plt.legend()
            # Finalize the graph
            plt.title("Portfolio Trends")
            plt.xlabel("Timestamp")
            plt.ylabel("Price (USD)")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()

            # Save or show the combined plot
            graph_name = "combined_portfolio_trend.png"
            graph_path = os.path.join(graphs_dir, graph_name)
            #if os.path.exists(graph_path):
            #    os.remove(graph_path)
            #    print("[*] Removed previous graph")

            plt.savefig(graph_path)
            print(Colors.ORANGE + f"Graph saved as {graph_path}" + Colors.R)
        else:
            print(Colors.RED + "[!] No data available to generate a graph." + Colors.R)

        plt.close()


def parse_relative_time(relative_time):
    """
    Parse relative time strings like '1 day', '2 months', or '1 year' into a datetime object.
    """
    match = re.match(r"(\d+)\s*(day|days|month|months|year|years|hour|hours|minute|minutes|second|seconds)", relative_time)
    if not match:
        raise ValueError(f"Invalid relative time format: {relative_time}")

    amount, unit = match.groups()
    amount = int(amount)
    if "day" in unit:
        return datetime.now() - timedelta(days=amount)
    elif "month" in unit:
        # Approximate a month as 30 days
        return datetime.now() - timedelta(days=amount * 30)
    elif "year" in unit:
        # Approximate a year as 365 days
        return datetime.now() - timedelta(days=amount * 365)
    elif "hour" in unit:
        return datetime.now() - timedelta(hours=amount)
    elif "minute" in unit:
        return datetime.now() - timedelta(minutes=amount)
    elif "second" in unit:
        return datetime.now() - timedelta(seconds=amount)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")


# Graph gains from each personal asset added in a single plot
def analysis3(cursor, conn, start_date="1 month", interval="1 day",):
    """
    Generate a graph tracking my total portfolio assets.
    """
    graphs_dir = get_graphs_directory(start_date)
    graphs_dir = graphs_dir.replace(' ', '_')

    # Handle start_date input
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    elif isinstance(start_date, str):
        try:
            # Try parsing as a relative time first
            start_date = parse_relative_time(start_date)
        except ValueError:
            # Fallback to parsing as a timestamp
            try:
                start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                raise ValueError(f"Invalid date format for start_date: {start_date}. "
                                 "Expected format: '1 day' or '%Y-%m-%d %H:%M:%S'")

    start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Execute the query
    cursor.execute("""
        SELECT 
            timestamp, 
            total_assets_usd
        FROM 
            total_assets
        WHERE 
            total_assets_usd > 125
            AND total_assets_usd < 190
            AND timestamp >= %s
            AND timestamp < %s
        ORDER BY 
            timestamp
    """, (start_date_str, end_date))

    results = cursor.fetchall()
    if not results:
        print(f"No data found for the specified criteria: start_date={start_date_str}, end_date={end_date}")
        return None

    timestamps = [row[0] for row in results]
    total_assets_usd = [float(row[1]) for row in results]

    # Plot USD trends
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, total_assets_usd, marker='o', markersize=2, label="Total Assets (USD)", color='blue')
    plt.title(f"Total Asset Value Trends\t{end_date}")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    graph_name = f"total_asset_trends_{end_date.replace(' ', '_')}.png"
    graph_path = os.path.join(graphs_dir, graph_name)
    if os.path.exists(graph_path):
        os.remove(graph_path)
    plt.savefig(graph_path)
    print(Colors.ORANGE + f"Graph saved as\t{Colors.BOLD_WHITE}{graph_path}" + Colors.R)
    plt.close()


def analysis4(cursor, conn, time_span="1_month"):
    """
    Analyze the volatility of crypto currencies
    """
    graphs_dir = get_graphs_directory(time_span)


    cursor.execute(
    """
    SELECT cryptocurrency
    FROM crawled_cryptos;
    """)

    assets = cursor.fetchall()

    crypto_cont = 1
    for row in assets:
        asset = row[0]

        cursor.execute(
        """
        SELECT timestamp, volatility
        FROM crypto_volatility
        WHERE cryptocurrency = %s
          AND timestamp >= CURRENT_DATE - INTERVAL %s
        ORDER BY timestamp DESC;
        """, (asset,time_span))

        results = cursor.fetchall()

        # Skip cryptocurrencies with no data in the specified time span
        if not results:
            print(f"[{crypto_cont}] Skipping {asset}: No data found for the specified time span.")
            crypto_cont += 1
            continue

        # Prepare the list
        timestamps = [row[0] for row in results]
        volatility = [float(row[1]) for row in results]

        # Create a DataFrame for the original data
        data = pd.DataFrame({'timestamp': pd.to_datetime(timestamps), 'volatility': volatility})

        # Remove duplicate timestamps by averaging their volatility
        data = data.groupby('timestamp', as_index=False).mean()

        # Plot the data dynamically based on available timestamps
        plt.figure(figsize=(10, 5))
        plt.plot(data['timestamp'], data['volatility'], marker='o', markersize=2, label=f"Volatility of {asset}", color='blue')
        plt.title(f"Volatility of {asset}")
        plt.xlabel("Timestamp")
        plt.ylabel("Volatility")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        graph_name = f"{asset}_volatility.png"
        graph_path = os.path.join(graphs_dir, graph_name)
        # Check if the file already exists and delete it
        if os.path.exists(graph_path):
            os.remove(graph_path)
        plt.savefig(graph_path)
        print(Colors.ORANGE + f"[{crypto_cont}] Graph saved as {graph_path}" + Colors.R)
        plt.close()

        crypto_cont += 1


def analysis5(cursor, conn, time_span="1_month"):
    """
    Analyze and plot the volatility of all cryptocurrencies on the same image.
    """

    graphs_dir = get_graphs_directory(time_span)


    # Fetch the cryptocurrencies sorted by their latest volatility using the subquery
    cursor.execute(
        """
        SELECT subquery.cryptocurrency, subquery.timestamp, subquery.volatility
        FROM (
            SELECT DISTINCT ON (cryptocurrency) cryptocurrency, timestamp, volatility
            FROM crypto_volatility
            ORDER BY cryptocurrency, timestamp DESC
        ) subquery
        JOIN crawled_cryptos ON subquery.cryptocurrency = crawled_cryptos.cryptocurrency
        ORDER BY subquery.volatility;
        """
    )

    # Fetch the sorted assets
    sorted_assets = cursor.fetchall()

    # Initialize the plot
    plt.figure(figsize=(15, 7))  # Larger figure for multiple assets

    # Process each cryptocurrency and plot its volatility
    print("Cryptocurrencies (sorted by volatility):")
    crypto_cont = 1
    for row in sorted_assets:
        asset = row[0]
        latest_volatility = row[2]
        print(f"[{crypto_cont}] {asset} (Latest Volatility: {latest_volatility})")
        crypto_cont += 1

        # Fetch all timestamps and volatility data for the current asset
        cursor.execute(
            """
            SELECT timestamp, volatility
            FROM crypto_volatility
            WHERE cryptocurrency = %s
              AND timestamp >= CURRENT_DATE - INTERVAL %s
            ORDER BY timestamp DESC;
            """,
            (asset,time_span),
        )

        results = cursor.fetchall()

        # Skip cryptocurrencies with no data in the specified time span
        if not results:
            print(f"[{crypto_cont}] Skipping {asset}: No data found for the specified time span.")
            crypto_cont += 1
            continue

        # Prepare the lists
        timestamps = [row[0] for row in results]
        volatility = [float(row[1]) for row in results]

        # Create a DataFrame for the original data
        data = pd.DataFrame({'timestamp': pd.to_datetime(timestamps), 'volatility': volatility})

        # Remove duplicate timestamps by averaging their volatility
        data = data.groupby('timestamp', as_index=False).mean()

        # Plot each cryptocurrency's volatility on the same graph
        plt.plot(data['timestamp'], data['volatility'], marker='o', markersize=2, label=f"{asset}")

    # Add graph details
    plt.title("Volatility of Cryptocurrencies (Sorted by Volatility)")
    plt.xlabel("Timestamp")
    plt.ylabel("Volatility")
    plt.xticks(rotation=45)

    # Generate the legend sorted by volatility
    plt.legend(title="Assets (Sorted by Volatility)", loc="upper left", fontsize="small")
    plt.grid(True)
    plt.tight_layout()

    # Save the combined plot
    graph_name = "combined_volatility_sorted.png"
    graph_path = os.path.join(graphs_dir, graph_name)
    # Check if the file already exists and delete it
    if os.path.exists(graph_path):
        os.remove(graph_path)
    plt.savefig(graph_path)
    print(
        Colors.ORANGE
        + f"Combined graph saved as "
        + Colors.BOLD_WHITE
        + f"graphs/{graph_name}"
        + Colors.R
    )
    plt.close()


# Rank cryptos by cumulative returns of backtest
def analysis6(top_rank, conn, cursor, interval, start_date, test_mode=True):
#    cursor.execute(
#        """
#        SELECT 
#            cs.symbol
#        FROM 
#            crawled_cryptos cc
#        INNER JOIN 
#            crypto_slugs cs
#        ON 
#            cc.cryptocurrency = cs.slug
#        WHERE 
#            cs.symbol IN (
#                SELECT 
#                    LEFT(symbol, LENGTH(symbol) - 4) -- Removes the "USDT" suffix
#                FROM 
#                    binance_symbols 
#                WHERE 
#                    symbol LIKE '%USDT'
#            );
#        """
#    )

    # This through all the available symbols on binance
#    cursor.execute(
#        """
#        SELECT symbol
#        FROM binance_symbols
#        WHERE symbol LIKE '%USDT';
#        """
#    )
#
#    crypto_list = cursor.fetchall()

    # Fetch valid symbols
    exchange_info = client.get_exchange_info()
    crypto_list = {symbol['symbol'] for symbol in exchange_info['symbols'] if symbol['symbol'].endswith('USDT')}

    crypto_list = list(crypto_list)
    random.shuffle(crypto_list)


    results = []
    cont = 1

    # Generate a single timestamp for this execution
    execution_timestamp = datetime.now()

    for symbol in crypto_list:
        #symbol = symbol[0] + "USDT"
        #symbol = symbol[0]
        try:
            print(f"[{cont}][{symbol}] Analysis ...")
            cont += 1
            data = download_data(symbol, interval, start_date)
            if data.empty:
                print(f"No data available for {symbol}. Skipping...")
                continue

            #if cont > 50:
            #    break
            
            # Ensure 'position' column exists
            if 'position' not in data.columns:
                data['position'] = None  # Initialize with None or any default value
            
            # Forward fill and handle missing values
            data['position'] = data['position'].ffill()
            data['position'] = data['position'].fillna(0)
            
            backtested_data = backtest(data)
            cumulative_return = backtested_data['cumulative_returns'].iloc[-1]
            
            results.append({
                'symbol': symbol,
                'cumulative_return': cumulative_return,
            })
            print(Colors.BOLD_WHITE + f"Completed analysis for {symbol}. Cumulative return:\t" + Colors.CYAN +  f"{cumulative_return}\n" + Colors.R)

        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
    
    # Rank by cumulative returns
    sorted_results = sorted(results, key=lambda x: x['cumulative_return'], reverse=True)

    try:
        # Insert each record individually
        for rank, result in enumerate(sorted_results, start=1):
            try:
                cursor.execute(
                    """
                    INSERT INTO ranked_cryptos (symbol, cumulative_return, rank, execution_timestamp)
                    VALUES (%s, %s, %s, %s);
                    """,
                    (result['symbol'], float(result['cumulative_return']), rank, execution_timestamp)
                )
                conn.commit()  # Commit after each insertion
            except Exception as insert_error:
                print(f"[x] Failed to insert symbol {result['symbol']}: {insert_error}")
    except Exception as e:
        print(f"[x] Critical error occurred during insertion: {e}")
    finally:
        cursor.close()
        conn.close()

    return sorted_results


def analysis7(cursor, conn):
    """
    Analyze columns of crypto_data for selecting which are the best crypto to buy.
    """
    # Fetch volume data
    cursor.execute(
        """
        SELECT 
            cryptocurrency, 
            AVG(
                CASE 
                    WHEN volume ~ 'K$' THEN (replace(volume, 'K', '')::numeric * 1e3)
                    WHEN volume ~ 'M$' THEN (replace(volume, 'M', '')::numeric * 1e6)
                    WHEN volume ~ 'B$' THEN (replace(volume, 'B', '')::numeric * 1e9)
                    WHEN volume ~ 'T$' THEN (replace(volume, 'T', '')::numeric * 1e12)
                    ELSE volume::numeric
                END
            ) AS avg_volume
        FROM crypto_data
        WHERE timestamp > NOW() - INTERVAL '7 days'
        GROUP BY cryptocurrency
        ORDER BY avg_volume DESC;
        """
    )
    volume_data = cursor.fetchall()

    # Fetch volatility data
    cursor.execute(
        """
        SELECT cryptocurrency, AVG(volatility) AS avg_volatility
        FROM crypto_volatility
        WHERE timestamp > NOW() - INTERVAL '1 week'
        GROUP BY cryptocurrency
        ORDER BY avg_volatility DESC;
        """
    )
    volatility_data = cursor.fetchall()

    # Fetch market_cap data
    cursor.execute(
        """
        SELECT cryptocurrency, price, market_cap
        FROM (
            SELECT 
                cryptocurrency, 
                price, 
                market_cap,
                CASE
                    WHEN market_cap ~ 'K$' THEN (replace(market_cap, 'K', '')::numeric * 1e3)
                    WHEN market_cap ~ 'M$' THEN (replace(market_cap, 'M', '')::numeric * 1e6)
                    WHEN market_cap ~ 'B$' THEN (replace(market_cap, 'B', '')::numeric * 1e9)
                    WHEN market_cap ~ 'T$' THEN (replace(market_cap, 'T', '')::numeric * 1e12)
                    ELSE market_cap::numeric
                END AS market_cap_numeric
            FROM crypto_data
            WHERE timestamp = (SELECT MAX(timestamp) FROM crypto_data)
        ) subquery
        ORDER BY market_cap_numeric DESC;
        """
    )
    market_cap_data = cursor.fetchall()

    # Combine data for ranking
    scores = {}
    rank_weight = {"volume": 0.4, "volatility": 0.3, "market_cap": 0.3}

    # Assign rankings based on volume
    for rank, (crypto, _) in enumerate(volume_data, start=1):
        scores.setdefault(crypto, 0)
        scores[crypto] += (len(volume_data) - rank + 1) * rank_weight["volume"]

    # Assign rankings based on volatility
    for rank, (crypto, _) in enumerate(volatility_data, start=1):
        scores.setdefault(crypto, 0)
        scores[crypto] += (len(volatility_data) - rank + 1) * rank_weight["volatility"]

    # Assign rankings based on market cap
    for rank, (crypto, _, _) in enumerate(market_cap_data, start=1):
        scores.setdefault(crypto, 0)
        scores[crypto] += (len(market_cap_data) - rank + 1) * rank_weight["market_cap"]

    # Sort cryptos by score
    ranked_cryptos = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print("Final Rankings (Best to Buy):")
    for rank, (crypto, score) in enumerate(ranked_cryptos, start=1):
        print(f"{rank}. {crypto} - Score: {score:.2f}")

    # Insert results into database one by one
    for rank, (crypto, score) in enumerate(ranked_cryptos, start=1):
        try:
            cursor.execute(
                """
                INSERT INTO crypto_rankings (cryptocurrency, rank, score)
                VALUES (%s, %s, %s)
                ON CONFLICT (cryptocurrency, timestamp) DO UPDATE
                SET rank = EXCLUDED.rank, score = EXCLUDED.score
                RETURNING cryptocurrency;
                """,
                (crypto, rank, score)
            )
            print(f"Inserted/Updated: {crypto} with rank {rank} and score {score:.2f}")
        except Exception as e:
            print(f"Skipped: {crypto} due to error: {e}")

    conn.commit()



def analysis8(cursor, conn):
    """
    Graphs the current portfolio distribution using a styled pie chart and includes a table of colors, assets, and percentages.
    """
    # Fetch current assets
    assets_df = fetch_current_assets()
    if assets_df is None or assets_df.empty:
        print("[!] No assets to graph.")
        return

    total_portfolio = 0  # Initialize total portfolio variable
    prices = []  # List to store fetched prices

    for idx, row in assets_df.iterrows():
        asset = row['Asset']
        free = row['Free']
        locked = row['Locked']
        print(Colors.BOLD_WHITE + f"\n[{idx+1}] Asset: {asset}, Free: {free}, Locked: {locked}" + Colors.R)
    
        if asset == 'USDT':
            print(Colors.ORANGE + f"Current value in USDT: [{free}] USDT" + Colors.R)
            # Directly add USDT value to total portfolio
            prices.append(1)  # Price of USDT in USDT is always 1
            total_portfolio += free

        else:
            symbol = f"{asset}USDT"
            try:
                # Retrieve the current price of the asset
                ticker = client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                print(Colors.GREEN + f"Current price of {symbol}: {current_price} USDT" + Colors.R)
                print(Colors.ORANGE + f"Current value in USDT: [{free * current_price}] USDT" + Colors.R)
                # Store the price in the list
                prices.append(current_price)
                total_portfolio += free * current_price  # Add to total portfolio value
            except Exception as e:
                print(f"Error fetching price for {symbol}: {str(e)}")
                # Append NaN or a default value to prices in case of failure
                prices.append(0)

    # Add the prices to the assets DataFrame
    assets_df['Current_Price'] = prices

    # Calculate total value per asset in USDT
    assets_df['Total_Value'] = assets_df['Free'] * assets_df['Current_Price']

    # Filter out assets where total value is zero or negative
    assets_df = assets_df[assets_df['Total_Value'] > 0]

    # Calculate percentages
    total_value = assets_df['Total_Value'].sum()
    assets_df['Percentage'] = (assets_df['Total_Value'] / total_value) * 100

    # Prepare data for the pie chart
    labels = assets_df['Asset']
    sizes = assets_df['Total_Value']
    colors = plt.cm.tab20.colors[:len(labels)]  # Use a colormap for distinct colors

    # Pie chart styling
    fig, ax = plt.subplots(figsize=(10, 10))
    wedges, _ = ax.pie(
        sizes,
        startangle=140,
        colors=colors,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )

    # Add a table with colors, cryptos, and percentages
    legend_table_data = [
        [f"{label}", f"{percentage:.1f}%", plt.Rectangle((0, 0), 1, 1, fc=color)]
        for label, percentage, color in zip(labels, assets_df['Percentage'], colors)
    ]

    # Prepare cell text and cell colors
    cell_text = [[row[0], row[1]] for row in legend_table_data]
    cell_colors = [[row[2].get_facecolor(), 'white'] for row in legend_table_data]

    # Create a table
    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        colLabels=["Crypto", "Percentage"],
        loc="bottom",
        cellLoc="center",
        bbox=[-0.5, -0.4, 1.5, 0.3],  # Adjust position and size
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1])

    # Set a title
    ax.set_title("Portfolio Distribution by Free Assets", fontsize=16)

    # Ensure the pie chart is circular
    ax.axis('equal')

    # Directory for saving pie charts
    charts_dir = "pie_charts"
    os.makedirs(charts_dir, exist_ok=True)

    # Save the plot as PNG
    timestamp = datetime.now().replace(microsecond=0)
    timestamp = timestamp.strftime("%Y-%m-%d-%H-%M-%S")

    chart_name = f"portfolio_distribution_pie_{timestamp}.png"
    chart_path = os.path.join(charts_dir, chart_name)
    plt.savefig(chart_path, bbox_inches='tight')
    print(f"[+] Pie chart with table saved at {chart_path}")
    plt.close()



def analysis9(cursor, conn, start_date='1 year'):
    """Graph exchange """
    graphs_dir = get_graphs_directory(start_date)
    graphs_dir = graphs_dir.replace(' ', '_')

    # Handle start_date input
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    elif isinstance(start_date, str):
        try:
            # Try parsing as a relative time first
            start_date = parse_relative_time(start_date)
        except ValueError:
            # Fallback to parsing as a timestamp
            try:
                start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                raise ValueError(f"Invalid date format for start_date: {start_date}. "
                                 "Expected format: '1 day' or '%Y-%m-%d %H:%M:%S'")

    start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Execute the query
    cursor.execute("""
    SELECT timestamp, exchange_value
    FROM exchange_rate
    ORDER BY timestamp DESC
    """, (start_date_str, end_date))

    results = cursor.fetchall()
    if not results:
        print(f"No data found for the specified criteria: start_date={start_date_str}, end_date={end_date}")
        return None

    timestamps = [row[0] for row in results]
    exchange_values = [float(row[1]) for row in results]

    # Plot USD trends
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, exchange_values, marker='o', markersize=2, label="(USD to MXN)", color='blue')
    plt.title(f"USD to MXN\t{end_date}")
    plt.xlabel("Timestamps")
    plt.ylabel("Value in MXN")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    graph_name = "exchange_rate.png"
    graph_path = os.path.join(graphs_dir, graph_name)
    if os.path.exists(graph_path):
        os.remove(graph_path)
    plt.savefig(graph_path)
    print(Colors.ORANGE + f"Graph saved as\t{Colors.BOLD_WHITE}{graph_path}" + Colors.R)
    plt.close()


def backfill_volatility(cursor, conn, chunk_size=100):
    """
    Backfill missing volatility values for each cryptocurrency in the database.
    """
    print(Colors.BOLD_WHITE + "[!] Backfilling missing historical volatilities" + Colors.R)

    # Fetch all distinct cryptocurrencies
    cursor.execute("SELECT DISTINCT cryptocurrency FROM crypto_data;")
    cryptocurrencies = [row[0] for row in cursor.fetchall()]

    def insert_in_chunks(data, chunk_size):
        """
        Insert data into the database in chunks to prevent overloading.
        """
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            print(
                Colors.CYAN
                + f"[~] Inserting batch {i + 1} to {i + len(chunk)} of {len(data)}"
                + Colors.R
            )
            cursor.executemany(
                """
                INSERT INTO crypto_volatility (cryptocurrency, timestamp, volatility)
                VALUES (%s, %s, %s)
                """,
                chunk
            )
            conn.commit()
            print(
                Colors.GREEN
                + f"[+] Successfully inserted batch {i + 1} to {i + len(chunk)}"
                + Colors.R
            )

    def process_cryptocurrency(cryptocurrency):
        """
        Process a single cryptocurrency to backfill its volatilities.
        """
        print(Colors.ORANGE + f"[~] Processing {cryptocurrency}" + Colors.R)

        # Fetch all timestamps and prices for the cryptocurrency
        cursor.execute(
            """
            SELECT timestamp, price
            FROM crypto_data
            WHERE cryptocurrency = %s
            ORDER BY timestamp ASC;
            """,
            (cryptocurrency,)
        )
        data = cursor.fetchall()

        timestamps = [row[0] for row in data]
        prices = [float(row[1]) for row in data]

        # Fetch existing volatility timestamps to avoid recalculating them
        cursor.execute(
            """
            SELECT timestamp
            FROM crypto_volatility
            WHERE cryptocurrency = %s;
            """,
            (cryptocurrency,)
        )
        existing_timestamps = {row[0] for row in cursor.fetchall()}

        # Collect new volatility records
        new_volatilities = []

        for i in range(1, len(prices)):
            current_timestamp = timestamps[i]
            if current_timestamp not in existing_timestamps:
                _, volatility = calculate_volatility(
                    cryptocurrency,
                    timestamps[:i + 1],
                    prices[:i + 1]
                )
                new_volatilities.append((cryptocurrency, current_timestamp, float(volatility)))

        # Insert volatilities in chunks
        if new_volatilities:
            insert_in_chunks(new_volatilities, chunk_size)
            print(
                Colors.GREEN
                + f"[+] Added {len(new_volatilities)} volatility entries for {cryptocurrency} in chunks of {chunk_size}"
                + Colors.R
            )
        else:
            print(Colors.BLUE + f"[~] No new volatilities to insert for {cryptocurrency}" + Colors.R)

    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_cryptocurrency, crypto): crypto for crypto in cryptocurrencies}

        for future in as_completed(futures):
            cryptocurrency = futures[future]
            try:
                future.result()
            except Exception as e:
                print(
                    Colors.RED
                    + f"[!] Error processing {cryptocurrency}: {e}"
                    + Colors.R
                )

    print(Colors.BOLD_WHITE + "[!] Backfilling complete" + Colors.R)


def process_file(file_path, cursor, conn, chunk_size):
    """
    Process a single CSV file and insert data into the database with progress messages.
    """
    print(f"[!] Starting to process file: {file_path.name}")
    total_rows = sum(1 for _ in file_path.open("r")) - 1  # Subtract 1 for the header
    processed_rows = 0

    with file_path.open("r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip header row
        
        batch = []
        for row in csv_reader:
            try:
                if len(row) != 10:
                    print(f"[!] Skipping malformed row:\t{len(row)}")
                    continue

                prices = [row[4], row[5], row[6], row[7]]
                crypto_currency = row[1].lower()
                timestamp = row[3]
                volume = row[8]
                market_cap = row[9]

                for price in prices:
                    batch.append((
                        crypto_currency,
                        timestamp,
                        price,
                        market_cap,
                        volume,
                        None  # fdv is not in the data, so we set it to NULL
                    ))

                    processed_rows += 1
                    if len(batch) % chunk_size == 0:
                        print(Colors.BOLD_WHITE + f"[{processed_rows}/{total_rows}] Inserting chunk into database..." + Colors.R)
                        cursor.executemany(
                            """
                            INSERT INTO crypto_data (cryptocurrency, timestamp, price, market_cap, volume, fdv)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING;
                            """,
                            batch
                        )
                        conn.commit()
                        print(Colors.GREEN + f"[{processed_rows}/{total_rows}] Chunk inserted successfully!" + Colors.R)
                        batch.clear()

            except Exception as e:
                print(f"[!] Error processing row {e}:\t{len(row)}")

        # Insert any remaining rows
        if batch:
            print(f"[{processed_rows}/{total_rows}] Inserting final batch...")
            cursor.executemany(
                """
                INSERT INTO crypto_data (cryptocurrency, timestamp, price, market_cap, volume, fdv)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
                """,
                batch
            )
            conn.commit()
            print(f"[{processed_rows}/{total_rows}] Final batch inserted successfully!")

    print(f"[+] Finished processing file: {file_path.name}")


cont_crypto = 1
def calculate_assets(db_cursor, db_conn, timestamp):
    global total_assets_sum
    global cont_crypto
    try:
        total_assets_sum = 0  # Ensure the variable is initialized

        # Execute the query to fetch cryptocurrency and amount
        db_cursor.execute("""
        SELECT cryptocurrency, amount
        FROM personal_assets;
        """)

        results = db_cursor.fetchall()

        for row in results:
            crypto = row[0]
            amount = float(row[1])

            if crypto is None:
                continue

            # Fetch the price for the cryptocurrency and timestamp
            db_cursor.execute("""
            SELECT price
            FROM crypto_data
            WHERE timestamp = %s
            AND cryptocurrency = %s;
            """, (timestamp, crypto))

            result = db_cursor.fetchone()

            # Backfill missing data
            if result is None or result[0] is None:
                print(f"[!] No price found for {crypto} at {timestamp}. Attempting to backfill.")
                db_cursor.execute("""
                SELECT price
                FROM crypto_data
                WHERE cryptocurrency = %s
                ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - %s)))
                LIMIT 1;
                """, (crypto, timestamp))
                result = db_cursor.fetchone()

            if result is None or result[0] is None:
                print(f"[!] Unable to find a price for {crypto}. Skipping.")
                continue

            price = float(result[0])

            # Calculate the asset value in USD
            asset_value_usd = amount * price
            total_assets_sum += asset_value_usd

            print(Colors.GREEN + f"[{cont_crypto}] Cryptocurrency: {crypto}, Amount: {amount}, Price: {price}, USD Value: {asset_value_usd}" + Colors.R)
            cont_crypto+=1

            # Insert into assets_per_crypto table
            db_cursor.execute("""
            INSERT INTO assets_per_crypto (cryptocurrency, timestamp, total_value_usd, total_value_mxn)
            VALUES (%s, %s, %s, %s)
            """, (crypto, timestamp, asset_value_usd, None))

        # Commit the inserted rows
        db_conn.commit()

        # Insert total assets
        db_cursor.execute("""
            INSERT INTO total_assets (timestamp, total_assets_usd, total_assets_mxm)
            VALUES (%s, %s, %s)
        """, (timestamp, total_assets_sum, None))
        db_conn.commit()

        print(Colors.CYAN + "\n[!] Total assets inserted!")
        print(f"[!] Total assets for {timestamp}: {total_assets_sum} USD\n" + Colors.R)

    except Exception as err:
        # Rollback the transaction and log the error
        db_conn.rollback()
        print(f"Error calculating total assets: {err}")




def backfill_portfolio(cursor, conn, time_span=""):
    """
    Backfill missing parameters in the portfolio for the last month.
    """
    try:
        # Parse the time span into a timedelta
        #time_delta = parse_time_span(time_span)
        time_delta = timedelta(days=11 * 365)

        # Calculate the start date based on the time span
        start_date = datetime.now() - time_delta

        # Fetch distinct timestamps from the specified time span that are missing in `total_assets`
        cursor.execute("""
        SELECT DISTINCT cd.timestamp
        FROM crypto_data cd
        LEFT JOIN total_assets ta
        ON cd.timestamp = ta.timestamp
        WHERE cd.timestamp >= %s
        AND ta.timestamp IS NULL
        ORDER BY cd.timestamp;
        """, (start_date,))

        timestamps = cursor.fetchall()

        if not timestamps:
            print("[!] No missing timestamps in the last month to backfill.")
            return

        # Backfill for each missing timestamp
        for ts in timestamps:
            timestamp = ts[0]
            print(Colors.BOLD_WHITE + f"[!] Backfilling for timestamp: {timestamp}" + Colors.R)
            calculate_assets(cursor, conn, timestamp)

    except Exception as err:
        print(f"Error during backfilling: {err}")



def fetch_and_plot_stock_data(symbol='ORCL', time_span="1_month"):
    """
    Fetch historical data for a stock using Yahoo Finance and save a plot of its price trends.

    Parameters:
        symbol (str): The stock symbol (default is 'ORCL' for Oracle).
        time_span (str): The time span for historical data (e.g., "1_month", "1_year").
    """
    # Map time_span to Yahoo Finance's period and interval
    time_span_map = {
        "1_day": ("1d", "1m"),
        "1_week": ("5d", "1d"),
        "1_month": ("1mo", "1d"),
        "3_months": ("3mo", "1d"),
        "6_months": ("6mo", "1d"),
        "1_year": ("1y", "1d"),
        "2_years": ("2y", "1d"),
        "5_years": ("5y", "1d"),
    }

    if time_span not in time_span_map:
        raise ValueError(f"Invalid time span: {time_span}")

    period, interval = time_span_map[time_span]

    try:
        # Fetch historical data from Yahoo Finance
        stock = yf.Ticker(symbol)
        historical_data = stock.history(period=period, interval=interval)

        # Skip if no data is found
        if historical_data.empty:
            print(f"No data found for {symbol} for the specified time span.")
            return

        # Extract timestamps and closing prices
        timestamps = historical_data.index
        prices = historical_data["Close"]

        # Fetch the latest price
        live_price = prices.iloc[-1]
        print(f"Live price of {symbol}: ${live_price:.2f}")

        # Plot the data
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, prices, marker='o', markersize=1, label=symbol)
        plt.title(f"{symbol} Price Trends ({time_span.replace('_', ' ')})")
        plt.xlabel("Timestamp")
        plt.ylabel("Price (USD)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Create a directory for saving the plot
        today = datetime.now().strftime("%Y-%m-%d")
        dir_name = f"stock_plots/stock_plots_{today}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # Save the plot
        graph_name = f"{symbol}_price_trend_{time_span}.png"
        graph_path = os.path.join(dir_name, graph_name)
        plt.savefig(graph_path)
        print(f"[!] Plot saved as:\t{Colors.BOLD_WHITE}{graph_path}{Colors.R}")
        plt.close()

    except Exception as e:
        print(f"Error fetching or processing data for {symbol}: {e}")



def show_examples():
    print()
    print(Colors.ORANGE + "[!] Command examples: " + Colors.R)
    print("python3 graph_cryptos.py -a 1\t\t(Graph historical prices of cryptocurrencies)")
    print("python3 graph_cryptos.py -a 1 -d localhost\t\t(Graph historical prices of cryptocurrencies)")
    print("python3 graph_cryptos.py -a 1 -v\t(Graph historical prices of cryptocurrencies and calculate volatility)")
    print("python3 graph_cryptos.py -a 2\t\t(Graph portafolio trends on different images)")
    print("python3 graph_cryptos.py -a 3\t\t(Graph portafolio trends on a single plot)")
    print("python3 graph_cryptos.py -a 4\t\t(Graph volatility trends of each cryptocurrency)")
    print("python3 graph_cryptos.py -a 5\t\t(Analyze and plot the volatility of all cryptocurrencies on the same image.)")
    print("python3 graph_cryptos.py -a 6\t\t(Rank cryptos backtesting cumulative returns)")
    print("python3 graph_cryptos.py -a 7\t\t(Rank cryptos backtesting volume, marketcap and cumulative returns)")
    print("python3 graph_cryptos.py -a 8\t\t(Graph portfolio as circular graph)")
    print("python3 graph_cryptos.py -a 9\t\t(Graph exchange rate of USD to MXN)")
    print("python3 graph_cryptos.py -a 10\t\t(Graph exchange rate of USD to ORCL)")
    print("python3 graph_cryptos.py -f\t\t(Graph full previous options)")
    print("python3 graph_cryptos.py -b volatility\t(Back fill missing volatility registries)")
    print("python3 graph_cryptos.py -b prices\t(Back fill missing volatility registries)")
    print("python3 graph_cryptos.py -b portfolio\t(Back fill missing portfolio registries)")
    print("python3 graph_cryptos.py -e\t\t(Show this message)")
    print("python3 graph_cryptos.py -h\t\t(Show help message)")
    sys.exit(0)




if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()
    
    if args.examples is True:
        show_examples()
    if args.analysis:
        args.analysis = int(args.analysis)

    # Load environment variables from .env file
    load_dotenv()

    # Database connection configuration from environment variables
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')

    # If db_host received setup
    if args.db_host:
        if args.db_host == 'localhost':
            DB_HOST = args.db_host
        else:
            try:
                ipaddress.IPv4Address(args.db_host)  # Validate if it's a proper IPv4 address
                DB_HOST = args.db_host
            except ipaddress.AddressValueError:
                print(f"[!] Invalid IPv4 address: {args.db_host}")
                sys.exit(1)  # Exit with an error if the address is not valid


    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        print(Colors.GREEN + "[!] Connection to database open!" + Colors.R)
        cursor = conn.cursor()

        # Analysis1
        if args.analysis == 1:
            if args.volatility:
                analysis1(cursor, conn, True)
            else:
                if args.time:
                    analysis1(cursor, conn, args.time, False)
                else:
                    analysis1(cursor, conn, False)

        # Analysis2
        if args.analysis == 2:
            if args.time:
                analysis2(cursor, conn, args.time)
            else:
                analysis2(cursor, conn)

        # Analysis3
        if args.analysis == 3:
            if args.time:
                analysis3(cursor, conn, args.time)
            else:
                analysis3(cursor, conn)

        # Analysis4
        if args.analysis == 4:
            if args.time:
                analysis4(cursor, conn, args.time)
            else:
                analysis4(cursor, conn)

        # Analysis5
        if args.analysis == 5:
            analysis5(cursor, conn)

        # Rank cryptos by their cumulative returns
        if args.analysis == 6:
            top_rank = 60
            interval = '1h'
            start_date = '1 Jan 2024'
            #interval = '1d'
            #start_date = '15 Dec 2024'
            best_cryptos = analysis6(top_rank, conn, cursor, interval, start_date)
            print("Top performing cryptos:")
            for crypto in best_cryptos:
                print(f"{crypto['symbol']}: {crypto['cumulative_return']:.2f}")

        # Fix and analyze volume
        if args.analysis == 7:
            analysis7(cursor, conn)
            pass

        # Graph a pie chart
        if args.analysis == 8:
            analysis8(cursor, conn)

        # Graph exchange rates along time
        if args.analysis == 9:
            if args.time:
                analysis9(cursor, conn, args.time)
            else:
                analysis9(cursor, conn)

        # Graph exchange rates along time of OCRL asset
        if args.analysis == 10:
            if args.time:
                fetch_and_plot_stock_data(symbol='ORCL', time_span=args.time)
            else:
                fetch_and_plot_stock_data(symbol='ORCL')


        # Plot all graphs
        if args.full:
            periods = ['1_year', '1_month', '1_day']

            for period in periods:
                analysis1(cursor, conn, period, False)
                analysis2(cursor, conn, period)
                analysis3(cursor, conn, period)
                analysis4(cursor, conn, period)
                analysis5(cursor, conn, period)


        # Backfill
        if args.backfill == "volatility":
            backfill_volatility(cursor, conn)

        # Backfill
        if args.backfill == "prices":
            backfill_prices(cursor, conn)

        # Backfill
        if args.backfill == "portfolio":
            backfill_portfolio(cursor, conn)
            if args.time:
                backfill_portfolio(cursor, conn, args.time)


    except psycopg2.Error as db_err:
        print(f"Database error occurred: {db_err}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            print(Colors.GREEN + "[!] Connection to database closed!" + Colors.R)
            conn.close()

