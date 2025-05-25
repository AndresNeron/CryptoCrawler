#!/usr/bin/env python3

# This script serves for scraping data related to cryptocurrencies from various sources.
import os
import re
import sys
import argparse
import requests
import psycopg2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from binance.client import Client

# Get the parent of the parent directory
parent_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_parent_path not in sys.path:
    sys.path.append(parent_parent_path)


# Personal packages
from utils.colors import Colors

# Load .env
load_dotenv()

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="A python based tools for crawling crypto data.")
    parser.add_argument("-p", "--proxychains",  action="store_true", help="\t\tThis option must be enabled when traffic will be routed by proxychains.")
    return parser.parse_args()


timestamp_global = datetime.now().replace(microsecond=0)
timestamp_global = timestamp_global.strftime("%Y-%m-%d %H:%M:%S")

def get_crawl_cryptos(conn, cursor):
    """
    Fetch the list of cryptocurrencies from the 'crawled_cryptos' table.
    """
    currencys = []

    # SQL query to fetch cryptocurrencies
    cursor.execute("SELECT cryptocurrency FROM crawled_cryptos;")
    rows = cursor.fetchall()

    # Process the results
    for row in rows:
        currencys.append(row[0])  # Assuming 'cryptocurrency' is the first column in the result

    return currencys


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

    # Ensure there is enough data for calculation
    if len(prices) < 2:
        return None, 0

    # Convert prices to a NumPy array
    prices = np.array(prices, dtype=np.float64)

    # Ensure no zero or negative prices
    if np.any(prices <= 0):
        return None, 0

    # Calculate percentage returns
    returns = np.diff(prices) / prices[:-1]

    # Handle edge cases with NaN or Inf values in returns
    returns = returns[np.isfinite(returns)]

    # If there are no valid returns, return 0 volatility
    if len(returns) == 0:
        return None, 0

    # Calculate the standard deviation (volatility)
    volatility = np.std(returns)

    # Use the latest timestamp for the result
    latest_timestamp = timestamps[-1]

    return latest_timestamp, volatility


cont_volatility = 1
def insert_volatility(cursor, conn):
    """
    Calculate volatility of each tracked cryptocurrency
    """
    global timestamp_global
    global cont_volatility

    cursor.execute(
    """
    SELECT DISTINCT cryptocurrency
    FROM crawled_cryptos;
    """)

    assets = cursor.fetchall()

    crypto_cont = 1
    for row in assets:
        asset = row[0]

        cursor.execute(
        """
        SELECT cryptocurrency, price, timestamp
        FROM crypto_data
        WHERE cryptocurrency = %s
        ORDER BY timestamp DESC;
        """, (asset,))

        results = cursor.fetchall()

        # Data preparation for the graphing
        timestamps = [result[2] for result in results]
        prices = [float(result[1]) for result in results]

        # Calculate a single volatility value
        latest_timestamp, single_volatility = calculate_volatility(asset, timestamps, prices)

        # Save the volatility to the database
        cursor.execute(
            """
            INSERT INTO crypto_volatility (cryptocurrency, timestamp, volatility)
            VALUES (%s, %s, %s)
            """,
            (asset, timestamp_global, float(single_volatility)),
        )
        conn.commit()
        print(Colors.BOLD_WHITE + f"[{cont_volatility}] Volatility of " + Colors.GREEN + f"{asset}" + Colors.BOLD_WHITE + " inserted!" + Colors.R)
        cont_volatility += 1


cont_crypto = 1
def scrap_currency(crypto_currency, db_cursor, db_conn):
    global cont_crypto
    url = f"https://coinmarketcap.com/currencies/{crypto_currency}/"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        registry = [crypto_currency]

        # Ensure timestamp_global exists
        try:
            registry.append(timestamp_global)
        except NameError:
            print(f"[!] 'timestamp_global' is not defined.")
            return

        # Extract dollar values
        span_elements = soup.find_all('span')
        values_found = []

        for span in span_elements:
            text = span.get_text(strip=True).replace(",", "")
            if '$' in text:
                values_found.append(text.replace('$', ''))
            if len(values_found) >= 4:
                break  # We only need 4 values

        registry.extend(values_found[:4])

        # Fill missing fields with NULL
        while len(registry) < 6:
            registry.append(None)

        final_registry = ','.join(str(val) if val is not None else 'NULL' for val in registry[:6])

        if registry[2] is not None:
            try:
                db_cursor.execute(
                    """
                    INSERT INTO crypto_data (cryptocurrency, timestamp, price, market_cap, volume, fdv)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    tuple(registry[:6])
                )
                db_conn.commit()
                print(Colors.ORANGE + f"[{cont_crypto}] Inserted: {final_registry}" + Colors.R)
            except Exception as db_err:
                db_conn.rollback()  # Roll back only this failed transaction
                print(f"[{cont_crypto}] DB insert failed for '{crypto_currency}': {db_err}")
        else:
            print(f"[{cont_crypto}] Skipped '{crypto_currency}' - price not found.")

        cont_crypto += 1

    except requests.exceptions.HTTPError as http_err:
        print(f"[{cont_crypto}] HTTP error for '{crypto_currency}': {http_err}")
        cont_crypto += 1
    except Exception as err:
        print(f"[{cont_crypto}] Error for '{crypto_currency}': {err}")
        cont_crypto += 1


# This method inserts a registry for calculating the total assets for a specific timestamp
total_assets_sum = 0

def calculate_assets(db_cursor, db_conn, timestamp):
    global total_assets_sum
    try:
        total_assets_sum = 0  # Ensure the variable is initialized

        # Execute the query to fetch cryptocurrency and amount
        db_cursor.execute("""
        SELECT c.slug, p.amount
        FROM personal_assets p
        JOIN crypto_slugs c ON p.cryptocurrency = c.symbol
        WHERE p.timestamp = (
            SELECT MAX(timestamp)
            FROM personal_assets
        );
        """)

        # Fetch all results
        results = db_cursor.fetchall()

        # Fetch the latest exchange rate for USD to MXN
        db_cursor.execute("""
        SELECT exchange_value FROM exchange_rate
        WHERE currency1 = %s AND currency2 = %s
        ORDER BY id DESC LIMIT 1;
        """, ('USD', 'MXN'))

        exchange_value = db_cursor.fetchone()
        if exchange_value is not None:
            exchange_value = float(exchange_value[0])
        else:
            print("[!] No exchange rate found. Defaulting to 1 for MXN calculation.")
            exchange_value = 1  # Default to 1 to avoid errors

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

            # Check if a result was found
            if result is None or result[0] is None:
                print(f"[!] No price found for {crypto} at {timestamp}. Skipping.")
                continue

            price = float(result[0])

            # Calculate the asset value in USD and MXN
            asset_value_usd = amount * price
            asset_value_mxm = asset_value_usd * exchange_value
            total_assets_sum += asset_value_usd

            print(Colors.GREEN + f"[!] Cryptocurrency: {crypto}, Amount: {amount}, Price: {price}, USD Value: {asset_value_usd}, MXM Value: {asset_value_mxm}" + Colors.R)

            # Insert into assets_per_crypto table
            db_cursor.execute("""
            INSERT INTO assets_per_crypto (cryptocurrency, timestamp, total_value_usd, total_value_mxn)
            VALUES (%s, %s, %s, %s)
            """, (crypto, timestamp, asset_value_usd, asset_value_mxm))

        # Commit after inserting all rows into assets_per_crypto
        db_conn.commit()

        # Calculate total assets in MXN
        total_assets_mxm = total_assets_sum * exchange_value

        # Insert the total assets into the total_assets table
        db_cursor.execute("""
            INSERT INTO total_assets (timestamp, total_assets_usd, total_assets_mxm)
            VALUES (%s, %s, %s)
        """, (timestamp, total_assets_sum, total_assets_mxm))
        db_conn.commit()

        print(Colors.CYAN + "\n[!] Total assets inserted!")
        print(f"[!] Total assets for {timestamp}: {total_assets_sum} USD, {total_assets_mxm} MXM\n" + Colors.R)
    except Exception as err:
        print(f"Error calculating total assets: {err}")


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


# Another method for retrieving prices, this source is Binance
total_assets_sum = 0
def calculate_assets_binance(db_cursor, db_conn, timestamp):
    global total_assets_sum
    total_assets_sum = 0
    assets_df = fetch_current_assets()


    if assets_df is not None:
        print(Colors.ORANGE + "[!] Looping through Binance personal assets:\n" + Colors.R)
        total_portfolio = 0

        for idx, row in assets_df.iterrows():
            asset = row['Asset']
            free = row['Free']
            locked = row['Locked']
            print(Colors.BOLD_WHITE + f"\n[{idx+1}] Asset: {asset}, Free: {free}, Locked: {locked}" + Colors.R)


            try:
                # Insert free amount into personal_assets
                db_cursor.execute("""
                INSERT INTO personal_assets (cryptocurrency, timestamp, amount)
                VALUES (%s, %s, %s)
                """, (asset, timestamp, free))

                # Commit after inserting all rows into assets_per_crypto
                db_conn.commit()
                print(Colors.GREEN + "[!] Total values per asset inserted into personal_assets" + Colors.R)
            except Exception as e:
                print(f"[x] An error ocurred:\t{e}")


            if asset == "ETHW":
                print(Colors.ORANGE + f"[!] Skipping {asset}!" + Colors.R)
                continue

            if free > 0:
                symbol = f"{asset}USDT"

                try:

                    # Fetch the latest exchange rate for USD to MXN
                    db_cursor.execute("""
                    SELECT exchange_value FROM exchange_rate
                    WHERE currency1 = %s AND currency2 = %s
                    ORDER BY id DESC LIMIT 1;
                    """, ('USD', 'MXN'))

                    exchange_value = db_cursor.fetchone()
                    if exchange_value is not None:
                        exchange_value = float(exchange_value[0])
                    else:
                        print("[!] No exchange rate found. Defaulting to 1 for MXN calculation.")
                        exchange_value = 1
                    

                    if asset == "USDT":
                        asset_value_usd = free

                    else:
                        # Retrieve the current price of the asset
                        ticker = client.get_symbol_ticker(symbol=symbol)
                        current_price = float(ticker['price'])

                        # Calculate the asset value in USD and MXN
                        asset_value_usd = free * current_price
                        asset_value_mxm = asset_value_usd * exchange_value

                    total_assets_sum += asset_value_usd

                    print(Colors.GREEN + f"[!] Cryptocurrency: {asset}, Amount: {free}, Price: {current_price}, USD Value: {asset_value_usd}, MXM Value: {asset_value_mxm}" + Colors.R)

                    # Insert into assets_per_crypto table
                    db_cursor.execute("""
                    INSERT INTO assets_per_crypto (cryptocurrency, timestamp, total_value_usd, total_value_mxn)
                    VALUES (%s, %s, %s, %s)
                    """, (asset, timestamp, asset_value_usd, asset_value_mxm))


                    # Commit after inserting all rows into assets_per_crypto
                    db_conn.commit()

                    print(Colors.GREEN + "[!] Amount per asset from Binance inserted into personal_assets!" + Colors.R)


                except Exception as e:
                    print(Colors.RED + f"Could not fetch price or run test trading for {symbol}: {e}" + Colors.R)

        try:
            # Calculate total assets in MXN
            total_assets_mxm = total_assets_sum * exchange_value

            # Insert the total assets into the total_assets table
            db_cursor.execute("""
                INSERT INTO total_assets (timestamp, total_assets_usd, total_assets_mxm)
                VALUES (%s, %s, %s)
            """, (timestamp, total_assets_sum, total_assets_mxm))
            db_conn.commit()

            print(Colors.CYAN + "\n[!] Total assets inserted!")
            print(f"[!] Total assets for {timestamp}: {total_assets_sum} USD, {total_assets_mxm} MXM\n" + Colors.R)

        except Exception as e:
            print(Colors.RED + f"[*] Error while trying to insert the total_assets: {e}" + Colors.R)



def calculate_exchange_rates(db_cursor, db_conn, timestamp):
    """
    Fetches the USD to MXN exchange rate by scraping a public website and stores it in the database.
    """
    url = "https://www.x-rates.com/calculator/?from=USD&to=MXN&amount=1"
    try:
        # Send a GET request to the exchange rate website
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses

        # Parse the HTML content with Beautiful Soup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the exchange rate from the relevant HTML element
        rate_element = soup.find("span", class_="ccOutputTrail").previous_sibling
        if rate_element is None:
            print("Failed to find the exchange rate on the page.")
            return

        # Clean and convert the extracted rate to a float
        usd_to_mxn = float(rate_element.get_text(strip=True).replace(",", ""))

        print(Colors.PURPLE + f"\n[!] USD to MXN exchange rate: {usd_to_mxn} at {timestamp}" + Colors.R)

        # Insert the rate into the database
        db_cursor.execute(
            """
            INSERT INTO exchange_rate (timestamp, currency1, currency2, exchange_value)
            VALUES (%s, %s, %s, %s)
            """,
            (timestamp, 'USD', 'MXN', usd_to_mxn)
        )
        db_conn.commit()

        print(Colors.PURPLE + "[!] Exchange rate successfully stored in the database.\n" + Colors.R)

    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except Exception as err:
        print(f"An error occurred: {err}")




if __name__ == "__main__":

    # Parse command-line arguments
    args = parse_arguments()

    # Database connection configuration from environment variables
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')


    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cursor = conn.cursor()

        if not conn or not cursor:
            print("[*] Something failed with the connection!")
            sys.exit(1)

        currencys = get_crawl_cryptos(conn, cursor)

        if not args.proxychains:
            # Loop through cryptocurrencies and scrape data from coinmarketcap
            for currency in currencys:
                scrap_currency(currency, cursor, conn)

            # Calculate exchange rate
            calculate_exchange_rates(cursor, conn, timestamp_global)

            # Calculate total assets after scraping data
            #calculate_assets(cursor, conn, timestamp_global)

            # Calculate and insert volatility
            insert_volatility(cursor, conn)

        else:
            # This part must be routed through proxychains
            try:
                # Database connection configuration from environment variables
                api_key = os.getenv('API_KEY')
                api_secret = os.getenv('SECRET_KEY')
                client = Client(api_key, api_secret)

                # Calculate exchange rate
                calculate_exchange_rates(cursor, conn, timestamp_global)

                # This could failed to restrictions from binance (IP, rate limiting)
                calculate_assets_binance(cursor, conn, timestamp_global)
            except:
                print("\n[!] Fetching prices from Binance failed!")


    except psycopg2.Error as db_err:
        print(f"Database error occurred: {db_err}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
