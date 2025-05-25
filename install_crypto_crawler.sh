#!/bin/bash

# This script sets up everything for running crypto_crawler

set -e

# Required system packages
REQUIRED_PACKAGES=(libpq-dev python3-dev gcc postgresql python3-virtualenv )

echo "[+] Installing required packages..."
sudo apt update
sudo apt install -y "${REQUIRED_PACKAGES[@]}"

# Define paths
rootPath="$(pwd)"
envPath="$rootPath/cryptoEnvU"
scrapCryptos="$rootPath/binance/scrap_cryptos.py"
launcherScript="$rootPath/scrap_cryptos.sh"

# Function to run a SQL script as the postgres user
run_sql_script() {
    local script_path="$1"
    local tmp_file="/tmp/$(basename "$script_path")"

    cp "$script_path" "$tmp_file"
    chmod 644 "$tmp_file"
    sudo -u postgres psql -f "$tmp_file"
    rm "$tmp_file"
}

# Define root path
sql_dir="$rootPath/analysis/sql"

# Run each SQL script
run_sql_script "$sql_dir/create.sql"
run_sql_script "$sql_dir/crawled_cryptos.sql"

# Create virtual environment
echo "[+] Creating virtual environment..."
virtualenv "$envPath" --python=python3

# Activate and install Python dependencies
echo "[+] Installing Python dependencies..."
source "$envPath/bin/activate"
pip install -r requirements.txt
deactivate

# Ensure launcher script is executable
chmod +x "$launcherScript"

# Install the cronjob
echo "[+] Installing cron job to /etc/cron.d/crypto_crawler..."

cron_file="/etc/cron.d/crypto_crawler"
cron_entry="*/5 * * * * root $launcherScript >> $rootPath/crypto_crawler.log 2>&1"

# Write to the system-wide cron file
echo "$cron_entry" | sudo tee "$cron_file" > /dev/null

# Ensure permissions and ownership are correct
sudo chmod 644 "$cron_file"
sudo chown root:root "$cron_file"

echo "[+] Cron job installed: $cron_entry"
