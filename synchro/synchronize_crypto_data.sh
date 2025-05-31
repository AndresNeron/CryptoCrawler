#!/bin/bash

# The intention of this script is to execute all SQL scripts under /tmp/crypto_data

# Function to run a SQL script as the postgres user
find /tmp/crypto_data -type f | grep -vE 'create.sql|crawled_cryptos_ranked'  | while read script_path;
do
	echo "$script_path"
	sudo -u postgres psql -d cryptocurrency -f "$script_path"
done
