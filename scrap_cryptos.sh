#!/bin/bash

# Imitate the shell environment
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH

DEBUG_LOG="$(pwd)/debug.log"
NORMAL_LOG="$(pwd)/graphs.log"

# Activate the virtual environment
VENV_NAME="cryptoEnvU"
VENV_ACTIVATE="$(pwd)/$VENV_NAME/bin/activate"
PYTHON_EXECUTABLE="$(pwd)/$VENV_NAME/bin/python3"
if [ -f "$VENV_ACTIVATE" ]; then
	source "$VENV_ACTIVATE"
	echo "Virtual environment ACTIVATED using $VENV_ACTIVATE" >> "$DEBUG_LOG"
else
	echo "Virtual environment not found at $VENV_ACTIVATE" >> "$DEBUG_LOG"
	exit 1
fi
source "$VENV_ACTIVATE"

scrap_crypto(){
	cd "$(pwd)/binance"
	echo "Test executed at $(date)	$(pwd)" >> "$DEBUG_LOG"

	# Execute the Python script
	PYTHON_SCRIPT="$(pwd)/binance/scrap_cryptos.py"
    echo "Step 1: Starting $PYTHON_SCRIPT" >> "$DEBUG_LOG"
    "$PYTHON_EXECUTABLE" "$PYTHON_SCRIPT"
    "$PYTHON_EXECUTABLE" "$PYTHON_SCRIPT" -p
    if [ $? -eq 0 ]; then
        echo "Step 2: Script completed successfully" >> "$DEBUG_LOG"
    else
        echo "Error: Script $PYTHON_SCRIPT failed" >> "$DEBUG_LOG"
    fi
    echo "Python script $PYTHON_SCRIPT executed at $(date)" >> "$NORMAL_LOG"

}

scrap_crypto

deactivate
