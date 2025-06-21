# CryptoCrawler
CryptoCrawler is a Python-based tool for automating the real-time fetching of cryptocurrency prices.

The core idea of this project is to distribute the crawling of various cryptocurrencies across nodes 
in a peer-to-peer (P2P) network, and store historical price data in a distributed PostgreSQL database.

Additional capabilities include graphing the price evolution of different cryptocurrencies over time, 
and performing predictions using Long Short-Term Memory (LSTM) and other neural network models.

    💡 Due to the use of TensorFlow, this project is intended to run on Ubuntu or other Debian-based 
	systems officially supported by TensorFlow.


# Features
- Graphing tools:
Located in the analysis directory, graph_cryptos.py lets you visualize cryptocurrency price data from 
the database over different time periods.

- AI-powered prediction:
TensorFlow is used to train and perform inference using LSTM and other models on timestamped price data.

# Track personal assets
CryptoCrawler supports tracking your personal crypto assets over time.
To enable this feature:

- Configure your Binance API key in a secure .env file.

- The system will create schemas in the database for logging and analyzing your wallet's historical value.

- If Binance API key with read permissions is loaded, the program will fetch the prices from Binance API too.

# Live Trading Bot

Located in the `binance` directory, `live_trading_bot.py` enables live trading on Binance using strategy based on:

- **SMA (Simple Moving Average)** crossover signals with SMA-30 and SMA-70
- **Stop-loss** mechanisms for risk management

To enable live trading, you must provide a Binance API key with **trading permissions** in your `.env` file.

> Use this feature in testing mode.
> ⚠️If you use this feature in real mode is at your own risk. It is strongly recommended to test in test mode
or with a paper trading setup before committing real funds.

# Requirements
- Tested on Debian-based systems, especially Ubuntu.

- Uses apt for package management.

- Requires Python 3.x, PostgreSQL, and TensorFlow.

# Installation

You can install this program simple executing the installer on your system like this:
```bash
git clone https://github.com/AndresNeron/CryptoCrawler.git
cd CryptoCrawler
./install_crypto_crawler.sh
```

The command above will install all the necessary apt packages for building the application. 
And will include cronjob located at /etc/cron.d/crypto_crawler like this:

```bash
*/15 * * * * root /home/eq12/GitHub/CryptoCrawler/scrap_cryptos.sh >> /home/eq12/GitHub/CryptoCrawler/crypto_crawler.log 2>&1
```

The cronjob above will fetch the real price of each crypto stored on crawled_cryptos table and will save it in the 
table crypto_data of the PostgreSQL database.


# Active development

This project is in active development. Some new planned features are, 

1. Track historical cryptocurrency prices from different sources.
2. Track historical stock prices from different sources.
