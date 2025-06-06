# CryptoCrawler
This is a tool for automating the fetching of cryptocurrency prices in real time
writen in python and intented to run on a Debian based system where Tensorflow will be build.

The idea of this project is to distribute the crawling of different cryptos across nodes.
And store historical on a distributed PostgreSQL database.

# Features
Under 'analysis' directory there exist graph_cryptos.py for graphing the different cryptos
with data from the database for different time periods.

Tensorflow is used inside this project for training inference AI models over timestamp data.

# Track personal assets
This project creates schemas for tracking your personal assets across timestamps. You need to setup
your binance API key on a secure file for this to work using your wallet.

# Requirements
This was tested on a Debian based system (Ubuntu in particular). The package manager apt is used in this code.

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
