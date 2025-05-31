# CryptoCrawler
This is a tool for automating the fetching of cryptocurrency prices in real time
writen in python and intented to run on a Debian based system where Tensorflow will be build.

The idea of this project is to distribute the crawling of different cryptos across nodes.

# Features
Under 'analysis' directory there exist graph_cryptos.py for graphing the different cryptos
with data from the database for different time periods.

Tensorflow is used inside this project for training inference AI models over timestamp data.

# 
This project creates schemas for tracking your personal assets across timestamps. You need to setup binance API keys
for this to work for your wallet on a secure file.

# Requirements
This was tested on a Debian based system (Ubuntu in particular). The package manager apt is used in this code.

# Installation

You can install this program simple executing the installer on your system like this:
```bash
./install_crypto_crawler.sh
```

The command above will install all the necessary apt packages for building the application.




