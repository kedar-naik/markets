"""
Script Name: yfinance_practice.py
Description: This script demonstrates how to use the yfinance library.
Author: Kedar Naik
Date: May 5, 2024
License: Personal Use License
"""

# Dependencies
import yfinance as yf
import os
import sys
import pandas as pd
import numpy as np

# Usage Instructions
"""
Usage: python Example.py [options]

Options:
    -h, --help      Show this help message and exit
    -f, --file      Input file path
    -o, --output    Output file path
"""

# [user input] input a ticker symbol of interest
ticker_of_interest = "QQQ"

# create a yfinance ticker object for the ticker of interest
ticker = yf.Ticker(ticker=ticker_of_interest)

# print out some information about the ticker of interest
print(f'\n\t- Ticker of interest: {ticker.info["symbol"]}\n')
print(f'\t  - Name: \t{ticker.info["longName"]}')
print(f'\t  - Type: \t{ticker.info["quoteType"]}')

history = ticker.history(period="3mo", interval="1d")

history['Percent Change'] = 100.0 * history['Close'].pct_change()

