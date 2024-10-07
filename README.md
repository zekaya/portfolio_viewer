# Portfolio Analysis and Visualization Tool

This tool fetches financial data, calculates portfolio values, and generates visualizations for tracking portfolio performance over time. It supports assets from both the US and Turkish markets, handling stocks, funds, and foreign exchange conversions.

## Features

- **Transaction Data Processing:** 
  - Reads transaction data from an Excel file (`Transactions_sample.xlsx`).
  - Groups assets based on their type (e.g., stocks, funds, cash).
  - Calculates portfolio value and performance based on transactions.

- **Data Fetching:**
  - Fetches historical stock price data from Yahoo Finance for US and Turkish markets.
  - Retrieves fund prices from TEFAS for Turkish funds (including retirement funds).
  - Converts fund prices into USD using the latest USD/TRY exchange rate.

- **Portfolio Calculation:**
  - Computes the total portfolio value by multiplying adjusted lot sizes (shares) by current market prices.
  - Tracks profit/loss both in absolute and percentage terms for each asset in the portfolio.

- **Visualization:**
  - Generates a pie chart showing the portfolio distribution across different assets.
  - Plots a stacked area chart that visualizes the portfolio value evolution over time, adjusted for each asset type.

- **Summary Output:**
  - Displays the total portfolio value, profit/loss, and asset details (e.g., adjusted lots, current price, market value).
  - Provides a detailed table of asset holdings, including weighted mean prices, total cost, and market value.

## Installation

To run this project, make sure you have the following Python packages installed:

```bash
pip install -r requirements.txt

