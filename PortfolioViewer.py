import pandas as pd
import yfinance as yf
from tefas import Crawler
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")

df_trans = pd.read_excel("Transactions_sample.xlsx")
start_date = '2024-07-12'

def get_stock_data(tickers, start_date):
  data = yf.download(tickers, start=start_date, end=datetime.now().strftime('%Y-%m-%d'))
  return data['Adj Close']

def get_tefas_inv_data(tickers, start_date):
  data = pd.DataFrame()
  for i in range(len(tickers)):
    ticker_data = tefas.fetch(start=start_date, end=last_bz_day_tr, name=tickers[i], columns=["code", "date", "price"], kind='YAT')
    #print(ticker_data)
    data['Date'] = pd.to_datetime(ticker_data['date']).dt.strftime('%Y-%m-%d')
    data[tickers[i]] = ticker_data['price']/usdtry
  data = data.set_index('Date')
  return data

def get_tefas_ret_data(tickers, start_date):
  data = pd.DataFrame()
  for i in range(len(tickers)):
    ticker_data = tefas.fetch(start=start_date, end=last_bz_day_tr, name=tickers[i], columns=["code", "date", "price"], kind='EMK')
    #print(ticker_data)
    data['Date'] = pd.to_datetime(ticker_data['date']).dt.strftime('%Y-%m-%d')
    data[tickers[i]] = ticker_data['price']/usdtry
  data = data.set_index('Date')
  return data

def calculate_portfolio_value(stock_data, holdings):
  portfolio_value = pd.DataFrame(index=stock_data.index, columns=holdings.keys())

  for ticker, transactions in holdings.items():
      shares = 0
      for date, price, amount in transactions:
          shares += amount
          mask = stock_data.index >= date
          portfolio_value.loc[mask, ticker] = stock_data.loc[mask, ticker] * shares

  return portfolio_value

def plot_stacked_area(portfolio_value):
  sns.set_palette('colorblind')
  ax = portfolio_value.plot.area(stacked=True, figsize=(12, 6), color=sns.color_palette('colorblind'))
  plt.title('Portfolio Value Over Time')
  plt.xlabel('Date')
  plt.ylabel('Value ($)')
  plt.legend(title='Assets', bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.grid()
  plt.tight_layout()
  plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
  plt.show()


try:
    wm = lambda x: np.ma.average(x, weights=df_trans.loc[x.index, "Qty:"])
    df_grouped = df_trans.groupby(["Asset:"]).agg(adjusted_lots=("Qty:", "sum"), price_weighted_mean=("Unit Cost (USD):", wm))
except Exception as e:
    print(e)

df_grouped['total_cost'] = df_grouped['adjusted_lots'] * df_grouped['price_weighted_mean']

df_grouped.reset_index(inplace=True)
df_grouped.rename(columns={"Asset:": "asset"}, inplace=True)

df_grouped['current_price'] = np.zeros(df_grouped['asset'].size)
df_grouped['market_value'] = np.zeros(df_grouped['asset'].size)

tefas = Crawler()

dummy_tckr_us = yf.Ticker("ADI")
ts_data_us = dummy_tckr_us.history(period='1d')
last_bz_day_us =ts_data_us.index[0].to_pydatetime().date().strftime("%Y-%m-%d")

dummy_tckr_tr = yf.Ticker("ASELS.IS") # Ticker for Turkish market open days
ts_data_tr = dummy_tckr_tr.history(period='1d')
last_bz_day_tr =ts_data_tr.index[0].to_pydatetime().date().strftime("%Y-%m-%d")

fx_ticker = yf.Ticker('TRY=X')
hist = fx_ticker.history()
usdtry = hist['Close'].iloc[-1]

for asset in df_grouped['asset']:
    type_name = df_trans['Type'][df_trans.index[df_trans['Asset:'] == asset].tolist()]

    if type_name.iloc[0] == 'Stock - US Tech':
        us_stock_ticker = yf.Ticker(asset)
        hist = us_stock_ticker.history()
        stock_price = hist['Close'].iloc[-1]
        df_grouped.loc[df_grouped['asset'] == asset, 'current_price'] = stock_price;
        df_grouped.loc[df_grouped['asset'] == asset, 'market_value'] = stock_price * df_grouped.loc[df_grouped['asset'] == asset, 'adjusted_lots'];
    elif type_name.iloc[0] == 'Fund - TR Stocks':
        data = tefas.fetch(start=last_bz_day_tr, end=last_bz_day_tr, name=asset, columns=["code", "date", "price"])
        df_grouped.loc[df_grouped['asset'] == asset, 'current_price'] = float(data['price']) / usdtry;
        df_grouped.loc[df_grouped['asset'] == asset, 'market_value'] = float(data['price'] / usdtry) * df_grouped.loc[df_grouped['asset'] == asset, 'adjusted_lots'];
    elif type_name.iloc[0] == 'Cash':
        df_grouped.loc[df_grouped['asset'] == asset, 'current_price'] = df_grouped.loc[df_grouped['asset'] == asset, 'price_weighted_mean']
        df_grouped.loc[df_grouped['asset'] == asset, 'market_value'] = df_grouped.loc[df_grouped['asset'] == asset, 'total_cost']
    elif type_name.iloc[0] == 'Fund - US Tech':
        us_stock_ticker = yf.Ticker(asset)
        hist = us_stock_ticker.history()
        stock_price = hist['Close'].iloc[-1]
        df_grouped.loc[df_grouped['asset'] == asset, 'current_price'] = stock_price;
        df_grouped.loc[df_grouped['asset'] == asset, 'market_value'] = stock_price * df_grouped.loc[df_grouped['asset'] == asset, 'adjusted_lots'];
    elif type_name.iloc[0] == 'Fund - TR Retirement':
        data = tefas.fetch(start=last_bz_day_tr, end=last_bz_day_tr, name=asset, columns=["code", "date", "price"], kind='EMK')
        df_grouped.loc[df_grouped['asset'] == asset, 'current_price'] = float(data['price']) / usdtry;
        df_grouped.loc[df_grouped['asset'] == asset, 'market_value'] = float(data['price'] / usdtry) * df_grouped.loc[df_grouped['asset'] == asset, 'adjusted_lots'];
    elif type_name.iloc[0] == 'Fund - TR Foreign':
        data = tefas.fetch(start=last_bz_day_tr, end=last_bz_day_tr, name=asset, columns=["code", "date", "price"])
        df_grouped.loc[df_grouped['asset'] == asset, 'current_price'] = float(data['price']) / usdtry;
        df_grouped.loc[df_grouped['asset'] == asset, 'market_value'] = float(data['price'] / usdtry) * df_grouped.loc[df_grouped['asset'] == asset, 'adjusted_lots'];
    elif type_name.iloc[0] == 'Stock - US Consumer Discretionary':
        us_stock_ticker = yf.Ticker(asset)
        hist = us_stock_ticker.history()
        stock_price = hist['Close'].iloc[-1]
        df_grouped.loc[df_grouped['asset'] == asset, 'current_price'] = stock_price;
        df_grouped.loc[df_grouped['asset'] == asset, 'market_value'] = stock_price * df_grouped.loc[df_grouped['asset'] == asset, 'adjusted_lots'];


df_grouped = df_grouped[df_grouped['total_cost'].notna()]
df_grouped = df_grouped[df_grouped['market_value'] > 100]
tc = df_grouped['total_cost'].sum()
tmarket_value = df_grouped['market_value'].sum()

# define Seaborn color palette to use
pltt = sns.color_palette("colorblind", 8)

# plotting data on chart
plt.pie(df_grouped['market_value'], labels=df_grouped['asset'], colors=pltt, autopct='%.0f%%')

# displaying chart
plt.show()

df_grouped['profit_loss'] = round(df_grouped['market_value'] - df_grouped['total_cost'],2)
df_grouped['profit_loss_percent'] = round(100 * df_grouped['profit_loss']  / df_grouped['total_cost'],1)

df_grouped = df_grouped.sort_values('market_value', ascending=False)

print(f"Total portfolio size: {np.sum(df_grouped['market_value'])}")
print(f"Total profit/loss: {np.sum(df_grouped['profit_loss'])}")
print(f"Profit/loss: %{100 * (tmarket_value - tc) / tc}")
print(f"Total portfolio size in TRY: {round(np.sum(df_grouped['market_value'])*usdtry/1000000,2)}M")

df_trans_sorted = df_trans.sort_values(by=['Transaction Date:'])
df_trans_sorted = df_trans_sorted.reset_index(drop=True)
psd = df_trans_sorted['Transaction Date:'][0]
portfolio_start_date = psd.date()
#print(f"Portfolio start date: {portfolio_start_date}")
#df_trans_sorted.head(10)

df_trans_sorted['Transaction Date:'] = pd.to_datetime(df_trans_sorted['Transaction Date:'])
df_trans_sorted['Transaction Date:'] = df_trans_sorted['Transaction Date:'].dt.strftime('%Y-%m-%d')
dataframes = {category: df_group for category, df_group in df_trans_sorted.groupby('Type')}
asset_transactions = {}
for category, dataframe in dataframes.items():
    #print(f"DataFrame for category {category}:")
    asset_transactions[category] = dataframe.groupby('Asset:').apply(lambda x: list(zip(x['Transaction Date:'], x['Unit Cost (USD):'], x['Qty:']))).to_dict()
    #print(asset_transactions[category])

stock_tickers = list(asset_transactions['Stock - US Tech'].keys()) + list(asset_transactions['Fund - US Tech'].keys()) + list(asset_transactions['Stock - US Consumer Discretionary'].keys())
tefas_inv_tickers = list(asset_transactions['Fund - TR Stocks'].keys()) + list(asset_transactions['Fund - TR Foreign'].keys())
tefas_ret_tickers = list(asset_transactions['Fund - TR Retirement'].keys())
cash_tickers = list(asset_transactions['Cash'].keys())

# Fetch stock data
stock_data = get_stock_data(stock_tickers, portfolio_start_date.strftime('%Y-%m-%d'))
tefas_inv_data = get_tefas_inv_data(tefas_inv_tickers, portfolio_start_date.strftime('%Y-%m-%d'))
tefas_ret_data = get_tefas_ret_data(tefas_ret_tickers, portfolio_start_date.strftime('%Y-%m-%d'))
cash_data = get_tefas_ret_data(cash_tickers, portfolio_start_date.strftime('%Y-%m-%d'))

stock_data.index = pd.to_datetime(stock_data.index).strftime('%Y-%m-%d')
tefas_inv_data.index = pd.to_datetime(tefas_inv_data.index).strftime('%Y-%m-%d')
tefas_ret_data.index = pd.to_datetime(tefas_ret_data.index).strftime('%Y-%m-%d')
cash_data.index = pd.to_datetime(cash_data.index).strftime('%Y-%m-%d')

history_df = pd.concat([stock_data, tefas_inv_data, tefas_ret_data, cash_data]).groupby(level=0).sum()
history_df = history_df.sort_index()

# Replace zeros with NaN to use forward fill
history_df.replace(0, np.nan, inplace=True)

# Forward fill NaN values with the previous row's value
history_df_no_zeros = history_df.fillna(method='ffill')

combined_asset_transactions = asset_transactions['Fund - TR Retirement'].copy()
combined_asset_transactions.update(asset_transactions['Fund - TR Foreign'])
combined_asset_transactions.update(asset_transactions['Fund - TR Stocks'])
combined_asset_transactions.update(asset_transactions['Fund - US Tech'])
combined_asset_transactions.update(asset_transactions['Stock - US Tech'])
combined_asset_transactions.update(asset_transactions['Stock - US Consumer Discretionary'])
combined_asset_transactions.update(asset_transactions['Cash'])

# Calculate portfolio value
portfolio_value = calculate_portfolio_value(history_df_no_zeros, combined_asset_transactions)
portfolio_value[portfolio_value < 1 ] = 0
# Plot stacked area chart
plot_stacked_area(portfolio_value)

df_tab = df_grouped
df_tab['adjusted_lots'] = np.round(df_tab['adjusted_lots'], 2)
df_tab['price_weighted_mean'] = np.round(df_tab['price_weighted_mean'], 2)
df_tab['total_cost'] = np.round(df_tab['total_cost'], 2)
df_tab['current_price'] = np.round(df_tab['current_price'], 2)
df_tab['market_value'] = np.round(df_tab['market_value'], 2)
df_tab['profit_loss'] = np.round(df_tab['profit_loss'], 2)
df_tab['profit_loss_percent'] = np.round(df_tab['profit_loss_percent'], 2)

print(tabulate(df_tab, headers=df_grouped.columns, tablefmt="grid"))

 

