# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:06:26 2020

@author: Taras
"""

#import numpy as np
import pandas as pd

from backtest_v4 import evaluate

c1m_data = pd.read_csv('alerts.dat')#, parse_dates=True, dayfirst=True)
c1m_data.loc[:, 'date'] = pd.to_datetime(c1m_data.loc[:, 'date'], dayfirst=True)

c1m_data.sort_values(['coin', 'date'], inplace=True)
c1m_data.drop(379, inplace=True)
c1m_data.drop(380, inplace=True)

# Convert to German timezone (our server is there)
c1m_data.loc[:, 'date'] = c1m_data.loc[:, 'date'] - pd.Timedelta('4h')

with open('c1m_backtest_official.dat', 'w') as f:
    f.write('timestamp,coin,price,label,profit,elapsed,min_price,max_price,profit_14,elapsed_14,min_price_14,max_price_14\n')

for coin in c1m_data['coin'].unique():
    cond = c1m_data['coin'] == coin
    fname = f"{coin}BTC_1MinuteBars.csv"
    print(f"Evaluating {coin} ...")
    df_ohlc_1m = pd.read_csv(fname, index_col=0, parse_dates=True)
    for timestamp, coin, price, label in zip(c1m_data['date'][cond],c1m_data['coin'][cond],c1m_data['price'][cond],c1m_data['label'][cond]):
        try: 
            price = float(price)
        except ValueError:
            continue
        profit, elapsed, min_price, max_price = evaluate(timestamp,price,df_ohlc_1m)
        profit_1_4, elapsed_1_4, min_price_1_4, max_price_1_4 = evaluate(timestamp,price,df_ohlc_1m,take_profit=0.01,stop_loss=0.04)
        with open('c1m_backtest_official.dat', 'a') as f:
            f.write(f"{timestamp},{coin},{price:.8f},{label}")
            f.write(f"{profit:.2f},{elapsed:.1f},{min_price:.2f},{max_price:.2f}")
            f.write(f"{profit_1_4:.2f},{elapsed_1_4:.1f},{min_price_1_4:.2f},{max_price_1_4:.2f}")
            f.write('\n')
