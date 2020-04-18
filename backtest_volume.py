# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 03:15:17 2020

@author: Taras
"""

import numpy as np
import pandas as pd

import os

from alert_new import get_symbols_BTC, is_hammer, is_doji, candle_pattern, candle_params
from evaluate import plot_it

symbols = get_symbols_BTC()

#prefix = '_5MinuteBars.csv'
prefix = '_1MinuteBars.csv'


MIN_PRICE = 0.00000200 # 200 Satoshi
MAX_PRICE = 0.009

# Define start date.
start_date = pd.Timestamp('2020-01-01')
#start_date = pd.Timestamp('2019-11-20') 

start_date = start_date - pd.Timedelta('1 day')
#end_date

STOP_LOSS = 0.05
N_RED = 2


MINS_BEFORE = 15
QUOTE_AV_MIN = 3

#symbols=['DATABTC']
#symbols=['DGDBTC']

for symbol in symbols:
    
    print("Working on %s" % symbol)
    fname = symbol + prefix
    
    #m_15 = 3
    #h_1 = m_15*4
    #d_1 = h_1*24
    try:
        df_ohlc = pd.read_csv(fname, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("File not found: ", fname)
        continue
    
    # Put start date 1 day earlier to make  technical indicators work from the beginning of the analysis.
    #start_date = start_date - pd.Timedelta('1 day')
    
    df_ohlc = df_ohlc[df_ohlc.index >= start_date]
    
    if len(df_ohlc) == 0: continue
    
#    df_open_5m = df_ohlc['open'].resample('5min').first()
#    df_high_5m = df_ohlc['high'].resample('5min').max()
#    df_low_5m = df_ohlc['low'].resample('5min').min()
#    df_close_5m = df_ohlc['close'].resample('5min').last()
#    df_quote_av_5m = df_ohlc['quote_av'].resample('5min').sum()
#    df_volume_5m = df_ohlc['volume'].resample('5min').sum()
#    df_trades_5m = df_ohlc['trades'].resample('5min').sum()
#    
#    df_ohlc_5m = pd.concat([df_open_5m,df_high_5m, df_low_5m, df_close_5m, df_quote_av_5m, df_volume_5m, df_trades_5m ], axis=1, keys=['open', 'high', 'low', 'close', 'quote_av', 'volume', 'trades'])
#    
#    df_ohlc = df_ohlc_5m 
    
    #df_ohlc = df_tmp[['open', 'high', 'low', 'close', 'quote_av', 'volume']] #    
    
    count = 0
    
    
    for i in range(MINS_BEFORE, len(df_ohlc['open'])) :
        vol_prev_1hr = df_ohlc['volume'].iloc[i-MINS_BEFORE : i].sum()
        vol_24h = df_ohlc['quote_av'].iloc[i-1440 : i].sum()
        candle_color = candle_params(df_ohlc['open'].iloc[i], df_ohlc['high'].iloc[i],df_ohlc['low'].iloc[i],df_ohlc['close'].iloc[i])[-1]
        vol_curr = df_ohlc['volume'].iloc[i]
        q_vol = df_ohlc['quote_av'].iloc[i]
        price_change = 100*(df_ohlc['close'].iloc[i] - df_ohlc['close'].iloc[i-1])/df_ohlc['close'].iloc[i-1]
        trades = df_ohlc['trades'].iloc[i]
        
        # Conditions:
        vol_cond = (vol_curr > vol_prev_1hr)
        quote_col_cond = (q_vol > QUOTE_AV_MIN)
        green_candle = (candle_color == 'green')
        price_cond = (price_change > 2)
        min_price_cond = (df_ohlc['close'].iloc[i] > MIN_PRICE)                
        
        if vol_cond  and quote_col_cond and price_cond and green_candle and min_price_cond:
            try :
                tmp = df_ohlc['open'].iloc[i+1]
            except IndexError:
                continue
            
            buy_price = df_ohlc['open'].iloc[i+1]
            min_price = df_ohlc['low'].iloc[i+1]
            buy_time = df_ohlc.index[i+1]            
            j = i+1
            red_count = 0
#            while candle_color == 'green':
            while red_count < N_RED:                
                candle_color = candle_params(df_ohlc['open'].iloc[j], df_ohlc['high'].iloc[j],df_ohlc['low'].iloc[j],df_ohlc['close'].iloc[j])[-1]
                if candle_color == 'red':
                    sell_price = df_ohlc['close'].iloc[j]
                    red_count += 1
                    profit = 100*(sell_price - buy_price)/buy_price
                    if profit < 0 : 
                        sell_time = df_ohlc.index[j]
                        break                        
                sell_time = df_ohlc.index[j]
                j += 1                           
                        
            
                        
            profit = 100*(sell_price - buy_price)/buy_price
            elapsed = (sell_time - buy_time)/pd.Timedelta('1min')
            went_down = 100*(min_price - buy_price)/buy_price
                   
            
            #if np.abs(profit) > 1: 
            print(buy_time, symbol, buy_price, profit, elapsed, went_down, vol_prev_1hr, vol_curr, q_vol, price_change)
            print("24h vol: ", vol_24h, "N trades: ", trades)
            
            count += 1
            #plot_it(df_ohlc, buy_time, sell_time, buy_price, sell_price, interval='1min', savefile="%s_%d" % (symbol, count) )
            
            filename = "backtest_volume_strategy_1min_2red.dat"
            
            #empty = os.path.getsize(filename) == 0
            with open(filename, 'a') as f:
                empty = os.path.getsize(filename) == 0
                if empty:
                    f.write("buy_time, symbol, buy_price, profit, elapsed, went_down, vol_prev_1hr, vol_curr, q_vol, price_change,trades,vol_24h\n")
                f.write(f"{buy_time},{symbol},{buy_price:.8f},{profit:.2f},{elapsed:.1f},{went_down:.2f},{vol_prev_1hr:.1f},{vol_curr:.1f},{q_vol:.1f},{price_change:.1f},{trades},{vol_24h:.1f}\n")
                #f.write("%s,%s,%.8f,%.2f,%.2f,%.2f,%.1f,%.1f,%.1f,%.2f,%d,%.2f\n" % (buy_time, symbol, buy_price, profit, elapsed, went_down, vol_prev_1hr, vol_curr, q_vol, price_change,trades,vol_24h) )
#    




                               