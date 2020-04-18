# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 00:04:53 2020

@author: Taras
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 03:15:17 2020

@author: Taras
"""

import numpy as np
import pandas as pd

import os
import concurrent.futures # Needs to use Threading and MultiProcessing
#from alert_new import get_symbols_BTC, is_hammer, is_doji, candle_pattern, candle_params
from binance_endpoints import get_symbols_BTC
from evaluate import plot_it
from indicators import candle_params
import ta
#import logging
#
#logging.basicConfig(filename='volume_backtest.log', level=logging.INFO,
#                    format='%(levelname)s:%(message)s')

#symbols = get_symbols_BTC()

#prefix = '_5MinuteBars.csv'
prefix = '_1MinuteBars.csv'
path = ''

# Define start date.
start_date = pd.Timestamp('2020-01-02')
#start_date = pd.Timestamp('2019-11-20') 

start_date = start_date - pd.Timedelta('1 day')
#end_date

PERCENT = 0.75

#symbols=['DATABTC']
#symbols=['DGDBTC']

#for symbol in symbols:
def backtest(symbol):    
    #logging.info("Working on %s" % symbol)
    print(("Working on %s" % symbol))
    fname = path + symbol + prefix
    
    #m_15 = 3
    #h_1 = m_15*4
    #d_1 = h_1*24
    try:
        df_ohlc = pd.read_csv(fname, index_col=0, parse_dates=True)
    except FileNotFoundError:
        #logging.error("File not found: ", fname)
        print(("File not found: ", fname))
        return []
        #continue
    
    # Put start date 1 day earlier to make  technical indicators work from the beginning of the analysis.
    #start_date = start_date - pd.Timedelta('1 day')
    
    df_ohlc = df_ohlc[df_ohlc.index >= start_date]
    
    if len(df_ohlc) == 0: return []
    
    BBands = ta.volatility.BollingerBands(df_ohlc['close'])  
    lower = BBands.bollinger_lband()
    upper = BBands.bollinger_hband()
    
    count = 0
    
    deals = {}
    
    for i in range( len(df_ohlc['open']) ) :
        #price_curr = df_ohlc['low'].iloc[i]
        min_price = df_ohlc['low'].iloc[i]
        max_price = df_ohlc['high'].iloc[i]
        lbb = lower.iloc[i]
        min_pc = 100*(lbb - min_price)/min_price        
        #ubb = upper.iloc[i]
        if min_pc > PERCENT :
            count += 1
            buy = lbb*(1-0.01*PERCENT)
            take_profit = buy*(1+0.01*PERCENT)
            stop_loss = buy*(1-0.01*PERCENT)
            deals[count] = {'time':df_ohlc.index[i], 'symbol':symbol, 'buy':buy, 'take_profit':take_profit, 'stop_loss':stop_loss}
            if min_price <= stop_loss:                
                print(df_ohlc.index[i], symbol, "INSTANT LOSS", 0, min_pc)
                del(deals[count])
            continue
        
        # evaluate:
        if len(deals) > 0:
            for item in dict(deals):
                if min_price <= deals[item]['stop_loss']:
                    elapsed = df_ohlc.index[i] - deals[item]['time']
                    print(df_ohlc.index[i], symbol, "LOSS", elapsed/pd.Timedelta('1min'))
                    del(deals[item])
                elif max_price >= deals[item]['take_profit']:
                    elapsed = df_ohlc.index[i] - deals[item]['time']
                    print(df_ohlc.index[i], symbol, "PROFIT", elapsed/pd.Timedelta('1min'))
                    del(deals[item])
                
            
            #filename = "backtest_volume_strategy_1min_1red_1-3April.dat"
            
            #empty = os.path.getsize(filename) == 0
#            with open(filename, 'a') as f:
#                empty = os.path.getsize(filename) == 0
#                if empty:
#                    f.write("buy_time, symbol, buy_price, profit, elapsed, went_down, vol_prev_1hr, vol_curr, q_vol, price_change,trades,vol_24h\n")
#                f.write(f"{buy_time},{symbol},{buy_price:.8f},{profit:.2f},{elapsed:.1f},{went_down:.2f},{vol_prev_1hr:.1f},{vol_curr:.1f},{q_vol:.1f},{price_change:.1f},{trades},{vol_24h:.1f}\n")
#                #f.write("%s,%s,%.8f,%.2f,%.2f,%.2f,%.1f,%.1f,%.1f,%.2f,%d,%.2f\n" % (buy_time, symbol, buy_price, profit, elapsed, went_down, vol_prev_1hr, vol_curr, q_vol, price_change,trades,vol_24h) )
#    
#
#with concurrent.futures.ProcessPoolExecutor() as executor:
#    executor.map(backtest, symbols) 


backtest('BTCUSDT')
                               