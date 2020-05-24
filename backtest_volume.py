# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 03:15:17 2020

@author: Taras
"""

#import numpy as np
import pandas as pd

import os

#from alert_new import get_symbols_BTC, is_hammer, is_doji, candle_pattern, candle_params
#from evaluate import plot_it
import indicators
from binance_endpoints import get_symbols_BTC

symbols = get_symbols_BTC()

#prefix = '_5MinuteBars.csv'
#path = 'Crypto_1MinuteBars/Jan2019-April2020/'
path = 'Crypto_1MinuteBars/'
prefix = '_1MinuteBars.csv'


MIN_PRICE = 0.00000200 # 200 Satoshi
MAX_PRICE = 0.009

# Define start date.
start_date = pd.Timestamp('2020-05-21')
#start_date = pd.Timestamp('2019-11-20') 
end_date = pd.Timestamp('2021-12-31')

start_date = start_date - pd.Timedelta('1 day')
#end_date

STOP_LOSS = 0.05
N_RED = 1


MINS_BEFORE = 15
QUOTE_AV_MIN = 3

#symbols=['DATABTC']
#symbols=['DGDBTC']

for symbol in symbols:
    
    print("Working on %s" % symbol)
    fname = path+'/'+symbol + prefix
    
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
    
    df_ohlc = df_ohlc[(df_ohlc.index >= start_date) & (df_ohlc.index <= end_date) ]
    
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
    i = MINS_BEFORE
    
    #for i in range(MINS_BEFORE, len(df_ohlc['open'])) :
    while i < len(df_ohlc['open']) :
        vol_prev_1hr = df_ohlc['volume'].iloc[i-MINS_BEFORE : i].sum()
        vol_24h = df_ohlc['quote_av'].iloc[i-1440 : i].sum()
        #candle_color = indicators.candle_params(df_ohlc['open'].iloc[i], df_ohlc['high'].iloc[i],df_ohlc['low'].iloc[i],df_ohlc['close'].iloc[i])[-1]
        last_candle = indicators.Candle(df_ohlc['open'].iloc[i], df_ohlc['high'].iloc[i],df_ohlc['low'].iloc[i],df_ohlc['close'].iloc[i])
        vol_curr = df_ohlc['volume'].iloc[i]
        q_vol = df_ohlc['quote_av'].iloc[i]
        price_change = 100*(df_ohlc['close'].iloc[i] - df_ohlc['close'].iloc[i-1])/df_ohlc['close'].iloc[i-1]
        trades = df_ohlc['trades'].iloc[i]
        taker_buy = df_ohlc['tb_quote_av'].iloc[i]
        
        # Conditions:
        vol_cond = (vol_curr > vol_prev_1hr)
        quote_col_cond = (q_vol > QUOTE_AV_MIN)
        green_candle = (last_candle.green)
        price_cond = (price_change > 2)
        min_price_cond = (df_ohlc['close'].iloc[i] > MIN_PRICE)
        shadow_to_body = last_candle.high_shadow/last_candle.body
        # Experimental feature. For now not to use it, but record the value of shadow_to_body
        shadow_cond = shadow_to_body < 0.334                
        
        if vol_cond  and quote_col_cond and price_cond and green_candle and min_price_cond : # and shadow_cond:
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
                #candle_color = candle_params(df_ohlc['open'].iloc[j], df_ohlc['high'].iloc[j],df_ohlc['low'].iloc[j],df_ohlc['close'].iloc[j])[-1]
                last_candle = indicators.Candle(df_ohlc['open'].iloc[j], df_ohlc['high'].iloc[j],df_ohlc['low'].iloc[j],df_ohlc['close'].iloc[j])
                if last_candle.red:
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
            
            filename = "backtest_volume_1min_shadow_from20May.dat"
            
            #empty = os.path.getsize(filename) == 0
            with open(filename, 'a') as f:
                empty = os.path.getsize(filename) == 0
                if empty:
                    f.write("buy_time, symbol, buy_price, profit, elapsed, went_down, vol_prev_1hr, vol_curr, q_vol, tb_quote_av, price_change,trades,vol_24h,last_shadow\n")
                f.write(f"{buy_time},{symbol},{buy_price:.8f},{profit:.2f},{elapsed:.1f},{went_down:.2f},{vol_prev_1hr:.1f},{vol_curr:.1f},{q_vol:.1f},{taker_buy:.2f},{price_change:.1f},{trades},{vol_24h:.1f},{shadow_to_body:.3f}\n")
            
            # After making a trade continue from the time we have sold the coin:
            i = j
        # Increase i to advance the while loop:
        i += 1
#
                #f.write("%s,%s,%.8f,%.2f,%.2f,%.2f,%.1f,%.1f,%.1f,%.2f,%d,%.2f\n" % (buy_time, symbol, buy_price, profit, elapsed, went_down, vol_prev_1hr, vol_curr, q_vol, price_change,trades,vol_24h) )
#    




                               
