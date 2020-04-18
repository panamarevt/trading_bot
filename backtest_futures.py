# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 00:16:18 2020

@author: Taras
"""

'''In this backtest we try to simulate the past months activities as close to reality as possible...'''

# import all necessary modules
#from datetime import datetime
#from pandas import DataFrame as df
import numpy as np
import pandas as pd
#import os

import concurrent.futures

#import time
#from binance_api import Binance

# Techincal indicators library by bablofil
# https://bablofil.ru/raschet-indikatorov-SMMA-EWMA-MMA-RMA-RSI-STOCH-STOCHRSI/
# https://bablofil.ru/bot-dlya-binance-s-indikatorami/
#import bablofil_ta as ta

# Technical indicators from TA library
# https://technical-analysis-library-in-python.readthedocs.io/en/latest/
# https://github.com/bukosabino/ta
import ta

from alert_new import get_symbols_BTC, is_hammer, is_doji, candle_pattern, candle_params




def save_signal_backtest(t_curr, coin, price, pattern, signal_origin):
    with open('buy_signals_backtest_4try.dat', 'a') as f:
        f.write("%s   %5s    %.8f   %12s  %5s\n" % (t_curr, coin, price, pattern, signal_origin))

def save_signal_backtest_csv(t_curr, coin, price, pattern, signal_origin, ranging, lower, stoch_k_15m, stoch_k_5m, last_candle, vol_pm_inc, profit, elapsed, min_price, max_price,profit_p17, elapsed_p17, min_price_p17, max_price_p17, profit_s2, elapsed_s2, min_price_s2, max_price_s2, profit_s12_p14, elapsed_s12_p14, min_price_s12_p14, max_price_s12_p14, profit_s12_p12, elapsed_s12_p12, min_price_s12_p12, max_price_s12_p12, profit_p1, elapsed_p1, min_price_p1, max_price_p1):
    #with open('buy_signals_backtest_detail_Jan-Mar2020_new.dat', 'a') as f:
    with open('buy_signals_backtest_futures_Apr_2020_stoch_10.dat', 'a') as f:
        f.write("%s,%s,%.8f,%s,%s,%2f,%.8f,%3f,%s,%s,%.3f,%.3f,%.1f,%.2f,%.2f,%.3f,%.1f,%.2f,%.2f,%.3f,%.1f,%.2f,%.2f,%.3f,%.1f,%.2f,%.2f,%.3f,%.1f,%.2f,%.2f,%.3f,%.1f,%.2f,%.2f\n" % (t_curr, coin, price, pattern, signal_origin, \
                                                                               ranging, lower, stoch_k_15m, stoch_k_5m, last_candle, \
                                                                               vol_pm_inc,\
                                                                               profit, elapsed, min_price, max_price,\
                                                                               profit_p17, elapsed_p17, min_price_p17, max_price_p17,\
                                                                               profit_s2, elapsed_s2, min_price_s2, max_price_s2,\
                                                                               profit_s12_p14, elapsed_s12_p14, min_price_s12_p14, max_price_s12_p14,\
                                                                               profit_s12_p12, elapsed_s12_p12, min_price_s12_p12, max_price_s12_p12, \
                                                                               profit_p1, elapsed_p1, min_price_p1, max_price_p1) )



def evaluate(start_time, buy_price, df_ohlc_1m, take_profit = 0.015, stop_loss = 0.03):
    #stop_loss /= 1.2
    time_cond = (df_ohlc_1m.index > start_time)
    loss_cond = ( df_ohlc_1m['low'] <= buy_price*(1-stop_loss) )
    profit_cond = ( df_ohlc_1m['high'] >= buy_price*(1+take_profit) )
    try:
        time_to_loss = df_ohlc_1m[time_cond & loss_cond]['low'].index[0]
    except IndexError:
        time_to_loss = None
    try:
        time_to_profit = df_ohlc_1m[time_cond & profit_cond]['high'].index[0]
    except IndexError:
        time_to_profit = None
    
    if (time_to_loss == None) and (time_to_profit == None):
        return None, None, None, None
    elif (time_to_loss == None):
        profit = 100*take_profit
        elapsed = time_to_profit - start_time
        min_price = df_ohlc_1m['low'][time_cond & (df_ohlc_1m.index <= time_to_profit)].min()
        high = df_ohlc_1m['high'][time_cond & (df_ohlc_1m.index >= time_to_profit - pd.Timedelta('1m') ) ]
        id_max = high[(high.shift(1) < high) & (high.shift(-1) < high)]
        #max_price = df_ohlc_1m['high'][time_cond & (df_ohlc_1m.index >= time_to_profit) ][0]
        max_price = id_max[0]
        went_down = 100*(min_price - buy_price)/buy_price
        went_up  = 100*(max_price-buy_price)/buy_price
        print("Went down by:", np.round(went_down, 2), "Elapsed: ", elapsed/pd.Timedelta('1m'), \
              "Max price after: ", np.round(went_up, 2) )
        return profit, elapsed/pd.Timedelta('1m'), min_price, max_price
    elif (time_to_profit == None):
        profit = -100*stop_loss*1.2
        elapsed = time_to_loss - start_time
        min_price = buy_price*(1-stop_loss)
        max_price = df_ohlc_1m['high'][time_cond & (df_ohlc_1m.index <= time_to_loss )].max()
        went_up  = 100*(max_price-buy_price)/buy_price
        went_down = 100*(min_price - buy_price)/buy_price
        return profit, elapsed/pd.Timedelta('1m'), went_down, went_up
    
    if time_to_profit < time_to_loss:
        print("Start: ", start_time, "Profit", time_to_profit)
        profit = 100*take_profit
        elapsed = time_to_profit - start_time
        min_price = df_ohlc_1m['low'][time_cond & (df_ohlc_1m.index <= time_to_profit)].min()
        #id_max = df_ohlc_1m['high'].idxmax()
        #id_max = df_ohlc_1m['high'][(df_ohlc_1m['high'].shift(1) < df_ohlc_1m['high']) & (df_ohlc_1m['high'].shift(-1) < df_ohlc_1m['high'])]
        high = df_ohlc_1m['high'][time_cond & (df_ohlc_1m.index >= time_to_profit - pd.Timedelta('1m') ) ]
        id_max = high[(high.shift(1) < high) & (high.shift(-1) < high)]
        #max_price = df_ohlc_1m['high'][time_cond & (df_ohlc_1m.index >= time_to_profit) ][0]
        max_price = id_max[0]
        went_down = 100*(min_price - buy_price)/buy_price
        went_up  = 100*(max_price-buy_price)/buy_price
        print("Went down by:", np.round(went_down, 2), "Elapsed: ", elapsed/pd.Timedelta('1m'), \
              "Max price after: ", np.round(went_up, 2) )
    else: 
        print("Start: ", start_time, "Loss", time_to_loss)
        profit = -100*stop_loss*1.2
        elapsed = time_to_loss - start_time
        min_price = buy_price*(1-stop_loss)
        max_price = df_ohlc_1m['high'][time_cond & (df_ohlc_1m.index <= time_to_loss)].max()
        went_down = 100*(min_price - buy_price)/buy_price
        went_up  = 100*(max_price-buy_price)/buy_price
        print("Went up before loss", np.round(went_up, 2) )
    return profit, elapsed/pd.Timedelta('1m'), went_down, went_up


def backtest(symbol, start_date, interval='15min'):
    #for symbol in symbols:
    
    print("Working on %s" % symbol)
    fname = symbol + prefix
    
    #m_15 = 3
    #h_1 = m_15*4
    #d_1 = h_1*24
    try:
        df_tmp = pd.read_csv(fname, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("File not found: ", fname)
        #continue
        return 0
    
    # Put start date 1 day earlier to make  technical indicators work from the beginning of the analysis.
    #start_date = start_date - pd.Timedelta('1 day')
    
    df_tmp = df_tmp[df_tmp.index >= start_date]
    
    if pd.Timedelta(interval) > pd.Timedelta('1min'): 
    
        try:
            df_open = df_tmp['open'].resample(interval).first()
            df_high = df_tmp['high'].resample(interval).max()
            df_low = df_tmp['low'].resample(interval).min()
            df_close = df_tmp['close'].resample(interval).last()
            df_volume = df_tmp['quote_av'].resample(interval).sum()
        except Exception as e:
            print("%s - problem!" % symbol)
            print(e)
            #continue
            return 0        
        try:
            df_ohlc = pd.concat([df_open,df_high, df_low, df_close, df_volume ], axis=1, keys=['open', 'high', 'low', 'close', 'volume'])
        except Exception as e:
            print("%s - problem!" % symbol)
            print(e)
            #continue    
            return 0
    else:
        df_ohlc = df_tmp[['open', 'high', 'low', 'close', 'quote_av', 'volume']] 
        
    df_ohlc_1m = df_tmp[['open', 'high', 'low', 'close', 'quote_av']] #
    
    df_open_5m = df_tmp['open'].resample('5min').first()
    df_high_5m = df_tmp['high'].resample('5min').max()
    df_low_5m = df_tmp['low'].resample('5min').min()
    df_close_5m = df_tmp['close'].resample('5min').last()
    
    df_ohlc_5m = pd.concat([df_open_5m,df_high_5m, df_low_5m, df_close_5m ], axis=1, keys=['open', 'high', 'low', 'close', 'volume'])
    
    
    
    BBands = ta.volatility.BollingerBands(df_ohlc['close'])
    #ohlc.values
    middle = BBands.bollinger_mavg()
    upper = BBands.bollinger_hband()
    lower = BBands.bollinger_lband()
    width = BBands.bollinger_wband()
    
    ranging= width/lower *100 # in %     
    
    # Stichastic RSI for 15 min cancles
    RSI = ta.momentum.RSIIndicator(df_ohlc['close']).rsi()
    stoch_RSI = ta.momentum.StochasticOscillator(RSI,RSI,RSI)
    
    #fastk; convert to numpy arrays to allow negative index -- DON't do that!
    fastk =  stoch_RSI.stoch_signal() 
    #slow_d is 3-day SMA of fast_k
    slowd = ta.volatility.bollinger_mavg(fastk, n=3) 
    
    # Stichastic RSI for 5 min cancles
    RSI_5m = ta.momentum.RSIIndicator(df_ohlc_5m['close']).rsi()
    stoch_RSI_5m = ta.momentum.StochasticOscillator(RSI_5m,RSI_5m,RSI_5m)    
    #fastk; convert to numpy arrays to allow negative index -- DON't do that!
    fastk_5m =  stoch_RSI_5m.stoch_signal() 
    #slow_d is 3-day SMA of fast_k
    slowd_5m = ta.volatility.bollinger_mavg(fastk_5m, n=3)     
    
    for i in range(96, len(df_ohlc) ):
        #if (df_ohlc['volume'].iloc[i-96:i].sum() > MIN_24h_VOLUME) and (MIN_PRICE < df_ohlc['close'].iloc[i] < MAX_PRICE)  :
        # Skip price and volume conditions for BTCUSDT Futures. Quick and dirty solution! :-) 
        if True:    
            # Check Bollinger Bands width in %
            if ranging.iloc[i] > RANGING_COND:
            
                lower_BB_signal = ((df_ohlc['close'].iloc[i-1] < lower.iloc[i-1] or df_ohlc['low'].iloc[i-1] < lower.iloc[i-1]) \
                or (df_ohlc['close'].iloc[i-2] < lower.iloc[i-2] or df_ohlc['low'].iloc[i-2] < lower.iloc[i-2]) )
        
                middle_BB_signal = ( (df_ohlc['close'].iloc[i-2] < middle.iloc[i-2] and df_ohlc['open'].iloc[i-2] > middle.iloc[i-2]) \
                or (df_ohlc['low'].iloc[i-2] < middle.iloc[i-2] and df_ohlc['open'].iloc[i-2] > middle.iloc[i-2] and df_ohlc['close'].iloc[i-2] > middle.iloc[i-2]) )        
            
                middle_BB_signal_2 = ( (df_ohlc['close'].iloc[i-1] < middle.iloc[i-1] and df_ohlc['open'].iloc[i-1] > middle.iloc[i-1]) \
                or (df_ohlc['low'].iloc[i-1] < middle.iloc[i-1] and df_ohlc['open'].iloc[i-1] > middle.iloc[i-1] and df_ohlc['close'].iloc[i-1] > middle.iloc[i-1]) )    
        
                #if lower_BB_signal or middle_BB_signal or middle_BB_signal_2:
                if lower_BB_signal or middle_BB_signal_2:
                    # Check condition for stoch RSI
                    #stochRSI_condition = fastk.iloc[i-1] < 20 and slowd.iloc[i-1] < 20
                    stochRSI_condition = True # quick and durty solution to check consistency with real time signals
                    
                    if stochRSI_condition:
                        #start_j= 3*i
                        #end_j = 3*i+3
#                        start_j = 0
#                        #end_j = 3
#                        end_j = 15
#                        for j in range(start_j, end_j):
#                            mins = j
                        time_curr_1m = df_ohlc.index[i-1]# + pd.Timedelta('%d min' % mins)
#                        try:
#                            #close_curr = pd.concat([pd.Series(df_ohlc['close'].iloc[:i-1]), pd.Series(df_ohlc_1m[df_ohlc_1m.index==time_curr_1m]['open'][0])])
#                            ### !!! Change open price on 1m candle to close price for better consistency with RSI calculation
#                            #close_curr = pd.concat([pd.Series(df_ohlc['close'].iloc[:i-1]), pd.Series(df_ohlc_1m[df_ohlc_1m.index==time_curr_1m]['close'][0])])
#                            ### !!! Change iloc[:i-1] to iloc[:i]
#                            close_curr = pd.concat([pd.Series(df_ohlc['close'].iloc[:i]), pd.Series(df_ohlc_1m[df_ohlc_1m.index==time_curr_1m]['close'][0])])
#                        except:
#                            print("No more 1 min data at this point")
#                            break
                        #close_curr = pd.concat([pd.Series(df_ohlc['close'].iloc[:i-1]), pd.Series(df_ohlc_1m['open'].iloc[j])])
                        #Get current srochRSI (at open price of the current candle)
                        
                        close_curr = df_ohlc['close']
                        
                        RSI_curr = ta.momentum.RSIIndicator(close_curr).rsi()
                        stoch_RSI_curr = ta.momentum.StochasticOscillator(RSI_curr,RSI_curr,RSI_curr)
                        fastk_curr = stoch_RSI_curr.stoch_signal()
                        slowd_curr = ta.volatility.bollinger_mavg(fastk_curr, n=3)
                        
                        final_stoch_condition = ( (fastk_curr.iloc[-1] > slowd_curr.iloc[-1]) and (fastk_curr.iloc[-1] > fastk_curr.iloc[-2]) and (slowd_curr.iloc[-1] > slowd_curr.iloc[-2]) and (10 < fastk_curr.iloc[-1] < 30) )  

                        #price_curr = df_ohlc_1m[df_ohlc_1m.index==time_curr_1m]['low'][0]
                        price_curr = df_ohlc_1m[df_ohlc_1m.index==time_curr_1m]['close'][0]

                        if lower_BB_signal:
                            dist_to_BB = 100*(middle.iloc[i-1] - price_curr)/price_curr
                        else:
                            dist_to_BB = 100*(upper.iloc[i-1] - price_curr)/price_curr
                        price_cond = (dist_to_BB > 1.5)                        
                        
                        #stochRSI_signal = (fastk.iloc[i-1] > slowd.iloc[i-1]) and (fastk.iloc[i-1] > fastk.iloc[i-2]) and (slowd.iloc[i-1] > slowd.iloc[i-2]) and (10 < fastk.iloc[i-1] < 30)
                        stochRSI_signal = (fastk.iloc[i-1] > slowd.iloc[i-1]) and (fastk.iloc[i-1] > fastk.iloc[i-2]) and (slowd.iloc[i-1] > slowd.iloc[i-2]) and  (fastk.iloc[i-1] > 10)
                        #if final_stoch_condition  and price_cond :
                        #if final_stoch_condition: 
                        if stochRSI_signal :
                            # Check candle patterns:
                            last = [df_ohlc['open'].iloc[i-1], df_ohlc['high'].iloc[i-1], df_ohlc['low'].iloc[i-1], df_ohlc['close'].iloc[i-1]]
                            before_last = [df_ohlc['open'].iloc[i-2], df_ohlc['high'].iloc[i-2], df_ohlc['low'].iloc[i-2], df_ohlc['close'].iloc[i-2]]
                            before_before_last = [df_ohlc['open'].iloc[i-3], df_ohlc['high'].iloc[i-3], df_ohlc['low'].iloc[i-3], df_ohlc['close'].iloc[i-3]]
                            last_candle_color = candle_params(*last)[-1]
                            
                            #Compute increase in volume per minute
#                            vol_pm_prev = df_ohlc['volume'].iloc[i-1]/15
#                            vol_pm_curr = df_ohlc_1m['quote_av'][df_ohlc.index[i] : time_curr_1m].sum()/(j+1)
#                            vol_pm_inc = vol_pm_curr/vol_pm_prev
                            vol_pm_inc = df_ohlc['volume'].iloc[i-1]/df_ohlc['volume'].iloc[i-2]
                            
                            
                            pattern = candle_pattern(last, before_last)
                            # If no pattern found then check one more candle
                            if pattern == 'no': pattern = candle_pattern(before_last, before_before_last)
                            

                            
                            if lower_BB_signal : 
                                signal_origin = 'lower'
                                #take_profit = lower.iloc[i-1] + 1/3* (middle.iloc[i-1] - lower.iloc[i-1]) # Absolute value of take profit
                                 # percentage
                                
                                                                           
                            else:
                                signal_origin = 'upper'
                                #take_profit = middle.iloc[i-1] + 1/3* (upper.iloc[i-1] - middle.iloc[i-1])
                                #print(f'take profit = {take_profit:.5f}')
                            
                            maker = 0.0002
                            taker = 0.0004
                            #fee = maker + taker
                            fee = 2*maker                            
                            #take_profit = (take_profit - price_curr)/price_curr
                            take_profit = 0.0025
                            if take_profit < 0.001 : continue
                                                                            
                            stop_loss = (2*take_profit - 2*fee) # Get this expression under the condition that (net profit) / (net loss) = 1/2
                            print(f'take profit = {take_profit:.5f}')
                            print(f'stop loss = {stop_loss:.5f}')
                            
                            #print(symbol, df_ohlc.index[i], df_ohlc_1m.index[j])
                            print(symbol, df_ohlc.index[i], df_ohlc_1m.index[df_ohlc_1m.index == time_curr_1m][0] )
                            #save_signal_backtest(df_ohlc.index[i], symbol[:-3], df_ohlc['open'].iloc[i], pattern)
                            #price_curr = df_ohlc_1m[df_ohlc_1m.index==time_curr_1m]['open'][0]
                            
                            #Compute current stoch RSI on 5m candles
                            close_curr_5m = pd.concat([pd.Series(df_ohlc_5m['close'][df_ohlc_5m.index<time_curr_1m]), pd.Series(df_ohlc_1m[df_ohlc_1m.index==time_curr_1m]['close'][0])])
                            
                            RSI_curr_5m = ta.momentum.RSIIndicator(close_curr_5m).rsi()
                            stoch_RSI_curr_5m = ta.momentum.StochasticOscillator(RSI_curr_5m,RSI_curr_5m,RSI_curr_5m)
                            fastk_curr_5m = stoch_RSI_curr_5m.stoch_signal()
                            slowd_curr_5m = ta.volatility.bollinger_mavg(fastk_curr_5m, n=3)    

                            
#                            stoch_k_5m = fastk_curr_5m.iloc[-1]
#                            stoch_k_15m = fastk_curr.iloc[-1]
                            stoch_k_15m = fastk.iloc[i-1]
                            #save_signal_backtest(time_curr_1m, symbol[:-3], price_curr, pattern, signal_origin)
                            #save_signal_backtest_csv(time_curr_1m, symbol[:-3], price_curr, pattern, signal_origin, ranging.iloc[i], lower.iloc[i], stoch_k_15m, stoch_k_5m)
                            
                            #final_stoch_condition_5m = ( (fastk_curr_5m.iloc[-1] > slowd_curr_5m.iloc[-1]) and (fastk_curr_5m.iloc[-1] > fastk_curr_5m.iloc[-2]) and (slowd_curr_5m.iloc[-1] > slowd_curr_5m.iloc[-2]) )# and (fastk_curr_5m.iloc[-1] > 20) )
                            final_stoch_condition_5m = (fastk_curr_5m.iloc[-1] < 20) or (fastk_curr_5m.iloc[-1] > fastk_curr_5m.iloc[-2])
                            print("time:", time_curr_1m, "price", price_curr)
                            
                            profit, elapsed, min_price, max_price = evaluate(time_curr_1m, price_curr, df_ohlc_1m, take_profit=take_profit, stop_loss=stop_loss)
                            profit15, elapsed15, min_price15, max_price15 = evaluate(time_curr_1m, price_curr, df_ohlc_1m, take_profit=0.002, stop_loss=2*0.002 - 2*fee)
                            profit20, elapsed20, min_price20, max_price20 = evaluate(time_curr_1m, price_curr, df_ohlc_1m, take_profit=0.0021, stop_loss=0.0015)
                       #     profit_p17, elapsed_p17, min_price_p17, max_price_p17 = evaluate(time_curr_1m, price_curr, df_ohlc_1m, take_profit=0.017)
                       #     profit_s2, elapsed_s2, min_price_s2, max_price_s2 = evaluate(time_curr_1m, price_curr, df_ohlc_1m, stop_loss=0.02)
                            profit_s12_p14, elapsed_s12_p14, min_price_s12_p14, max_price_s12_p14 = evaluate(time_curr_1m, price_curr, df_ohlc_1m, take_profit=0.0025, stop_loss=0.0019)
                            profit_s12_p12, elapsed_s12_p12, min_price_s12_p12, max_price_s12_p12 = evaluate(time_curr_1m, price_curr, df_ohlc_1m, take_profit=0.003, stop_loss=0.0024)
                            profit_p1, elapsed_p1, min_price_p1, max_price_p1 = evaluate(time_curr_1m, price_curr, df_ohlc_1m, take_profit=0.0033+fee, stop_loss=2*0.0035-fee)
                            #elapsed = elapsed/pd.Timedelta("1min")
#                            if profit == None: 
#                                break
                            #else:
                            
                                #save_signal_backtest(time_curr_1m, symbol[:-3], price_curr, pattern, signal_origin)
                            save_signal_backtest_csv(time_curr_1m, symbol[:-3], price_curr, pattern, signal_origin, \
                                                         ranging.iloc[i], lower.iloc[i], stoch_k_15m, final_stoch_condition_5m, last_candle_color,\
                                                         vol_pm_inc, profit, elapsed, min_price, max_price, \
                                                         profit15, elapsed15, min_price15, max_price15,\
                                                         profit20, elapsed20, min_price20, max_price20,\
                                                         profit_s12_p14, elapsed_s12_p14, min_price_s12_p14, max_price_s12_p14,\
                                                         profit_s12_p12, elapsed_s12_p12, min_price_s12_p12, max_price_s12_p12, \
                                                         profit_p1, elapsed_p1, min_price_p1, max_price_p1)
                                
                                #break


if __name__ == '__main__':
    symbols = get_symbols_BTC()
    
    # Strategy: C1M
    #path = 'D:\PROJECTS\Binance\CryptoData'
    
    #prefix = '_5MinuteBars.csv'
    prefix = '_1MinuteBars.csv'
    
    NDAYS = 365
    MIN_PRICE = 0.00000200 # 200 Satoshi
    MAX_PRICE = 0.009
    MIN_24h_VOLUME = 150 #BTC\
    
    # Define start date.
    start_date = pd.Timestamp('2020-04-01')
    #start_date = pd.Timestamp('2019-01-01') 
    
    start_date = start_date - pd.Timedelta('1 day')
    #end_date
    # Condition of minimum Bolliger Bands width
    #RANGING_COND = 4.0
    RANGING_COND = 0.2
    #symbol = 'XTZBTC'
    
    #symbols = ['PERLBTC']
    
    #for symbol in symbols[66:]:
#    i = 0
#    while i <= len(symbols) - 8:
#        
#        with concurrent.futures.ProcessPoolExecutor() as executor:
#            executor.map(backtest, symbols[i:i+8])
#        i += 8    
    
    # Backtest strategy on multiple symbols in parallel:
#    i = 0
#    while i < len(symbols):
#
#        with concurrent.futures.ProcessPoolExecutor() as executor:
#            #executor.map(binanceBarExtractor, symbols[i:i+8])
#            if i + 8 <= len(symbols):
#                executor.map(backtest, symbols[i:i+8])
#            else:
#                executor.map(backtest, symbols[i:])
#        i += 8

    backtest('BTCUSDT', start_date = start_date, interval='1min')
