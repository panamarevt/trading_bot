# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:56:57 2020

@author: Taras
"""

import pandas as pd
#from pandas import DataFrame as df

import numpy as np


def load_data(filename):
    data = pd.read_csv(filename,index_col=0, parse_dates=True)#, keys=['timestamp', 'coin', 'buy_price', 'pattern', 'origin'])
    #data = np.loadtxt(filename, dtype='str')
    #data = df(data, columns=['timestamp', 'coin', 'buy_price', 'pattern', 'origin'])
    return data

#def save_eval(buy_time, coin, buy_price, pattern, origin, ranging, lower_BB, profit,  went_down_by, went_up_after, elapsed):
#    with open('backtest_evaluate_1month.dat', 'a') as f:
#        f.write("%s,%s,%.8f,%s,%s,%.2f,%.8f,%.2f,%.2f,%.2f,%.1f\n" % (buy_time, coin, buy_price, pattern, origin, ranging, lower_BB, profit, went_down_by, went_up_after, elapsed))

def save_eval(buy_time, coin, buy_price, pattern, origin, ranging, lower_BB, stoch_k_15m, stoch_k_5m, profit,  went_down_by, went_up_after, elapsed):
    with open('backtest_evaluate_1year_15-5min_2.dat', 'a') as f:
        f.write("%s,%s,%.8f,%s,%s,%.2f,%.8f,%.3f,%.3f,%.2f,%.2f,%.2f,%.1f\n" % (buy_time, coin, buy_price, pattern, origin, ranging, lower_BB,stoch_k_15m, stoch_k_5m, profit, went_down_by, went_up_after, elapsed))


def evaluate(start_time, buy_price, df_ohlc_1m, take_profit = 0.015, stop_loss = 0.03):
    stop_loss /= 1.2
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



def plot_it(df_ohlc, start_time, end_time, buy_price, sell_price, interval = '15min', savefile = '1'):
    '''df_ohlc is considered to be 1 minute candlesticks'''
    import matplotlib.dates as mdates
    from mpl_finance import candlestick_ohlc
    import matplotlib.pyplot as plt
    import ta
    
    if pd.Timedelta(interval) > pd.Timedelta('1min'):
        #If we want to plot intervals exceeding 1 minutes, then we resample first:
        df_open = df_ohlc['open'].resample(interval).first()
        df_high = df_ohlc['high'].resample(interval).max()
        df_low = df_ohlc['low'].resample(interval).min()
        df_close = df_ohlc['close'].resample(interval).last()
        df_ohlc_15 = pd.concat([df_open,df_high, df_low, df_close], axis=1, keys=['open', 'high', 'low', 'close'])
        df_ohlc_15.reset_index(inplace=True)
    else:
        df_ohlc_15 = df_ohlc
        df_ohlc_15 = df_ohlc_15.reset_index()
    #Get technical indicators:
    BBands = ta.volatility.BollingerBands(df_ohlc_15['close'])
    middle = BBands.bollinger_mavg()
    upper = BBands.bollinger_hband()
    lower = BBands.bollinger_lband()
    
    start_15m_tmp = df_ohlc_15['timestamp'][(df_ohlc_15['timestamp'] <= start_time) & (df_ohlc_15['timestamp'] > start_time - pd.Timedelta(interval)) ].iloc[0] 
    end_15m_tmp = df_ohlc_15['timestamp'][(df_ohlc_15['timestamp'] <= end_time) & (df_ohlc_15['timestamp'] > end_time - pd.Timedelta(interval)) ].iloc[0]
        
    p_range = 6*pd.Timedelta(interval)
    
    start_15m = start_15m_tmp - p_range
    end_15m = end_15m_tmp + p_range
    #plt.clear()
    plot_range = (df_ohlc_15['timestamp'] >= start_15m) & (df_ohlc_15['timestamp'] <= end_15m)
    #df_ohlc_15_plot = df_ohlc_15[ (df_ohlc_15['timestamp'] >= start_15m) & (df_ohlc_15['timestamp'] <= end_15m)]
    df_ohlc_15_plot = df_ohlc_15[ plot_range ]

    df_ohlc_15_plot['timestamp'] = df_ohlc_15_plot['timestamp'].map(mdates.date2num)
    
    plt.close()
    
    fig, ax = plt.subplots(figsize = (10,5))
    
    #ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    #ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
    ax.xaxis_date()
    
    width = pd.Timedelta(interval)/pd.Timedelta('1min')
    
    candlestick_ohlc(ax, df_ohlc_15_plot.values, colorup='g', width=0.0005*width) # ,width=5
    ax.plot(df_ohlc_15_plot['timestamp'], upper[plot_range], c='b', lw=2 )
    ax.plot(df_ohlc_15_plot['timestamp'], middle[plot_range], c='b', lw=1 )
    ax.plot(df_ohlc_15_plot['timestamp'], lower[plot_range], c='b', lw=2 )
    
    ax.annotate('buy', xy=(mdates.date2num(start_time), buy_price),  xycoords='data',\
            xytext=(0.3, 0.45), textcoords='axes fraction',\
            arrowprops=dict(facecolor='black', shrink=0.05),\
            horizontalalignment='left', verticalalignment='top',\
            )
    
    ax.annotate('sell', xy=(mdates.date2num(end_time), sell_price),  xycoords='data',\
        xytext=(0.8, 0.45), textcoords='axes fraction',\
        arrowprops=dict(facecolor='black', shrink=0.05),\
        horizontalalignment='right', verticalalignment='top',\
        )
    
    plt.savefig('%s.png' % savefile, dpi=200)
    

#filename = 'buy_signals_backtest_4try_OGN.dat'
#filename = 'buy_signals_backtest_7try_1month_2.dat'

if __name__ == '__main__':

    filename = 'buy_signals_backtest_8try_1month.dat'
    data = load_data(filename)
    
        #Define main variables:
    TAKE_PROFIT = 0.015 # 1.5% profit
    STOP_LOSS = 0.03 # 3% stop loss
    
    PLOT = False

    #fwrite = 'backtest_evaluate.dat'
    
    
    for i in range(len(data)):
    #for i in range(754, len(data)):
        coin = data['coin'][i]
        symbol = coin + 'BTC'
        pattern = data['pattern'][i]
        origin= data['origin'][i]
        
        #if coin=='BTG' : PLOT = True
        
        hist_data = load_data("%s_1MinuteBars.csv" % symbol)
        
        #start = hist_data.index==data.index[i]
        start_time = data.index[i]
        
        PROFIT = (1+TAKE_PROFIT)*data['buy_price'][i]
        LOSS = (1-STOP_LOSS)*data['buy_price'][i]
        
        ranging = data['ranging'][i]
        lower_BB = data['lower_BB'][i]
        
        stoch_k_15m, stoch_k_5m = data['stoch_k_15m'][i], data['stoch_k_5m'][i]
        
        price = data['buy_price'][i]
        buy_price = price
        min_price = price
        buy_time = data.index[i]
        step = '1m'
        elapsed = pd.Timedelta('0m')
        time_curr = start_time
        #print("Start time %s" % start_time)
        count = 0
        while (price <= 1.5*PROFIT) or (price >= 1.5*LOSS):
            #print( time_curr)
            count += 1
            try :
                price = hist_data[hist_data.index==time_curr]['open'][0]
                price_min = hist_data[hist_data.index==time_curr]['low'][0]
                price_max = hist_data[hist_data.index==time_curr]['high'][0]
            except IndexError:
                break
            #if min_price > price : min_price = price
            if (price_min <= LOSS) or (price_max >= PROFIT) : 
                profit = 100*(price - buy_price)/buy_price
                min_price = hist_data[(hist_data.index>=start_time) & (hist_data.index<=time_curr)]['low'].min()
                max_price = hist_data[(hist_data.index>=start_time) & (hist_data.index<=time_curr)]['high'].max()
                went_down_by = 100*(min_price - buy_price)/buy_price
                went_up_after = 100*(max_price - buy_price)/buy_price
                elapsed = elapsed/pd.Timedelta('1m')
                
                #save_eval(buy_time, coin, buy_price, pattern, origin, ranging, lower_BB, profit, went_down_by, went_up_after, elapsed)
                save_eval(buy_time, coin, buy_price, pattern, origin, ranging, lower_BB, stoch_k_15m, stoch_k_5m, profit, went_down_by, went_up_after, elapsed)
                
                if PLOT: plot_it(hist_data,start_time, time_curr, buy_price, price, savefile="%s_%d_%s" % (pattern, coin, count))
    #                
                break
    #        if price >= PROFIT:
    #            profit = 100*(price - buy_price)/buy_price
    #            min_price = hist_data[(hist_data.index>=start_time) & (hist_data.index<=time_curr)]['low'].min()
    #            max_price = hist_data[(hist_data.index>=start_time) & (hist_data.index<=time_curr)]['high'].max()
    #            went_down_by = 100*(min_price - buy_price)/buy_price
    #            went_up_after = 100*(max_price - buy_price)/buy_price
    #            elapsed = elapsed/pd.Timedelta('1m')
    #            save_eval(buy_time, coin, buy_price, pattern, origin, profit, went_down_by, went_up_after, elapsed)
    #            break
            elapsed += pd.Timedelta(step)
            time_curr += pd.Timedelta(step)
