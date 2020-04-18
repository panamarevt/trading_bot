# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:58:31 2020

@author: Taras
"""

import numpy as np
from pandas import DataFrame as df
import pandas as pd

#Import technical indicators (to be moved from here!)
import ta

from binance_endpoints import get_symbols_BTC, get_ticker
from binance.client import Client

client = Client() # to be moved to binance_endpoints

class C1M:
    '''Body of the C1M strategy'''      
    def __init__(self, min_24h_volume = 150, min_price = 0.00000200, max_price = 0.009, min_ranging = 0.04):
        self.min_24h_volume = min_24h_volume
        self.min_price = min_price
        self.max_price = max_price
        self.min_ranging = min_ranging
        
    
    def get_active(self, symb, active):
    #def get_active(symb, active):
        '''input: symbol, list of symbols
        output: None
        Check if the trading pair satisfies the conditions: 24h volume > 150 BTC and price > 200 Satoshi
        If yes, append the pair in list of active coins
        TODO: Merge with refresh active!
        '''
        #global active
        tmp = get_ticker(symb)
        price = np.float64(tmp['lastPrice'])
        volume = np.float64(tmp['quoteVolume'])
        
        if (self.min_price < price < self.max_price) & (volume > self.min_24h_volume) :
            active.append([symb, price, volume])


    def refresh_active(self):
        '''Create a list of active trading pairs
        Return list of active pairs as a pandas.DataFrame with 3 columns: symbol, price, 24h volume
        '''
        active = []
        symbols = get_symbols_BTC()
        
        for symbol in symbols:
            #Thread(target=get_active, args=(symbol,)).start()
            self.get_active(symbol, active)
        #    tmp = client.get_ticker(symbol=symbol)
        #    price = float(tmp['lastPrice'])
        #    volume = float(tmp['quoteVolume'])
        #    if (0.00000200 < price < 0.001) & (volume > 150) :
        #        active.append([symbol, price, volume])
        active = pd.DataFrame(active, columns=['symbol', 'Price', "Volume"])
        print("Active list formed")
        print(active)
        return active
    
    
    def get_promising(self, active_list): 
        '''Select list of promising coins from the active list
        Inut: active coins
        Return: promising coins
        Promising coins satisfy the condition that distanse between lower and upper Bollinger bands > 4% (relative to lower)
        '''
        promising = {}
        symbols_tmp = []
        vol_tmp = []
        ranging_curr = []
        #count = 0
        for symbol in active_list["symbol"]:
            #Get 15 minute candles
            candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=200)
            #Convert the candles to DataFrame object:
            candles_df = df(candles, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])            
            #Create another dataframe with time and open, high, low, close (ohlc) values only
            ohlc = candles_df.loc[:, ['timestamp', 'open', 'high', 'low', 'close']]
            ohlc = ohlc.astype(float)
            #Get 24h volume:
            quote_av = candles_df.loc[-96:,'quote_av'].astype(float).sum()
            
            coin = symbol[:-3]
            #Compute Bollinger bands using TA-lib from bukosabino            
            # Get technical indicators:    
            # Bollinger Bands
            BBands = ta.volatility.BollingerBands(ohlc['close'])
            #middle = BBands.bollinger_mavg()
            #upper = BBands.bollinger_hband()
            lower = BBands.bollinger_lband()
            width = BBands.bollinger_wband()
            
            #RSI and stochastic:
            #RSI = ta.momentum.RSIIndicator(ohlc['close']).rsi()
            ##stoch = ta.momentum.StochasticOscillator(ohlc['high'],ohlc['low'],ohlc['close']).stoch()
            
            # Stochastic RSI 
    #        stoch_RSI = ta.momentum.StochasticOscillator(RSI,RSI,RSI)
    #        #fastk
    #        fast_k = stoch_RSI.stoch_signal()
    #        #slow_d is 3-day SMA of fast_k
    #        slow_d = ta.volatility.bollinger_mavg(fast_k, n=3)

            # Distance benween bands in '%'
            #ranging =  (upper - lower)/lower
            ranging= width/lower
            ## DEBUG!
            ##print(ranging)
            # Compute mean distance in case current ranging is small, but the coin was ranging high recently
            mean_range = np.mean(ranging[-96:]) # 96 15 minute candles give 24 hours
            max_range = np.max(ranging[-96:])
            
            #The condition for a promising coin:
            if  (max_range > self.min_ranging ) :        
                print("Promising: %s, mean range: %.1f, max: %.1f, current: %.1f" % (coin, 100*mean_range,100*max_range, 100*ranging.iloc[-1])  )
                symbols_tmp.append(symbol)
                vol_tmp.append(quote_av)
                ranging_curr.append(100*ranging.iloc[-1])
                #promising[symbol] = quote_av
        promising = {'symbol':symbols_tmp, 'volume':vol_tmp, 'ranging':ranging_curr}
        print("Promising list formed")
        promising = df(promising)#, columns=['symbol', 'volume'])
        promising.sort_values(by='volume', inplace=True, ascending=False)
        print(promising)
        return promising
    
    def search_signals(self, promising):
        '''Put here all the conditions...'''
        pass
    
    def c1m_flow(self, active_update_interval = 600, promise_update_interval = 300):
        '''Main flow of the strategy: 
        get_active -> get_promising -> search_signals -> repeat'''
        pass