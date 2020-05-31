# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:58:31 2020

@author: Taras
"""

import numpy as np
from pandas import DataFrame as df
import pandas as pd

import time
from datetime import datetime
import os

import concurrent.futures
#from binance_endpoints import get_symbols_BTC, get_ticker
#from binance_endpoints import *   # quick and dirtly solution to check consisntency with previous version. to be changed!
import binance_endpoints
from binance.client import Client

import indicators

def save_signal(t_curr, coin, price, counter, pattern):
    with open('buy_signals_pattern_new.dat', 'a') as f:
        f.write("%s   %5s    %.8f   %2d  %12s\n" % (t_curr, coin, price, counter, pattern))

def check_profit_new(coin_dict, price_curr, take_profit = 0.015, stop_loss = 0.03, trading="NO"):
    '''This is old function, to be deprecated'''
    coin = coin_dict['coin']
    bought_price = coin_dict['buy_price']
    bought_time = coin_dict['start_signal']
    #Check minimum price:
    if price_curr < bought_price : coin_dict['min_price'] = price_curr
    #Check stop loss or take profit
    if price_curr >= (1+take_profit)*bought_price:
        time_now = time.time()
        elapsed = (time_now - bought_time)/60
        status = 1
    elif price_curr <= (1-stop_loss)*bought_price:
        time_now = time.time()
        elapsed = (time_now - bought_time)/60
        status = -1
    else:
        status = 0
    #if trading complete, save coin data:
    if status != 0 :    
        bought_time_fmt = coin_dict['buy_time']
        profit = (price_curr - bought_price)/bought_price * 100
#        save_coin_full(coin, bought_time_fmt, bought_price, profit, elapsed, \
#                       coin_dict['pattern'], coin_dict['origin'], coin_dict['ranging'], \
#                       coin_dict['vol_1hr'], coin_dict['fastk15'], coin_dict['min_price'], trading)
    return status    

def weighted_avg_orderBook(book, qty):
    '''book is list of asks or bids with prices and quatities, e.g.:
        asks': [['0.00000644', '6749.00000000'],
      ['0.00000645', '25059.00000000'],
      ['0.00000646', '33069.00000000'],
      ['0.00000647', '1007.00000000'],]'''
    book = np.array(book)   
    pr = book[:,0].astype(float) # list pf prices
    qt = book[:,1].astype(float) # list of corresponding quantities
    StartQty = qt[0]
    #If first quantity in the order book is larger then our quantity, return the first price:
    if StartQty >= qty: return pr[0]
    qty_tmp = qty # We will reduce this value in the while loop below
    price_times_qty = 0
    j = 0
    while qty_tmp > StartQty:
        price_times_qty += pr[j]*qt[j]
        qty_tmp -= StartQty # We reduce our quantity here
        j += 1
        StartQty = qt[j] # Set start quantity to the next in the list
        # If StartQty happens to exceed our remaining quantity the while loop will terminate
    # Now we have to compute the final product: last price times remaining quantity
    price_times_qty += pr[j]*qty_tmp
    # The total weighted average price is :
    avg_price = price_times_qty/qty
    return avg_price   

def run_parallel(symbols, func, n_threads = 8):
# use threading here to run the function in parallel
    i = 0
    while i < len(symbols):

        with concurrent.futures.ThreadPoolExecutor() as executor:
            #executor.map(binanceBarExtractor, symbols[i:i+8])
            if i + n_threads <= len(symbols):
                executor.map(func, symbols[i:i+n_threads])
            else:
                executor.map(func, symbols[i:])
        i += n_threads

client = Client() # to be moved to binance_endpoints

class C1M:
    '''Body of the C1M strategy'''      
    def __init__(self, min_24h_volume = 150, min_price = 0.00000200, max_price = 0.009, min_ranging = 0.04):
        self.min_24h_volume = min_24h_volume
        self.min_price = min_price
        self.max_price = max_price
        self.min_ranging = min_ranging
        #Initialize empty dictionaries for coins in trade:
        self.triggered = {}
        self.trading_coins = {}
    
    def get_active(self, symb, active):
    #def get_active(symb, active):
        '''input: symbol, list of symbols
        output: None
        Check if the trading pair satisfies the conditions: 24h volume > 150 BTC and price > 200 Satoshi
        If yes, append the pair in list of active coins
        TODO: Merge with refresh active!
        '''
        #global active
        try:
            tmp = binance_endpoints.get_ticker(symb)
        except Exception as e:
            print(f"Warning! Didn't get tickers for {symb}!", e)
            return 0
        price = np.float64(tmp['lastPrice'])
        volume = np.float64(tmp['quoteVolume'])
        
        if (self.min_price < price < self.max_price) & (volume > self.min_24h_volume) :
            active.append([symb, price, volume])


    def refresh_active(self):
        '''Create a list of active trading pairs
        Return list of active pairs as a pandas.DataFrame with 3 columns: symbol, price, 24h volume
        '''
        active = []
        
        symbols = binance_endpoints.get_symbols_BTC()
        
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
            
            try:
                candles_df = binance_endpoints.GetKlines(symbol, interval='15m')
            except Exception as e:
                print(f"Warning! Didn't get klines for {symbol}!", e)
                continue
            #Create a dataframe with time and open, high, low, close (ohlc) values only
            ohlc = candles_df.loc[:, ['timestamp', 'open', 'high', 'low', 'close']]
            
            #Get 24h volume:
            quote_av = candles_df.loc[-96:,'quote_av'].astype(float).sum()
            
            coin = symbol[:-3]                  
            # Get technical indicators:    
            # Bollinger Bands
            lower, _, _, width = indicators.get_BBands(ohlc['close'])
            #ranging =  (upper - lower)/lower
            ranging= width/lower
    #        slow_d = ta.volatility.bollinger_mavg(fast_k, n=3)
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
        #promising.sort_values(by='volume', inplace=True, ascending=False)
        promising.sort_values(by='ranging', inplace=True, ascending=False)
        print(promising)
        return promising
    
    def search_signals(self, promising):
        '''Put here all the conditions...'''
        '''Watch the promising coins:
        Check if the price is hitting lower Bollinger bands, stochastic RSI k>d and both going up
        If coin fills conditions of BBands and stoch RSI, buy signal is triggered, coin is stored in the dictionary of triggered coins
        If after 5 minutes the coin is still satisfying the conditions, buy signal is triggered again
        '''
        #global triggered, trading_coins #trade_modeling 
        
        #for symbol in promising:
        for symbol in promising.loc[:,'symbol']:
            
            try:
                candles_df = binance_endpoints.GetKlines(symbol)
            except Exception as e:
                print(f"Warning: didn't get klines for {symbol}!", e)
                continue
            
            Open, high = candles_df['open'], candles_df['high']
            low, close = candles_df['low'], candles_df['close']
            volume = candles_df['quote_av']
            
            coin = symbol[:-3]
            
            # Check if coin was bought in previous step of the loop
            if coin in list(self.trading_coins):
                if 'orderId' in self.trading_coins[coin]['order'].keys():
                    try:
                        binance_endpoints.check_buy_order(self.trading_coins[coin]['order'], self.trading_coins, coin, trade_type=self.TRADE_TYPE)
                    except:
                        print("Warning: didn't check BUY order!")
                else:
                    try:
                        binance_endpoints.check_oco_order(self.trading_coins[coin]['order'], self.trading_coins, coin, trade_type=self.TRADE_TYPE)
                    except:
                        print("Warning: didn't check OCO order!")
            # Check if coin signalled   in previous step of the loop                              
            if coin in self.triggered.keys():
                #status = check_profit(triggered[coin]['coin'], close.iloc[-1], triggered[coin]['start_signal'], triggered[coin]['buy_price'], triggered[coin]['buy_time'] )
                status = check_profit_new(self.triggered[coin], close.iloc[-1])
                #status = check_profit_new(triggered[coin], candles_df.loc[:,'timestamp':'close'])
                if status == 0:
                    continue
                else: 
                    del self.triggered[coin]        
                
            # Get technical indicators:    
            # Bollinger Bands
            lower, middle, upper, width = indicators.get_BBands(close) 
            ranging= width/lower *100 # in %
            # Check for Bollinger band breach (lower or middle):
            lower_BB_signal = indicators.check_lower_BB(close, low, lower)
            middle_BB_signal = indicators.check_middle_BB(Open, close, low, middle)
      
           # Check if range is > 4% and price is hitting lower Bollinger Bands:
            if ranging.iloc[-1] > self.min_ranging :
                # Note. Use iloc() to get access to position (same as indexing in numpy arrays). Reason that there is no negative indexing in pandas.Series        
                if lower_BB_signal or middle_BB_signal: 
                    # If Bollingers are hit: compute stochastic RSI                                
                    #StochasticRSI:
                    fastk, slowd = indicators.get_StochRSI(close)
                    #Stochastic RSI signal fask > slowd, both going up and 10 < fastk < 30
                    stochRSI_signal = indicators.check_stochRSI_signal(fastk, slowd, kmin=10, kmax=30)
                    #Condition that distance from buy price to mid(up) bollinder band is greater than 1.5% 
                    if stochRSI_signal:    
                        try:
                            buy_price = float(binance_endpoints.get_buy_price(symbol,self.BUY_METHOD))
                        except Exception as e:
                            print("Warning! Didn't get buy price!", e)
                            continue
                        if lower_BB_signal:
                            dist_to_BB = 100*(middle.iloc[-1] - buy_price)/buy_price
                            price_to_range = (buy_price - lower.iloc[-1])/(middle.iloc[-1] - lower.iloc[-1])
                        else:
                            dist_to_BB = 100*(upper.iloc[-1] - buy_price)/buy_price
                            price_to_range = (buy_price - middle.iloc[-1])/(upper.iloc[-1] - middle.iloc[-1])
                        price_cond = (dist_to_BB > 1.5)
                        
                        #price_dist_cond = (price_to_range < 0.1)
                        price_dist_cond = True 
                        
                    if stochRSI_signal and price_cond and price_dist_cond:
                        # Get the current time for output:                   
                        now = datetime.now()
                        current_time = now.strftime("%d/%m/%y %H:%M:%S")
                        # Check if the coin was already signalling less than 5 minutes ago:           
                        #counter = 0
                        if coin in self.triggered.keys(): 
                            # Get current time to measure time intervals between buy signals of a coin
                            time_now = time.time()    
    #                        # If the coin was signalling more than 5 minutes (300 sec) ago, remove it from the list
                            if time_now - self.triggered[coin]['start_signal'] > 300: 
                                buy_price = self.triggered[coin]['buy_price']
                                self.triggered[coin]['counter'] += 1
                                
                                save_signal(current_time, coin, buy_price, self.triggered[coin]['counter'])
                                #del triggered[coin]
                            else:
                                continue
                               
                        else:
                            #buy_price = close.iloc[-1]
                            try:
                                buy_price = float(binance_endpoints.get_buy_price(symbol,self.BUY_METHOD))
                                min_price = buy_price
                                rec_price = binance_endpoints.get_rec_price(symbol,buy_price)
                            except Exception as e:
                                print("Warning! Didn't get buy price and rec price!", e)
                                continue
                            print(current_time, f"BUY signal! {coin} at {buy_price:.8f}; rec: {rec_price:.8f} Stoch RSI: ", fastk.iloc[-1], slowd.iloc[-1])
                            start_signal = time.time()
                            #save signaling coins to file:
                            counter = 0
                            # Check if there was a candle pattern 1 or 2 canclesticks before:
                            pattern = indicators.check_candle_patterns(Open, high, low, close)
                            
                            save_signal(current_time, coin, buy_price, counter, pattern)
                            # Create a new item in the dictionary of signalling coins, where coin is the key and time at the 1st signal is the item
                            status = 0
                            if lower_BB_signal:
                                origin = 'lower'
                            else: origin = 'upper'
                            vol_1hr = volume[-4:].sum()
                            #triggered[coin] = [start_signal, buy_price, current_time, status]
                            #Create a dictionary with all coins that triggered a buy signal:
                            self.triggered[coin] = {'coin':coin, 'start_signal':start_signal, 'buy_price':buy_price, 'buy_time':current_time, 'status':status, 'counter':counter, \
                                     'vol_1hr':vol_1hr, 'pattern':pattern, 'origin':origin, 'ranging':ranging.iloc[-1], 'fastk15': fastk.iloc[-1], 'min_price':min_price }
                            n_trades = len(self.trading_coins)
                            if (n_trades < self.MAX_TRADES) and (coin not in self.trading_coins.keys() ):
                                #in_trade = len(trading_coins)
                            
                                print("Placing Buy Order")
                                    
                                try:
                                    order, am_btc = binance_endpoints.place_buy_order(symbol,self.MAX_TRADES,n_trades,self.BUY_METHOD,self.DEPOSIT_FRACTION, trade_type=self.TRADE_TYPE)
                                except Exception as e:
                                    print("Exception during pacing BUY order occured", e)
                                    continue                             
                                    
                                if self.BUY_METHOD == 'MARKET':
                                    buy_price = order['price']
                                    print("Placing OCO order", symbol)
                                    try:                                    
                                        order = binance_endpoints.place_oco_order(symbol, buy_price, take_profit=0.017, stop_loss = 0.02)
                                    except Exception as e:
                                        print("Exception during placing OCO order:", symbol, buy_price)
                                        print(e)
                                else:
                                    #If we didn't use market buy then we have to check the order status                                    
                                    #print("Check order status")
                                    if self.TRADE_TYPE == 'REAL':
                                        print("Check order status")
                                        try:
                                            order = binance_endpoints.check_order_status(order)
                                        except Exception as e:
                                            print("Didn't manage to check order status", e)
                                
#                                self.trading_coins[coin] = {'coin':coin, 'start_signal':start_signal, 'buy_price':buy_price, \
#                                          'rec_price':rec_price, 'buy_time':current_time, 'status':status, 'counter':counter, \
#                                          'vol_1hr':vol_1hr, 'pattern':pattern, 'origin':origin,\
#                                          'ranging':ranging.iloc[-1], 'fastk15': fastk.iloc[-1], 'min_price':min_price, 'order':order, 'am_btc': am_btc,'n_oco':0 }
#     
                                self.trading_coins[coin] = {'buy_time':current_time, 'coin':coin, 'buy_price':buy_price, \
                                          'rec_price':rec_price,  'status':status, 'counter':counter, 'start_signal':start_signal,\
                                          'vol_1hr':vol_1hr, 'pattern':pattern, 'origin':origin,\
                                          'ranging':ranging.iloc[-1], 'fastk15': fastk.iloc[-1], 'min_price':min_price, 'order':order, 'am_btc': am_btc,'n_oco':0 }
    
    def c1m_flow(self, active_update_interval = 600, promise_update_interval = 300, 
                 TRADE_TYPE='PAPER', BUY_METHOD='LIMIT', MAX_TRADES=4, DEPOSIT_FRACTION=0.1):
        '''Main flow of the strategy: 
        get_active -> get_promising -> search_signals -> repeat'''
        
        self.BUY_METHOD = BUY_METHOD # Specify if we use LIMIT or MARKET order
        self.MAX_TRADES = MAX_TRADES # Number of open positions
        self.DEPOSIT_FRACTION = DEPOSIT_FRACTION # Fraction of the deposit in BTC we use for trading
        self.TRADE_TYPE = TRADE_TYPE
        
        active_list = self.refresh_active()
        promising = self.get_promising(active_list)
        
        total_watch = 0
        last_active_uptade = 0
        
        #start_loop = time.time()
        # Start infinite loop:
        while True:        
            # Watch promising coins
            start_watch = time.time()   
            #args = (promising,)
            #try_func(watch, 10,3600, *args)
            self.search_signals(promising)
            end_watch = time.time()
            #Check how many coins are in trade at the moment
            are_trading = len(self.trading_coins)
            #If there are coins in trade, check their status
            if are_trading > 0: 
                #print("%d coins are in trade:" % are_trading)
                for item in list(self.trading_coins): #here better to use list(trading_coins) instead of trading_coins.keys()
                    if 'orderId' in self.trading_coins[item]['order'].keys():
                        print("BUY PENDING:", item)
                        try:
                            binance_endpoints.check_buy_order(self.trading_coins[item]['order'], self.trading_coins, item, self.TRADE_TYPE)
                        except Exception as e:
                            print("Didn't check BUY order", item)
                            print(e)
                    else:
                        print("OCO:", item)
                        try:
                            binance_endpoints.check_oco_order(self.trading_coins[item]['order'], self.trading_coins, item, trade_type=self.TRADE_TYPE)
                        except Exception as e:
                            print("Didn't check OCO order", item)
                            print(e)                            
    
            # Measure total time spent for watching:
            total_watch += end_watch - start_watch
            # If watching time is more than 5 minutes, update promising list: 
            if total_watch > promise_update_interval: 
                #args = (active_list,)
                #promising = try_func(get_promising, 10, 3600, *args)
                promising = self.get_promising(active_list)
                last_active_uptade += total_watch
                total_watch = 0
            # Update active list every 10 minutes
            if last_active_uptade > active_update_interval:
                #active_list = try_func(refresh_active)
                try:
                    active_list = self.refresh_active()
                except Exception as e:
                    print("Warning! Active list wasn't updated!", e)
                #args = (active_list,)
                
                promising = self.get_promising(active_list)
                #promising = try_func(get_promising, 10, 3600, *args)
    
                are_trading = len(self.trading_coins)
                print("%d coins are in trade:" % are_trading)            
                if are_trading > 0:                 
                    for item in self.trading_coins.keys():
                        print(item, self.trading_coins[item]['buy_time'])
                        
                total_watch = 0
                last_active_uptade = 0
            


class Volume:
    '''Body of the Volume strategy
    Idea of the strategy is to track large movements of volume
    and if the price goes up, jump on the board :-)
    For now it is designed to work with BTC trading pairs, but can be generalazied to other pairs'''      
    def __init__(self, MIN_PRICE = 0.00000200, MAX_PRICE = 0.009, N_RED = 1, MINS_BEFORE = 15,
                 QUOTE_AV_MIN = 3, PRICE_CHANGE_PC = 2, STOP_LOSS = 0.05):
        # Initialize main parameters of the strategy
        # Minimum and maximum price
        self.MIN_PRICE = MIN_PRICE # 200 Satoshi
        self.MAX_PRICE = MAX_PRICE # Setting to a large value makes it possible to trade without upper price limit
        # Stop loss percentage
        self.STOP_LOSS = STOP_LOSS
        # Number of red candles to sell after. Backtests show that it is better to sell after the first red candle
        self.N_RED = N_RED
        # Interval in minutes to campare with the current volume.
        self.MINS_BEFORE = MINS_BEFORE
        # Minimum volume in BTC for current 1 minute candle
        self.QUOTE_AV_MIN = QUOTE_AV_MIN
        # Price change criterion for the current candle
        self.PRICE_CHANGE_PC = PRICE_CHANGE_PC
        self.in_trade = {}
        
    def search(self, symbol):
            if symbol in list(self.in_trade) : 
        # Don't buy the same coin multiple times
        #logger.info(f"This coin is already in trade: {symbol}")
        #continue
                return 0
            try:
                candles_df = binance_endpoints.GetKlines(symbol, interval='1m', limit=100)
                #candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=100)
            except Exception as e:
                #logger.exception(f"Didn't get kline intervals for {symbol}")
                print(f"Didn't get kline intervals for {symbol}")
                print(e)
                #continue
                return 0
            #candles_df = df(candles, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
            #candles_df = candles_df.astype('float')
        
            #close = candles_df['close']
            price_curr = candles_df['close'].iloc[-1]
            
            min_price_cond = (price_curr  > self.MIN_PRICE) #If minimum price condition is not satisfied, we don't need to do anything here:
            if not min_price_cond: 
                #continue
                return 0
            
            #print("Check status!")
            #save_volumes(symbol, candles_df)
            
            prev_index = -1*self.MINS_BEFORE-1 # Subtract one to exclude current candle with index -1
            vol_prev = candles_df['volume'].iloc[prev_index : -1].sum()
            #vol_24h = candles_df['quote_av'].iloc[i-1440 : i].sum()
            #candle_color = candle_params(candles_df['open'].iloc[i], candles_df['high'].iloc[i],candles_df['low'].iloc[i],candles_df['close'].iloc[i])[-1]
            vol_curr = candles_df['volume'].iloc[-1]
            q_vol = candles_df['quote_av'].iloc[-1]
            
            trades = int(candles_df['trades'].iloc[-1])
            
            vol_curr_last = candles_df['volume'].iloc[-2]
            q_vol_last = candles_df['quote_av'].iloc[-2]
            vol_prev_last = candles_df['volume'].iloc[prev_index-1 : -2].sum()
            trades_last = int(candles_df['trades'].iloc[-2])
            
            # Conditions:
            vol_cond = (vol_curr > vol_prev)
            quote_col_cond = (q_vol > self.QUOTE_AV_MIN)
            #green_candle = (candle_color == 'green')       
            
            vol_cond_last = (vol_curr_last > vol_prev_last)
            quote_col_cond_last = (q_vol_last > self.QUOTE_AV_MIN)
            #green_candle = (candle_color == 'green')
        
            if vol_cond_last and (not vol_cond):
                price_last = candles_df['close'].iloc[-3]
            else:
                price_last = candles_df['close'].iloc[-2]
            
            # Price conditions:
            #price_change = 100*(price_curr - price_last)/price_last
            # Don't take into account the current price to make it more comparable with backtest
            price_change = 100*(candles_df['close'].iloc[-2] - candles_df['close'].iloc[-3])/candles_df['close'].iloc[-3]     
            price_cond = (price_change > self.PRICE_CHANGE_PC) 
           
            if (vol_cond or vol_cond_last) and (quote_col_cond or quote_col_cond_last) :
                #logger.info(f"Volume condition! {symbol}:{vol_cond}:{vol_cond_last}:{vol_curr}:{vol_prev}:{price_change:.1f}:{q_vol:.2f}")
                print(f"Volume condition! {symbol}:{vol_cond}:{vol_cond_last}:{vol_curr}:{vol_prev}:{price_change:.1f}:{q_vol:.2f}")
            if price_cond:
                #logger.info(f"Price condition! {symbol}:{vol_cond}:{vol_cond_last}:{vol_curr}:{vol_prev}:{price_change:.1f}:{q_vol:.2f}")
                print(f"Price condition! {symbol}:{vol_cond}:{vol_cond_last}:{vol_curr}:{vol_prev}:{price_change:.1f}:{q_vol:.2f}")
                
        #    if (vol_cond and quote_col_cond and price_cond and min_price_cond) or \
        #            (vol_cond_last and quote_col_cond_last and price_cond and min_price_cond):
        
            # Use only condition for the finished candles! (This should give better performance as compared to the backtest)
            if (vol_cond_last and quote_col_cond_last and price_cond and min_price_cond):    
            
                buy_time = time.time()
                book = client.get_order_book(symbol=symbol, limit=1000)
                buy_price = book['asks'][0][0]
                #buy_price = binance_endpoints.get_buy_amount(symbol, "MARKET")
                qties = [am/float(buy_price) for am in self.AMOUNTS]
                deals = [weighted_avg_orderBook(book['asks'], qty) for qty in qties]
                #logger.info(f"Buy {symbol} for {buy_price}!")
                print(f"Buy {symbol} for {buy_price}!")
                if vol_cond_last and (not vol_cond) :
                    vol_curr = vol_curr_last
                    trades = trades_last
                    q_vol = q_vol_last
                self.in_trade[symbol] = {'symbol':symbol, 'buy_time':buy_time, 'vol_curr':vol_curr, 'q_vol':q_vol, 'price_change':price_change,'trades':trades,
                        'quantities':qties, 'buy_deals':deals}
    

      
    def evaluate(self, symbol):
        '''Check if sell condition is hit and sell the `symbol`'''
        try:
            candles_df = binance_endpoints.GetKlines(symbol, interval='1m', limit=100)
        except:
            #logger.exception(f"WARNING: Didn't get kline intervals for {symbol} during evaluation!")
            print(f"WARNING: Didn't get kline intervals for {symbol} during evaluation!")
            return 0
            #continue
        op, hi, lo, cl = candles_df['open'].iloc[-2], candles_df['high'].iloc[-2],candles_df['low'].iloc[-2],candles_df['close'].iloc[-2]
        last_candle_color = indicators.candle_params(op, hi, lo, cl)[-1]
        print(last_candle_color)
        time_now = time.time()
        elapsed = time_now - self.in_trade[symbol]['buy_time']
        if (last_candle_color == 'red') and (elapsed > 60): # Here elapsed time is measured in seconds
            # We wait at least 1 minute before we can sell. Maybe it is better to sell imediately if the candle where we bought closes red? 
            # We can use 'close_time' of the buy candle to check whether that candle has been closed or not.
            # Another idea for this strategy: don't wait until candle closes red, check only the sell volume.
            book = client.get_order_book(symbol=symbol)
            sell_deals = [ weighted_avg_orderBook(book['bids'], qty) for qty in self.in_trade[symbol]['quantities'] ]                
            self.in_trade[symbol]['sell_deals'] = sell_deals
            self.in_trade[symbol]['elapsed'] = elapsed/60
            self.in_trade[symbol]['profits'] = [100*(sell - buy)/buy for sell, buy in zip(self.in_trade[symbol]['sell_deals'], self.in_trade[symbol]['buy_deals'] )]
            ##logger.info(f"Sell {symbol} for {sell_deals[0]:.8f}! Profit: {in_trade[symbol]['profits'][0]:.2f}")
            print(f"Sell {symbol} for {sell_deals[0]:.8f}! Profit: {self.in_trade[symbol]['profits'][0]:.2f}")
            self.save_trade(symbol)
            del self.in_trade[symbol]


    def save_trade(self, symbol):
        fname = 'trades_volume_strategy_demo3_1-10.dat'
        print(f"Saving trade info for {symbol} ...")
        with open(fname, 'a') as f:
            empty = os.path.getsize(fname) == 0
            if empty:
                f.write("buy_time, symbol, buy1, buy2, buy3, buy4, qty1, qty2, qty3, qty4, sell1, sell2, sell3, sell4, profit1, profit2, profit3, profit4, elapsed, vol_curr, q_vol, price_change, trades\n")
            buy_time = pd.to_datetime(self.in_trade[symbol]['buy_time'], unit='s')
            buy1,buy2,buy3,buy4 = self.in_trade[symbol]['buy_deals']
            qty1,qty2,qty3,qty4 = self.in_trade[symbol]['quantities']        
            sell1,sell2,sell3,sell4 = self.in_trade[symbol]['sell_deals']
            p1,p2,p3,p4 = self.in_trade[symbol]['profits']
            f.write(f"{buy_time:%Y-%m-%d %H:%M:%S},{symbol},{buy1:.8f},{buy2:.8f},{buy3:.8f},{buy4:.8f},{qty1:.2f},{qty2:.2f},{qty3:.2f},{qty4:.2f},{sell1:.8f},{sell2:.8f},{sell3:.8f},{sell3:.8f},{p1:.2f},{p2:.2f},{p3:.2f},{p4:.2f},{self.in_trade[symbol]['elapsed']:.1f},{self.in_trade[symbol]['vol_curr']:.1f},{self.in_trade[symbol]['q_vol']:.2f},{self.in_trade[symbol]['price_change']:.1f},{self.in_trade[symbol]['trades']}\n")
            
    
    @staticmethod
    def save_volumes(symbol, candles):
        '''Record real-time volume data
        canldes -- a DataFrame containing the candlestick information
        '''
        b_vol = candles['volume']
        q_vol = candles['quote_av']
        
        vol_curr = b_vol.iloc[-1]
        q_vol_curr = q_vol.iloc[-1]    
        vol_prev = b_vol.iloc[-2]
        q_vol_prev = q_vol.iloc[-2]
        vol_prev_15 = b_vol.iloc[-16:-1].sum()
        q_vol_prev_15 = q_vol.iloc[-16:-1].sum()
        
        fname= f'VolumeData/{symbol}_volume.dat'
        with open(fname, 'a') as f:
            if os.path.getsize(fname) == 0:
                f.write("timestamp,vol_curr,q_vol_curr,vol_prev,q_vol_prev,vol_prev_15,q_vol_prev_15\n")
            timestamp = candles['timestamp'].iloc[-1]
            f.write(f"{timestamp},{vol_curr},{q_vol_curr},{vol_prev},{q_vol_prev},{vol_prev_15},{q_vol_prev_15}\n")
            
    

    
    def volume_flow(self, AMOUNTS = [0.05, 0.1, 0.2, 1], n_jobs=8):
        '''Main flow of the strategy'''
        # Amounts in BTC for paper trading
        self.AMOUNTS = AMOUNTS
        # Get active trading symbols:
        symbols = binance_endpoints.get_symbols_BTC()
        # The infinite loop starts here:
        while True:
            t1 = time.time()
            # Run in parallel the main function to search for the signal of the strategy
            run_parallel(symbols, self.search, n_threads=n_jobs)
            
            t2 = time.time()
            # Measure time takes to check all the symbols
            elpsd = t2 - t1
            #logger.debug(f"Checked all symbols in {elpsd} sec.")
            print(f"Checked all symbols in {elpsd} sec.")  
            # Check if some a coin is in trade:
            if len(self.in_trade) > 0:
    
                ##logger.debug(f"{len(in_trade.keys())} coins are in trade")
                print(self.in_trade.keys())
                then = time.time()                
                #logger.debug("Start evaluating strategy")
                # Evaluate the trading coins:
                for symbol in list(self.in_trade):
                    self.evaluate(symbol)
                
                now = time.time()
                elpsd = now - then
        
        
        