# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 01:27:45 2020

@author: Taras

copied from: 
https://fxgears.com/index.php?threads/how-to-acquire-free-historical-tick-and-bar-data-for-algo-trading-and-backtesting-in-2020-stocks-forex-and-crypto-currency.1229/#post-19305
"""

import pandas as pd
from binance.client import Client
import datetime
import keys

#import os
import concurrent.futures
#import logging
import os

from binance_endpoints import get_symbols_BTC

#logging.basicConfig(filename='bar_extractor.log', level=logging.INFO,
#                    format='%(levelname)s:%(message)s')

# YOUR API KEYS HERE
#api_key = ""    #Enter your own API-key here
#api_secret = "" #Enter your own API-secret here

bclient = Client(api_key=keys.Pkey, api_secret=keys.Skey)

#today = datetime.datetime.strptime('31 Mar 2020', '%d %b %Y')

def binanceBarExtractor(symbol, start_date = datetime.datetime.strptime('1 Jan 2020', '%d %b %Y'), today = datetime.datetime.today(), 
        where='Crypto_1MinuteBars/'):
    #print('working...%s' % symbol)
    #logging.info(f'working...{symbol}')
    #filename = '{}_1MinuteBars.csv'.format(symbol)
    filename = where + symbol + '_1MinuteBars.csv'
    try:
        klines = bclient.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, start_date.strftime("%d %b %Y %H:%M:%S"), today.strftime("%d %b %Y %H:%M:%S"), 1000)
        data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    
        data.set_index('timestamp', inplace=True)
        data.to_csv(filename)
        #print('finished!')
        #logging.info('finished!')
    except Exception as e:
        #logging.exception(f"Didn't download {symbol}")
        print("Didn't download %s" % symbol)
        print(e)


def binanceBarUpdater(symbol,  today = datetime.datetime.today(),  where='Crypto_1MinuteBars/'):
    #filename = '{}_1MinuteBars.csv'.format(symbol)
    filename = where + symbol + '_1MinuteBars.csv'
    #logging.info(f'working...{symbol}')
    
    if os.path.isfile(filename):
        print(f"Update {filename} ...")
        last_line = open(filename, 'r').readlines()[-1]
        start_date = pd.to_datetime(last_line.split(',')[0])
        #today = pd.to_datetime(datetime.datetime.now())
        today = datetime.datetime.now()
        print(start_date)
        try:
            klines = bclient.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, start_date.strftime("%d %b %Y %H:%M:%S"), today.strftime("%d %b %Y %H:%M:%S"), 1000)
            data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    
            data.set_index('timestamp', inplace=True) 
            data.to_csv(filename, mode='a', header=False)
            print(f'finished {symbol}!')
            #logging.info('finished!')
        except Exception as e:
            #logging.exception(f"Didn't download {symbol}")
            print(f"Didn't download {symbol}")
            print(e)
    else:
        print(f"File {filename} doesn't exist") 
        binanceBarExtractor(symbol, where=where)
    

def extract_parallel(symbols):
    # use threading here to run the function in parallel
    i = 0
    while i < len(symbols):

        with concurrent.futures.ThreadPoolExecutor() as executor:
            #executor.map(binanceBarExtractor, symbols[i:i+8])
            if i + 8 <= len(symbols):
                executor.map(binanceBarUpdater, symbols[i:i+8])
            else:
                executor.map(binanceBarUpdater, symbols[i:])
        i += 8
     
    

if __name__ == '__main__':
    # Obviously replace BTCUSDT with whichever symbol you want from binance
    # Wherever you've saved this code is the same directory you will find the resulting CSV file
    #symbols = get_symbols_BTC()

    #start_date = datetime.datetime.strptime('1 A 2020', '%d %b %Y')
    #today = datetime.datetime.today()

#symbols=['ETHBTC','STEEMBTC']
#    for symbol in symbols:
#        #if not os.path.isfile("%s_1MinuteBars.csv" % symbol):        
#        try :                
#            binanceBarExtractor(symbol)
#        except:
#            print("Didn't download %s" % symbol)
#            continue

    #extract_parallel(symbols)   
    
    symbol = 'BTCUSDT'
    binanceBarUpdater(symbol, where='')


#    for symbol in symbols:
#        binanceBarUpdater(symbol)
#
#
#        or ( (df_ohlc['close'].iloc[i-1] < middle.iloc[i-1] and df_ohlc['open'].iloc[i-1] > middle.iloc[i-1]) \
#        or (df_ohlc['low'].iloc[i-1] < middle.iloc[i-1] and df_ohlc['open'].iloc[i-1] > middle.iloc[i-1] and df_ohlc['close'].iloc[i-1] > middle.iloc[i-1]) )
#    
