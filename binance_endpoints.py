# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 05:15:10 2020

@author: Taras
"""

import numpy as np
import time
import datetime

from binance.client import Client
from binance.enums import *
import keys

client = Client(api_key=keys.Pkey, api_secret=keys.Skey)


def get_symbols_BTC():
    import json
    import requests
    #import time
    BASE_URL = 'https://api.binance.com'
    symbols = []
    try:
        resp = requests.get(BASE_URL + '/api/v1/ticker/allBookTickers')
    except:
        print("Time out error. No internet connection?")
        #print('Wait 10 sec. and retry...')
        start = time.time()
        done = start
        elapsed = done - start
        interval = 3600*5 #  seconds
        while elapsed < interval:
            print("Connecting....")
            try:
                resp = requests.get(BASE_URL + '/api/v1/ticker/allBookTickers')
                print("Connection established")
                break
            except:
                done = time.time()
                elapsed = done - start
                if elapsed < 10 : time.sleep(10)
            
    tickers_list = json.loads(resp.content)
    
    # Select only pairs with BTC
    for ticker in tickers_list:
        if (str(ticker['symbol'])[-3:] == 'BTC')  and (float(ticker['askQty']) > 0) :
            symbols.append(ticker['symbol'])
    return symbols

def save_filled_oco(coin_dict, profit):
    '''Saves info about filled OCO orders in a file'''
    elapsed = (time.time() - coin_dict['start_signal'])/60
    bought_time, coin, bought_price, pattern,origin,ranging,vol_1hr,fastk15,min_price = \
    coin_dict['buy_time'],coin_dict['coin'],float(coin_dict['buy_price']),coin_dict['pattern'], coin_dict['origin'],\
    coin_dict['ranging'],coin_dict['vol_1hr'],coin_dict['fastk15'],coin_dict['min_price']
    am_btc, g_profit = coin_dict['am_btc'], coin_dict['g_profit']
    with open('filled_oco_orders_market.dat', 'a') as f:
        f.write("%s,%s,%.8f,%.2f,%.1f,%s,%s,%.2f,%.3f,%.2f,%.8f,%.8f,%.8f\n" % (bought_time, coin, bought_price, profit, elapsed,\
                                                                      pattern,origin,ranging,vol_1hr,fastk15,min_price,\
                                                                      am_btc,g_profit))


def get_buy_price(symbol, BUY_METHOD):
    '''For now use very basic approach - place order at 2nd place in the order book for a limit order
    For a market order take the first price. For market order we need price to determine the quatity of the coin we are going to buy
    '''
    order_book = client.get_order_book(symbol=symbol)
    if BUY_METHOD == 'MARKET':
        buy_price = order_book['asks'][0][0]
    else:        
        buy_price = order_book['bids'][1][0]
    return buy_price

def weighted_avg(fills, symbol):
    '''Computes weghted average price of a market order
    fills is array of dictionaries with prices and quantities (from market order API response)
    "fills": [
        {
            "price": "4000.00000000",
            "qty": "1.00000000",
            "commission": "4.00000000",
            "commissionAsset": "USDT"
        },
        {
            "price": "3999.00000000",
            "qty": "5.00000000",
            "commission": "19.99500000",
            "commissionAsset": "USDT"
        }]
    '''
    price_times_qty = 0
    qty = 0    
    #
    for fill in fills:       
        price_times_qty += float(fill['price']) * float(fill['qty'])
        qty += float(fill['qty'])
    #
    avg_price = price_times_qty/qty
    # Round the price to the nearest step
    price_prec = get_price_precision(symbol)
    avg_price = np.round( avg_price, price_prec)
    avg_price = "%.8f" % avg_price
    
    return  avg_price
    
       

#Orders:

def place_market_sell_order(symbol, qty):
    market_order = client.order_market_sell(symbol=symbol, quantity=qty)
    return market_order

def place_market_buy_order(symbol, qty):
    market_order = client.order_market_buy(symbol=symbol, quantity=qty)
    return market_order
    
def place_buy_order(symbol,MAX_TRADES,in_trade, BUY_METHOD):
    '''MAX_TRADES is maximum number of simulateneous trades
    in_trade - amount of coins that are in trade at the moment
    BUY_METHOD should be 'MARKET' or 'LIMIT'
    '''
    #amount_btc = 0.0058 #Our BTC amount for each trade (about $50 as of 01:40 29.02.2020)
    BuyPrice = get_buy_price(symbol, BUY_METHOD)
    #qty = amount_btc/np.float(BuyPrice)
    #qty = np.round(qty,0)
    qty = get_buy_amount(BuyPrice,MAX_TRADES,in_trade)
    prec = get_lot_precision(symbol)
    qty = float(truncate(qty, prec))
    
    if BUY_METHOD == 'MARKET':
        order = place_market_buy_order(symbol, qty)
        BuyPrice = weighted_avg(order['fills'], symbol)
        order['price'] = BuyPrice # !!! Check if market order API response returns weighted average price!!!!!!!
    else:
        order = client.order_limit_buy(symbol=symbol, quantity=qty, price=BuyPrice)
    
    with open('placed_buy_orders.dat', 'a') as f:
        now = datetime.now()
        time_curr = now.strftime("%d/%m/%y %H:%M:%S")
        am_btc = qty*float(BuyPrice)                        
        f.write("%s,%s,%s,%.3f,%.8f\n" % (time_curr, symbol, BuyPrice, qty, am_btc))

    return order, am_btc

def get_buy_amount(BuyPrice, MAX_TRADES, in_trade, asset='BTC'):
    '''Determine the amount of coin to buy. 
    MAX_TRADES - number of maximum simulteneous trades
    in_trade - number of coins tradind at the moment
    Takes total available amount in BTC and devides it by MAX_TRADES
    Algorithm: take 'free' balance of BTC and devide by (MAX_TRADES - in_trade)
    Note, in trade is computed before the coin is added to the trading list
    '''
    #amount_btc = float(client.get_asset_balance(asset)['free'])+float(client.get_asset_balance(asset)['locked'])
    amount_btc = float(client.get_asset_balance(asset)['free'])
    #in_trade = len(trading_coins)
    amount_btc = 0.1*amount_btc/(MAX_TRADES - in_trade)
    qty = amount_btc/float(BuyPrice)
    return qty

def place_oco_order(symbol, buy_price, take_profit = 0.015, stop_loss = 0.03):
    #from binance.enums import *
    buy_price = float(buy_price)
    #price = np.round( (1+take_profit)*buy_price, 8)
    #Get precision for price value of the symbol:
    price_prec = get_price_precision(symbol)
    price = np.round( (1+take_profit)*buy_price, price_prec)
    price = "%.8f" % price
    #print('price = ", price)
    #stop_price = np.round( (1-stop_loss/1.2)*buy_price, 8)
    stop_price = np.round( (1-stop_loss/1.2)*buy_price, price_prec)
    #stop_price = str(stop_price)
    stop_price = "%.8f" % stop_price
    #stop_limit_price = np.round( (1-stop_loss)*buy_price, 8)
    stop_limit_price = np.round( (1-stop_loss)*buy_price, price_prec)
    #stop_limit_price = str(stop_limit_price)
    stop_limit_price = "%.8f" % stop_limit_price
    qntity = float( client.get_asset_balance(asset=symbol[:-3])['free'] )
    #qntity = int(qntity//1) # take integer part from the balance
    prec = get_lot_precision(symbol)
    qntity = float(truncate(qntity, prec)) #It's important to truncate qunatity, NOT to round it, otherwise may be not enough balance
    #print("Place OCO order for %s for %s with stop price %s and limit price %s" % (symbol, price, stop_price, stop_limit_price) )    
    oco_order = client.create_oco_order(symbol=symbol,side=SIDE_SELL, stopLimitTimeInForce=TIME_IN_FORCE_GTC,\
                                        quantity=qntity, stopPrice=stop_price, price=price,\
                                        stopLimitPrice=stop_limit_price)
    now = datetime.now()
    time_curr = now.strftime("%d/%m/%y %H:%M:%S")
    with open('placed_oco_orders.dat', 'a') as f:
        f.write('%s,%s,%s,%s,%s\n' % (time_curr, symbol, price, stop_price, stop_limit_price))
    #sell_limit = client.order_limit_sell(symbol=symbol, quantity=qntity, price=price)
    #return sell_limit
    return oco_order

def check_order_status(order):
    try:
        order_status = client.get_order(symbol=order['symbol'], orderId=order['orderId'])
    except Exception as e:
        print("Warning! Didn't manage to check order status", order['symbol'])
        print(e)
        order_status = order
    #client.order_market_sell(symbol='XTZBTC',quantity=qntity)
    return order_status

def cancel_order(order):
    result = client.cancel_order(symbol=order['symbol'], orderId=order['orderId'])
    return result

def check_buy_order(order, trading_coins, coin):    
    '''order status response:
       {'symbol': 'WRXBTC',
         'orderId': 7895233,
         'orderListId': -1,
         'clientOrderId': 'beMu1tPLtTpJlEYjuXSzGR',
         'price': '0.00002132',
         'origQty': '1111.00000000',
         'executedQty': '1111.00000000',
         'cummulativeQuoteQty': '0.02368652',
         'status': 'FILLED',
         'timeInForce': 'GTC',
         'type': 'LIMIT',
         'side': 'BUY',
         'stopPrice': '0.00000000',
         'icebergQty': '0.00000000',
         'time': 1583698931057,
         'updateTime': 1583699099624,
         'isWorking': True,
         'origQuoteOrderQty': '0.00000000'}
    '''
    #!!!TODO: Improce Partial filling
    BUY_TIME_LIMIT = 10
    try:
        trading_coins[coin]['order'] = check_order_status(order)
    except Exception as e:
        print("Warning didn't manage to check order status! (Called from fuction 'check buy order')")
        print(e)
    status = trading_coins[coin]['order']['status']
    try:
        place_time = trading_coins[coin]['order']['time']/1000 #Time in Binance is given in milliseconds, convert to seconds here.
    except KeyError:
        place_time = trading_coins[coin]['order']['transactTime']/1000
    time_curr = time.time()
    elapsed_min = (time_curr - place_time)/60
    if elapsed_min > BUY_TIME_LIMIT:
       order = cancel_order(order)
       print("Buy order didn't fill in %d minutes" % BUY_TIME_LIMIT)
       executedQty = float(trading_coins[coin]['order']['executedQty'])
       if executedQty > 0:
           print("Order Partially Filled", coin)
           try:
               sell_leftovers(trading_coins[coin]['order']['symbol'])    
           except Exception as e:
               print("Warning! Order partially filled, but it was not sold!", coin, executedQty)
               print(e)
               del trading_coins[coin]
       status = "CANCELLED"
       del trading_coins[coin]
    if status == "FILLED":
        try:
            trading_coins[coin]['order'] = place_oco_order(order['symbol'], order['price'])
        except Exception as e:
            print("WARNING!!!! OCO order didn't place!", e)
        
    return status
        
def check_oco_order(order, trading_coins, coin, time_limit = 10):
    '''OCO order response:
    {'orderListId': 2804760,
         'contingencyType': 'OCO',
         'listStatusType': 'EXEC_STARTED',
         'listOrderStatus': 'EXECUTING',
         'listClientOrderId': '4vtKEEteQ4sMxeeJdedcFY',
         'transactionTime': 1583699174031,
         'symbol': 'WRXBTC',
         'orders': [{'symbol': 'WRXBTC',
           'orderId': 7897973,
           'clientOrderId': '6RTVZNxjMTBBt4UOL350Xm'},
          {'symbol': 'WRXBTC',
           'orderId': 7897974,
           'clientOrderId': 'jAdBKYiZNYTeGP2lB34P2t'}],
         'orderReports': [{'symbol': 'WRXBTC',
           'orderId': 7897973,
           'orderListId': 2804760,
           'clientOrderId': '6RTVZNxjMTBBt4UOL350Xm',
           'transactTime': 1583699174031,
           'price': '0.00002068',
           'origQty': '1109.00000000',
           'executedQty': '0.00000000',
           'cummulativeQuoteQty': '0.00000000',
           'status': 'NEW',
           'timeInForce': 'GTC',
           'type': 'STOP_LOSS_LIMIT',
           'side': 'SELL',
           'stopPrice': '0.00002079'},
          {'symbol': 'WRXBTC',
           'orderId': 7897974,
           'orderListId': 2804760,
           'clientOrderId': 'jAdBKYiZNYTeGP2lB34P2t',
           'transactTime': 1583699174031,
           'price': '0.00002164',
           'origQty': '1109.00000000',
           'executedQty': '0.00000000',
           'cummulativeQuoteQty': '0.00000000',
           'status': 'NEW',
           'timeInForce': 'GTC',
           'type': 'LIMIT_MAKER',
           'side': 'SELL'}]}
    '''
    #OCO_TIME_LIMIT = 240
    #trading_coins[coin]['order'] = check_order_status(order)
    #OCO order consists of 2 orders. Check their status separetely:
    #Grab order responses:
    stop_loss_order = trading_coins[coin]['order']['orderReports'][0]
    limit_maker = trading_coins[coin]['order']['orderReports'][1]
    #Check their status
    stop_loss_order = client.get_order(symbol = stop_loss_order['symbol'], orderId=stop_loss_order['orderId'])    
    limit_maker = client.get_order(symbol = limit_maker['symbol'], orderId=limit_maker['orderId'])    
    if (limit_maker['status'] == 'EXPIRED') or (stop_loss_order['status'] == 'EXPIRED'):       
        now = datetime.now().strftime("%d/%m/%y %H:%M:%S")
        print("OCO starts filling: %s, place time: %s, execute time: %s" % (coin, trading_coins[coin]['buy_time'], now) )
        limit_filled = (limit_maker['status'] == 'FILLED')
        loss_filled = (stop_loss_order['status'] == 'FILLED')
        if limit_filled :
            now = datetime.now().strftime("%d/%m/%y %H:%M:%S")
            print("OCO LIMIT filled: %s, place time: %s, execute time: %s" % (coin, trading_coins[coin]['buy_time'], now) )
            profit = 100*(float(limit_maker['price']) - float(trading_coins[coin]['buy_price']))/float(trading_coins[coin]['buy_price'])
            trading_coins[coin]['profit'] = profit
            trading_coins[coin]['g_profit'] = 0.01*profit*trading_coins[coin]['am_btc']
            try:
                save_filled_oco(trading_coins[coin], profit)
            except Exception as e:
                print("File save error: ", e)
                with open("save_filled_oco.dat", 'a') as f:
                    f.write(str(trading_coins[coin]))
                    f.write("\n")
            #del trading_coins[coin]
        if loss_filled:
            now = datetime.now().strftime("%d/%m/%y %H:%M:%S")
            print("OCO STOP LOSS filled: %s, place time: %s, execute time: %s" % (coin, trading_coins[coin]['buy_time'], now) )
            profit = 100*(float(stop_loss_order['price']) - float(trading_coins[coin]['buy_price']))/float(trading_coins[coin]['buy_price'])
            trading_coins[coin]['profit'] = profit
            trading_coins[coin]['g_profit'] = 0.01*profit*trading_coins[coin]['am_btc']
            try:
                save_filled_oco(trading_coins[coin], profit)
            except Exception as e:
                print("File save error: ", e)
                with open("save_filled_oco.dat", 'a') as f:
                    f.write(trading_coins[coin])
                    f.write("\n")
            #del trading_coins[coin]   
        if limit_filled or loss_filled :
            #Make sure that the coin has been completely sold:
            try:
                sell_leftovers(trading_coins[coin]['order']['symbol'])
            except Exception as e:
                print("Warning! Didn't manage to sell leftovers!", e)
            del trading_coins[coin]
            #print (trading_coins)
        return "FILLED"
    elif (limit_maker['status'] == 'CANCELED') or (stop_loss_order['status'] == 'CANCELED'):
        status = 'CANCELED'
        del trading_coins[coin]
    else:
        trading_coins[coin]['order']['orderReports'][0] = stop_loss_order
        trading_coins[coin]['order']['orderReports'][1] = limit_maker
        status = "EXECUTING"
        #last_price = 
        print("BUY: %s, SELL: %s" % (trading_coins[coin]['buy_price'],limit_maker['price']) )
        place_time = order['transactionTime']/1000
        time_curr = time.time()
        elapsed_min = (time_curr - place_time)/60
        if elapsed_min > time_limit and trading_coins[coin]['n_oco'] == 0 :
            #qty = float(stop_loss_order["origQty"])
            symbol = stop_loss_order['symbol']
            cancel_order(stop_loss_order)
            try: 
                trading_coins[coin]['order'] = place_oco_order(symbol,trading_coins[coin]['buy_price'], take_profit=0.015,stop_loss=0.03)
                trading_coins[coin]['n_oco'] += 1
                print("Placed 2nd OCO order", symbol)
            except Exception as e:
                print("Error occured while placing 2nd OCO order!", e)
                sell_leftovers(symbol)
                del trading_coins[coin]
            
            #market_sell = place_market_sell_order(symbol, qty)            
            #price = weighted_avg(market_sell['fills'], symbol)
            #profit = 100*(float(price) - float(trading_coins[coin]['buy_price']))/float(trading_coins[coin]['buy_price']) 
            #trading_coins[coin]['profit'] = profit
            #trading_coins[coin]['g_profit'] = 0.01*profit*trading_coins[coin]['am_btc']
            #status = 'SOLD'
            #save_filled_oco(trading_coins[coin], profit)
            #place market sell order
            #poka bez etogo oboidemsya

    return status

def sell_leftovers(symbol):
    coin = symbol[:-3] 
    info = client.get_symbol_info(symbol)
    stepSize = float(info['filters'][2]['stepSize'])
    leftover = float(client.get_asset_balance(coin)['free'])
    if leftover > stepSize:
        print("Not everything has been sold for some reason!")
        qty = leftover
        prec = get_lot_precision(symbol)
        qty = float(truncate(qty, prec))
        try:
            print("Place market sell order", symbol, qty)
            place_market_sell_order(symbol, qty)
            
            #del trading_coins[coin]
        except Exception as e:
            print("Warning! Market sell didn't execute!" , e, symbol )
        
    
#def follow_buy_order(order):
#    status = ''
#    start_time = time.time()
#    MAX_PENDING_TIME = 30 #minutes
#    while status != 'FILLED':
#        status = check_order_status(order)['status']
#        elapsed = (time.time() - start_time)/60
#        print(status)
#        if elapsed > MAX_PENDING_TIME : 
#            cancel_order(order)
#            print( "Cancel order, didn't fill in %.1f minutes" % elapsed )
#            break

#def follow_oco_order(order):
#    pass

def get_lot_precision(symbol):
    '''return precision of the lot for the order
    algorithm: take step size from the exchange, then split it by '.', after that find '1' in the part after '.'
    It returns the position of digit 1 after '.'. If it is not found then it returns -1.
    When we add 1 to the result it also handles the case when the digit 1 is not there (zero precision)
    Works for BTC trading pairs, but have to be tested on other trading assests'''
    info = client.get_symbol_info(symbol)
    stepSize = info['filters'][2]['stepSize']
    prec = stepSize.split('.')[-1].find('1') + 1
    return prec

def get_price_precision(symbol):
    '''return precision of the price for the order
    algorithm: take step size from the exchange, then split it by '.', after that find '1' in the part after '.'
    It returns the position of digit 1 after '.'. If it is not found then it returns -1.
    When we add 1 to the result it also handles the case when the digit 1 is not there (zero precision)
    Works for BTC trading pairs, but have to be tested on other trading assests'''
    info = client.get_symbol_info(symbol)
    stepSize = info['filters'][0]['tickSize']
    prec = stepSize.split('.')[-1].find('1') + 1
    return prec

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding
    Taken from: https://stackoverflow.com/questions/783897/truncating-floats-in-python
    '''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


   