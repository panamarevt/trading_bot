# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 05:12:14 2020

@author: Taras
"""
import numpy as np

# Technical indicators from TA library
# https://technical-analysis-library-in-python.readthedocs.io/en/latest/
# https://github.com/bukosabino/ta
import ta
import pandas_ta as pd_ta

def get_BBands(close):
    '''Compute Bollinger Bands'''
    # Create instance of the class
    BBands = ta.volatility.BollingerBands(close)
    
    middle = BBands.bollinger_mavg()
    upper = BBands.bollinger_hband()
    lower = BBands.bollinger_lband()
    width = BBands.bollinger_wband()
    
    return lower, middle, upper, width

def check_lower_BB(close, low, lower):
    '''Check for lower BB breach'''
    lower_BB_signal = ((close.iloc[-1] < lower.iloc[-1] or low.iloc[-1] < lower.iloc[-1]) \
                    or (close.iloc[-2] < lower.iloc[-2] or low.iloc[-2] < lower.iloc[-2]) \
                    or (close.iloc[-3] < lower.iloc[-3] or low.iloc[-3] < lower.iloc[-3]))
    return lower_BB_signal

def check_middle_BB(Open, close, low, middle):
    '''Check for middle BB breach'''
    middle_BB_signal = (  ( ( (close.iloc[-2] < middle.iloc[-2]) and (Open.iloc[-2] > middle.iloc[-2]) ) \
                           or ( (low.iloc[-2] < middle.iloc[-2]) and (Open.iloc[-2] > middle.iloc[-2]) \
                           and (close.iloc[-2] > middle.iloc[-2]) ) )\
                           and (close.iloc[-1] > middle.iloc[-1])  )
    return  middle_BB_signal   

def get_StochRSI(close):
    # RSI
    RSI = ta.momentum.RSIIndicator(close).rsi()         
    # Stochastic RSI 
    stoch_RSI = ta.momentum.StochasticOscillator(RSI,RSI,RSI)
    fastk =  stoch_RSI.stoch_signal() 
    #slow_d is 3-day SMA of fast_k
    slowd = ta.volatility.bollinger_mavg(fastk, n=3)     
    
    return fastk, slowd

def check_stochRSI_signal(fastk, slowd, kmin=10, kmax=30):
    '''Returns True if %K > %D, both go up and kmin < %K < kmax'''
    stochRSI_signal = (fastk.iloc[-1] > slowd.iloc[-1]) and (fastk.iloc[-1] > fastk.iloc[-2]) and \
                      (slowd.iloc[-1] > slowd.iloc[-2]) and (kmin < fastk.iloc[-1] < kmax)

    return stochRSI_signal


def EMA(close, period):
    '''Returns EMA of given period by accessing EMA from TA library'''
    return ta.trend.ema(close, period)

def supertrend_signal(op, hi, lo, cl, period, mult, cond='touch', side='long'):
    '''Check if there is a BUY/SELL signal for a SuperTrend strategy
    returns True if the signal conditions are met, otherwise False.
    ------------------------
    parameters:
    op,hi,lo,cl : (pd.Series) - open, high, low and close prices
    period : (int) - time period to compute average true range in the supertrend indicator
    mult : (float, but usually an integer) - multiplier to use in the computation for the supertrend indicator     
    cond - condition for the signal 
         = 'touch' - if last candle touched supertrend line, buy on open of current candle
         = 'green' - current candle closed green (red for short) after supertrend line, previous candle touched the supertrend line
    side - long or short strategy
         = 'long' - to check for LONG signal
         = 'short' to check for SHORT signal'''
    super_trend = pd_ta.supertrend(hi,lo,cl,period,mult)
    uptrend = super_trend.iloc[-1,1] == 1 # Second column (1) shows the trend direction: 1 - uptrend, -1 - downtrend
    super_trend = super_trend.iloc[-1,0] # take only the latest (current) value; first (0) column is the numerical value of the trend      
    print(f"SuperTrend value: {super_trend}, uptrend: {uptrend}")
    signal = False
    if cond == 'touch':
        if (side=='long') and uptrend:
            signal = lo.iloc[-1] < super_trend and cl.iloc[-1] > super_trend
        if (side=='short') and (not uptrend):
            signal = hi.iloc[-1] > super_trend and cl.iloc[-1] < super_trend
    if cond == 'green':
        if (side=='long') and uptrend:
            signal = (lo.iloc[-1] < super_trend and cl.iloc[-1] > super_trend) \
                and (cl.iloc[-1] > op.iloc[-1] and cl.iloc[-1] > super_trend)
        if (side=='short') and (not uptrend):
            signal = (hi.iloc[-1] > super_trend and cl.iloc[-1] < super_trend) \
                and (cl.iloc[-1] < op.iloc[-1] and cl.iloc[-1] < super_trend)                
    return signal


##### Candle patterns:

class Candle:
    def __init__(self, Open, High, Low, Close):
        ''' Initialize candlestick basic parameters: color, high_shadow, low_shadow and body.
        Note that when Open = Close we treat the candle as green.'''
        self.green = Close >= Open
        self.red = Close < Open
        self.color = 'green' if self.green else 'red'
        self.high_shadow = High - Close if self.green else High - Open
        self.low_shadow = Open - Low if self.green else Close - Low
        self.body = np.abs(Open - Close)
        
    def is_hammer(self, low_to_body = 2.0, high_to_body = 1.5):
        '''We define here that the candle is called 'hammer' if its low to body ratio >= 'low_to_body'
        and its high to body ratio <= 'high_to_body'. Body can not be zero.
        In most cases we used 2.0 and 1.5 for these values correspondingly '''
        if self.body == 0:
            return False
        if self.low_shadow/self.body >= low_to_body and self.high_shadow/self.body <= high_to_body :
            return True
        else: 
            return False
        
    def is_doji(self):
        doji = False
        if self.body == 0: 
            return True
        elif self.high_shadow != 0:
            if (0.9 < self.low_shadow/self.high_shadow < 1.1) and (self.low_shadow/self.body > 5):
                return True # standard doji, but body is not zero
            if (self.low_shadow/self.high_shadow > 5) and (self.low_shadow/self.body > 5) : 
                return True  # Dragonfly doji
        elif self.low_shadow != 0:
            if (self.high_shadow/self.low_shadow > 5) and (self.high_shadow/self.body > 5):
                return True # gravestone doji
        else:
            return False
        return doji
              
     
        

def candle_params(Open, High, Low, Close):
    #import math
    green = Close >= Open
    red = Close < Open
    #doji = Close == Open
    color = 'green'
    if green:
        low_shadow = Open - Low
        high_shadow = High - Close
        color = 'green'
    #if red:
    else:
        low_shadow = Close - Low
        high_shadow = High - Open
        color = 'red'
    body = np.abs(Close - Open)
    return body, low_shadow, high_shadow, color

def is_hammer(body, low_shadow, high_shadow):
    hammer = False
    if body == 0: hammer = False
    elif low_shadow/body >= 2.0 and high_shadow/body <= 1.5:
        hammer = True
    else: hammer = False
    return hammer

def is_doji(body, low_shadow, high_shadow):
    doji = False
    if body == 0:  
        doji = True # Any type of doji
    elif high_shadow != 0:
        if (0.9 < low_shadow/high_shadow < 1.1) and (low_shadow/body > 5):
            doji = True # standard doji, but body is not zero
        if (low_shadow/high_shadow > 5) and (low_shadow/body > 5) : doji = True  # Dragonfly doji
    elif low_shadow != 0:
        if (high_shadow/low_shadow > 5) and (high_shadow/body > 5): doji = True # gravestone doji
    return doji
    

def candle_pattern(last, before_last):
    '''Start counting from last candle
    '''
    #import math
    pattern = {'last':'no', 'blast':'no', 'total':'no'}
    Open_l, High_l, Low_l, Close_l = [item for item in last]
    Open_bl, High_bl, Low_bl, Close_bl = [item for item in before_last]
    Body_l, high_shadow_l, low_shadow_l, color_l = candle_params(Open_l, High_l, Low_l, Close_l)
    Body_bl, high_shadow_bl, low_shadow_bl, color_bl = candle_params(Open_bl, High_bl, Low_bl, Close_bl)
    if is_doji(Body_l, high_shadow_l, low_shadow_l):
        pattern['last'] = 'Doji'
        return pattern['last']
    elif is_hammer(Body_l, high_shadow_l, low_shadow_l):
        pattern['last'] = 'Hammer'
        return pattern['last']
    else: pattern['last'] = 'no'
    
    if is_doji(Body_bl, high_shadow_bl, low_shadow_bl):
        pattern['blast'] = 'Doji'
        return pattern['blast']
    elif is_hammer(Body_bl, high_shadow_bl, low_shadow_bl):
        pattern['blast'] = 'Hammer'
        return pattern['blast']
    else: pattern['blast'] = 'no'
    
    # Check for Bullish engulfing and Harami
    if color_l == 'green' and color_bl == 'red':
        Bullish_eng = Open_l < Close_bl and Close_l > Open_bl
        Harami = Open_l > Close_bl and Close_l < Open_bl
    else: 
        Bullish_eng = False
        Harami = False
    if Bullish_eng: pattern['total'] = 'Bullish eng.'
    if Harami: pattern['total'] = 'Harami'
    
    if pattern['total'] != 'no': 
        pattern = pattern['total']
    elif pattern['last'] != 'no' :
        pattern = pattern['last']
    else:
        pattern = 'no'
    
    return pattern

def check_candle_patterns(Open, high, low, close):
    '''Check if there were some candle patterns. 
    1st check the last candle, if no patterns, check the one before.
    Note. Current candle (that has not formed yet) is NOT taken into account'''
    
    last = [Open.iloc[-2], high.iloc[-2], low.iloc[-2], close.iloc[-2]]
    before_last = [Open.iloc[-3], high.iloc[-3], low.iloc[-3], close.iloc[-3]]
    before_before_last = [Open.iloc[-4], high.iloc[-4], low.iloc[-4], close.iloc[-4]]
    pattern = candle_pattern(last, before_last)
    if pattern == 'no': pattern = candle_pattern(before_last, before_before_last)
    
    return pattern
    
    
    
    