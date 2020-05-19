# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:25:17 2020

@author: Taras
"""
import time
import strategies


def try_func(func, step=10, duration=3600, *args, **kwargs):
    '''Allows to run any function 'func' with one argument arg inside the code in the body of the function.
    Trying to execute the function, if there is an exception, 
    assume that it is timeout error and call the fucntion func again 
    after every 'step' seconds during 'duration' period
    *args means that any number of positional arguments may be passed (can be empty)
    **kwargs means that any number of keyword arguments may be passed (can be empty)
    Note. We don't use conditions for the 'Connection Abort' exception because 
    there are many exceptions related to connection issues.
    '''
    try:
        resp = func(*args, **kwargs)
    except Exception as e:
#        if (type(e) == 'ReadTimeout') or (type(e) == 'ConnectTimeout') or type(e) == 'ConnectionError':
#            print("Time out error. Retry connection...")
        print("WARNING! Exception occured : ", func, e)    
        #print('Wait 10 sec. and retry...')
        start = time.time()
        done = start
        elapsed = done - start
        interval = duration #  seconds
        while elapsed < interval:
            print("Connecting....")
            try:
                resp = func(*args, **kwargs)
                print("Connection established")
                break
            except:
                done = time.time()
                elapsed = done - start
                if elapsed < step : time.sleep(step)
#        else:
#            print(func, '\n Unexpected Error:', e)
#            func(*args, **kwargs)
    return resp

if __name__=='__main__':
# Initialize C1M class instance:
    c1m = strategies.C1M()
    
    # Start the strategy:
    #c1m.c1m_flow(MAX_TRADES=4, DEPOSIT_FRACTION=0.25, TRADE_TYPE='PAPER')
    
    # Use try_func here to avoid some connection issues:
    # TODO! Better to add try and except EACH TIME we connect to an exchange!
    try_func(c1m.c1m_flow, step=10, duration=3600, MAX_TRADES=4, DEPOSIT_FRACTION=0.25, TRADE_TYPE='PAPER')


