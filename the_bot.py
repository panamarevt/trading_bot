# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:25:17 2020

@author: Taras
"""

import strategies

# Initialize C1M class instance:
c1m = strategies.C1M()

# Start the strategy:
c1m.c1m_flow(MAX_TRADES=0, DEPOSIT_FRACTION=0.01)