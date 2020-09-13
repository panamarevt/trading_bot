#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from xgboost import XGBClassifier

import pickle # to save and load pre-trained models


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#trades_df = pd.read_csv('buy_signals_backtest_ML_July-Aug2020.dat')
trades_df = pd.read_csv('buy_signals_backtest_ML_Jan2019-Aug2020.dat')
#trades_df.set_index('time_curr', inplace=True)
#trades_df.index=pd.to_datetime(trades_df.index)


# In[4]:


trades_df.columns


# In[5]:


profitability = [0 if item < 0 else 1 for item in trades_df['profit']]
trades_df['profitability'] = profitability
profitability_s15 = [0 if item < 0 else 1 for item in trades_df['profit_s15']]
trades_df['profitability_s15'] = profitability_s15


# Convert 'time_curr' to `datetime` object

# In[6]:


trades_df['time_curr'] = pd.to_datetime(trades_df['time_curr'])


# In[7]:


# Sort the dataframe by the date:
trades_df.sort_values(by=['time_curr'], inplace=True)


# In[8]:


trades_df.columns


# In[9]:


time_span = trades_df.time_curr.iloc[-1] - trades_df.time_curr.iloc[0]
print(f"Total profitability in {time_span}: ", trades_df['profitability'].sum()/len(trades_df['profitability']) )
print("total n trades: ", len(trades_df['profitability']))
print(f"Total profitability in {time_span} with 0.15 stop loss: ", trades_df['profitability_s15'].sum()/len(trades_df['profitability_s15']) )
print("total n trades: ", len(trades_df['profitability_s15']))


# In[10]:


trades_df['profitability']


# In[11]:


pd.set_option('max_columns',100)


# In[12]:


trades_df.dropna(axis=1, inplace=True)
#trades_df.fillna(0.0, inplace=True)


# In[13]:


# Remove the problematic coin that has been identified later
#trades_df = trades_df[trades_df.symbol != 'UMABTC']


# In[14]:


trades_df.head()


# ### Now we start preparing the dataset to train a model using various Machine Learning methods

# 1. Features:

# In[15]:


X = trades_df.drop(['symbol','profit',
       'elapsed', 'min_price', 'max_price', 'profit_s15', 'elapsed_s15',
       'min_price_s15', 'max_price_s15',  'profitability',
       'profitability_s15'], axis=1)

# For the 1st trial, let us also drop all categorical variables:
# X = trades_df.drop(['time_curr','profit',
#                     'pattern','origin','symbol','stoch_cond_5','candle_color',
#        'elapsed', 'min_price', 'max_price', 'profit_s15', 'elapsed_s15',
#        'min_price_s15', 'max_price_s15',  'profitability',
#        'profitability_s15'], axis=1)


# In[16]:


X['stoch_cond_5'] = X['stoch_cond_5'].astype('str')


# In[17]:


X['day'] = X.time_curr.dt.day.astype('uint8')
X['hour'] = X.time_curr.dt.hour.astype('uint8')
X['minute'] = X.time_curr.dt.minute.astype('uint8')


# In[18]:


X.drop(['time_curr'], axis=1, inplace=True)


# In[19]:


X.columns


# In[20]:


# Let's try to drop all prices and keep only the slopes and percentages: (No!)
## X = X.drop(['price', 'lower', 'upper', 'middle', 'ema_10'], axis=1)


# In[21]:


###for col in ['d_lower', 'd_upper', 'd_middle', 'd_ema_10' ]: X[col] = X[col]*1e7


# In[22]:


X.head()


# 2. Target:

# Here we predict whether the trade is going to be profitable or not.

# In[23]:


y = trades_df.profitability_s15
# what if we predict profitability for stop loss 0.3?
#y = trades_df.profitability


# In[24]:


y


# In[25]:


#test_size = 2000

# Make sure that there are no new symbols in the test dataset

# for item_val in X.symbol.tail(test_size).unique():
#     if item_val not in X.symbol.iloc[:-test_size].unique(): print(item_val)

    
    


# In[26]:


# X = X[X.symbol != 'UMABTC']
# y = y[X.symbol != 'UMABTC']


# In[27]:



# for item_val in X.symbol.tail(test_size).unique():
#     if item_val not in X.symbol.iloc[:-test_size].unique(): print(item_val)

 


# ### Train and test split

# In[28]:


# Random split
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=101)

# More realistic split is where we choose the latest deals as test data. Remove last rows:
test_size = 2000
# Remove last rows from the dataset
X_train, y_train = X.iloc[:-test_size], y.iloc[:-test_size]
X_val, y_val = X.tail(test_size), y.tail(test_size)


# ### Categorical encoding

# 3. Now it's time to make cathegorical encodings (or wait, let's do it later!) :-)

# In[29]:


# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()


# In[30]:


# categorical = ['pattern', 'origin', 'stoch_cond_5', 'candle_color']
# label = ['symbol']


# In[31]:


# for col in label:
#     X_train[col] = label_encoder.fit_transform(X_train[col])
#     X_val[col] = label_encoder.transform(X_val[col])


# In[32]:


# For now go with the simplest One-Hot encoding provided by pandas:
X_train, X_val = pd.get_dummies(X_train), pd.get_dummies(X_val)


# In[33]:


#X_train, X_val = preprocessing.scale(X_train), preprocessing.scale(X_val)


# In[34]:


X_train


# In[ ]:





# ### Define a function to compare different methods

# In[55]:


# function for comparing different approaches
def score_dataset(model, X_train, X_val, y_train, y_val, **kwargs):
    '''Trains a model, makes predictions. 
    Prints classification report
    Returns mean absolute error'''
    #Modified from: https://www.kaggle.com/alexisbcook/exercise-categorical-variables
    model.fit(X_train, y_train, **kwargs)
    preds = model.predict(X_val)
    print(classification_report(y_val,preds))
    return mean_absolute_error(y_val, preds)


# 4. Let's already try some methods:

# ### Random Forest

# In[36]:


random_forest_basic = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=1)


# In[37]:


print("Random Forest :")
score_dataset(random_forest_basic, X_train, X_val, y_train, y_val)


# ### Logistic regression

# In[38]:


logmodel_basic = LogisticRegression(solver='warn')


# In[39]:



print("Logistic regression :")
score_dataset(logmodel_basic, X_train, X_val, y_train, y_val)


# Check feature importance

# In[40]:


logmodel_basic.fit(X_train, y_train)


# In[41]:


importance = logmodel_basic.coef_[0]


# In[42]:


plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[43]:


len(importance)


# In[44]:


# try different solvers for logostic regression
# solvers = {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}
# for solver in solvers:
#     logmodel_solv = LogisticRegression(solver=solver)
#     print(f"Solver : {solver}")
#     score_dataset(logmodel_solv, X_train, X_val, y_train, y_val)


# ### XG Boost

# In[45]:


XGB_basic = XGBClassifier(n_estimators=1000, learning_rate=0.01, random_state=0)


# In[46]:


# XGB_basic.fit(X_train, y_train,
#              early_stopping_rounds=5, 
#              eval_set=[(X_val, y_val)], 
#              verbose=False)


# In[ ]:





# In[57]:


print("XG Boost :")
score_dataset(XGB_basic, X_train, X_val, y_train, y_val,
             early_stopping_rounds=10,
             eval_set=[(X_val, y_val)],
             verbose=False)


# In[ ]:





# In[48]:


#val_preds_XGB = XGB_basic.predict(X_val)


# In[49]:


#mae = mean_absolute_error(val_preds_XGB, y_val)


# In[50]:


#print(f"MAE for the XG Boost without categorical features: {mae}")


# ### The best model
# 
# According to the classification reports, the best model that predicts succesfull trasdes is happened to be logistic regression.
# Now, save this model using `pickle`

# In[51]:


# let's train the model again:
logmodel_basic = LogisticRegression(solver='warn')
logmodel_basic.fit(X_train, y_train)


# In[52]:


## Load pre-trained model:
# with open('logregression.pickle', 'wb') as f:
#     pickle.dump(logmodel_basic, f)


# In[53]:


# Open pre-trained model:
pickle_in = open('logregression.pickle', 'rb')
logmodel_loaded = pickle.load(pickle_in)


# In[54]:


# Check if loaded model shows the same results:
score_dataset(logmodel_loaded, X_train, X_val, y_train, y_val)


# <b> Note. </b> `XGBoost` shows better accuracy in general, but it better predicts loosing trades rather than succesfull. 

# In[ ]:




