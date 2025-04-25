# Cryptocurrency Trading Bot Project

This project contains a collection of Python scripts and Jupyter notebooks designed for cryptocurrency trading, backtesting, and machine learning model training. It includes modules for interacting with the Binance API, implementing trading strategies, and evaluating model performance.

## Project Overview

The primary goal of this project is to develop and refine algorithmic trading strategies for the cryptocurrency market. It incorporates functionalities for backtesting historical data, training machine learning models to predict market movements, and ultimately, executing trades in real-time. The project is organized into several modules, each focusing on different aspects of the trading process.

## File Descriptions

### Core Logic and Utilities

-   **Indicators.py:** Defines various technical indicators used in trading strategies (e.g., moving averages, RSI).
-   **binance_bar_extractor.py:** Extracts historical bar data from the Binance API.
-   **binance_endpoints.py:** Handles communication with the Binance API for fetching market data and placing orders.
-   **strategies.py:** Contains the main classes for trading strategies.
-   **supertrend.py:** Implements the Supertrend indicator.
-   **supertrend_live.py:** Supertrend indicator live implementation.

### Backtesting

-   **backtest_bollinger.py:** Backtests a trading strategy based on Bollinger Bands.
-   **backtest_for_ML.py:** Backtesting module, it is adapted for machine learning.
-   **backtest_futures.py:** Backtesting for futures trading.
-   **backtest_general.py:** General framework for backtesting various trading strategies.
-   **backtest_v4.py:** Advanced version of the backtesting module.
-   **backtest_volume.py:** Strategy backtest based on volume.
-   **c1m_alerts_official_backtest.py:** Backtesting of the C1M alert strategy.

### Real-Time Trading

-   **realtime_bot_C1M_v1.py:** Implements a real-time trading bot using the C1M strategy.
-   **realtime_bot_volume_v1.py:** Real-time trading bot that uses volume strategies.
-   **the_bot.py:** Main script that contains the implementation of a trading bot.
- **two-sigma-live.py:**  Real time strategy similar to two-sigma but with enhancements.
- **two-sigma.py:** Implementation of a two-sigma trading strategy.

### Evaluation

-   **evaluate.py:** Provides tools for evaluating the performance of trading strategies.

### Machine Learning

-   **ML-training/C1M_ML_analysis-volume.ipynb:** Jupyter notebook for ML analysis, using volume as one of the main features.
-   **ML-training/C1M_ML_analysis.ipynb:** Jupyter notebook for training ML models, based on the C1M alert strategy.
-   **ML-training/C1M_ML_analysis.py:** Python script to perform machine learning analysis, based on the C1M strategy.
- **ML-training/C1M_ML_analysis_5methods.ipynb:** Jupyter notebook that makes an analysis with 5 different ML techniques.
-   **ML-training/\*.pickle:** Trained machine learning models (Gradient Boosting, Logistic Regression, Multi-Layer Perceptron, Random Forest, Support Vector Machine, Light Gradient Boosting Machine).
-   **ML-training/logregression.pickle:**  Logistic regression model.
-   **ML-training/logregression_vol.pickle:** Logistic regression model that includes volume data.
- **ML-training/xgboost_vol.pickle:**  XGboost model that includes volume data.
-   **ML-training/img/\*:** Images used for the results in the notebooks.

### Miscellaneous

-   **.vscode/settings.json:** Settings for the Visual Studio Code environment.

This project is structured to support both systematic backtesting and real-time trading, with an emphasis on utilizing machine learning to enhance strategy performance.