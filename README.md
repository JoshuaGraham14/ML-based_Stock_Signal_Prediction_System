# ML-based Stock Signal Prediction System
This repository contains a machine learning-based system designed to predict buy/sell signals by classifying stock price points as local minima or maxima. The system utilises historical stock data, technical indicators, and a logistic regression model to assist in making informed trading decisions.

<img width="1441" alt="Screenshot 2024-08-26 at 00 10 14" src="https://github.com/user-attachments/assets/86864c23-19bf-4370-a509-973cb7a11e26">
<img width="500" alt="Screenshot 2024-08-26 at 00 09 05" src="https://github.com/user-attachments/assets/5dfee32d-39d2-45cf-8cb5-0092e5786952">

## Features
- **Data Collection:** Fetches stock data using the Twelve Data API, using a custom API handler, with integrated caching and data processing capabilities.
- **Technical Indicators:** Computes and fetches over 10 technical indicators (e.g., EMA, RSI and n-day regressions) as features for model training. 
- **Stock Signal Classification:** Uses logistic regression to classify stock price points as local minima or maxima, generating actionable buy/sell signals.
- **Visualisation:** Provides tools to visualise stock trends and model predictions using Matplotlib and Seaborn, enhancing decision-making.
- **Extensibility:** Easily adaptable for different stocks and trading intervals.

## Understanding My Project

### - *The Strategy*

This project is centered around one of the most fundamental strategies in the stock market: **buying low and selling high**. The goal is to develop a model that predicts the low and high points of stock prices as accurately as possible. At a high level, the project can be broken down into two main steps:

1. **The ML Classification Model**: Identify price points as either local minima (buying opportunities), local maxima (selling opportunities), or neutral points.
2. **Backtesting and The Trading Strategy**: Use backtesting to devise a trading strategy that uses these predictions to automate trades; the objective being to maximise profits.

### - *The ML Classification Model*

1. **Creating the Training Set:**

   - The training data is constructed using a variety of technical indicators, including n-day regressions, RSI, SMA, EMA, and normalised values. The model is trained on multiple inputted stocks, with the idea of finding an underlying relationship between all the stocks.
   - I then create the training set labels, by calculating the local minimums and maximums of the given stocks. For example, in the figure below of Google's stock chart, the green dots are the local minima and red dots are the local maxima:

<img width="1352" alt="Screenshot 2024-08-25 at 10 38 10" src="https://github.com/user-attachments/assets/667f8cae-8c4b-4d91-a87d-64fc6400c13b">

2. **Training the Model:**

   - I used a logistic regression model to classify the points.
   - The model is trained exclusively on points labeled as local minima or maxima. This is the confusion matrix I get after using an 80/20 split to evaluate my classification:

<img width="388" alt="Screenshot 2024-08-26 at 00 07 23" src="https://github.com/user-attachments/assets/80941fdc-d08d-4c15-a581-3a96bbd09ff7">

3. **Using the Model for Prediction:**

   - The trained model is then used to predict whether an inputted stock's price on a given day is a local minimum or maximum.
   - Initially, I used a threshold of 0.5 - where points below 0.5 were classified as minima and points above 0.5 as maxima. However, this approach led to an excessive number of false positives, making the predictions completely unreliable:

<img width="1491" alt="Screenshot 2024-08-25 at 10 40 50" src="https://github.com/user-attachments/assets/64eedb0a-9a91-478d-ab0b-8ab47b37cf03">

   - Therefore, to solve this issue, a much stricter threshold of 0.001 for minima and 0.999 for maxima was applied, significantly reducing the number of false positives. As you can see, these results look much better!:

<img width="1490" alt="Screenshot 2024-08-25 at 10 42 06" src="https://github.com/user-attachments/assets/46a4cd29-9d1b-4cb0-afeb-9e3adcc5b0c5">

   - When the true local minima and maxima are overlaid on the predicted points, as you can see, the model does a pretty good job of identifing these key turning points, despite a few false positives still remaining:

<img width="1493" alt="Screenshot 2024-08-25 at 10 39 37" src="https://github.com/user-attachments/assets/34dc2752-48d2-44a0-bce8-bb1c554bc8b3">

### - *Backtesting and The Trading Strategy*

I built a stock simulator and backtesting script to use the model's predictions to making buy and sell decisions. The backtesting simulation has three parameters:

1. **min_threshold**: Defines the threshold for identifying buy opportunities.
2. **max_threshold**: Defines the threshold for identifying sell opportunities.
3. **perc_lost**: Determines the tolerance level for losses before selling a stock.

The way the trading strategy logic works is as follows:

- **Buying**: A stock is purchased when the model predicts a minimum.
- **Selling**: A stock is sold if:
  - The model predicts a maximum (i.e., a sell signal).
  - Or the portfolio's loss exceeds the `perc_lost` threshold.

In testing, starting with an initial capital of $10,000, the parameters (0.001, 0.999, 0.005) were found to be most effective. A lower `perc_lost` value ensures that buying/selling signals which were predicted incorrectly are quickly exited, thereby minimising losses.

<img width="708" alt="Screenshot 2024-08-26 at 00 09 05" src="https://github.com/user-attachments/assets/5dfee32d-39d2-45cf-8cb5-0092e5786952">

<img width="1441" alt="Screenshot 2024-08-26 at 00 10 14" src="https://github.com/user-attachments/assets/86864c23-19bf-4370-a509-973cb7a11e26">

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-prediction-system.git
   ```
   
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your API key:
   - Replace the `api_key` property with your Twelve Data API key in `config.json`:
     
   ```json
   {
    "api_key": "your_twelvedata_api_key"
   }
   ```
   
## Usage

### Prerequisites
Ensure you have installed all necessary dependencies as listed in `requirements.txt`. 

### Step 1: Fetch Latest Stock Data (Optional)

Before running the main prediction pipeline or backtesting, you may want to fetch the latest stock data for the day. This step is optional because the system automatically fetches live data via the API if the requested stock data is not available locally.

To fetch the latest stock data, run the following script:

```bash
python stock_prediction/scripts/fetch_data.py
```

### Step 2: Running the Prediction Pipeline

The main functionality of the project is executed through the `StockPredictorPipeline` class. You have the option to either run the prediction model to identify potential buy/sell signals or perform backtesting to evaluate the trading strategy's profitability.

#### Running the Prediction Model

To run the model and get predictions for potential buy and sell signals, use the `run()` method:

```python
pipeline.run()
```

This will process the stock data, train the model, and generate predictions.

#### Running the Backtesting

To simulate trading based on the model's predictions and evaluate the strategy, use the `run_backtest()` method:

```python
pipeline.run_backtest(initial_capital=10000, sell_perc=0.04, hold_till=5, stop_perc=0.0005)
```

This will backtest the strategy with an initial capital of $10,000, a sell percentage of 4%, a hold period of 5 days, and a stop-loss percentage of 0.05%.

### Customizing the Parameters

In the `main.py` file, you can customize the following parameters to fit your needs:

- **training_symbols**: List of stock symbols to be used for training the model.
- **testing_symbols**: List of stock symbols to be used for testing or backtesting.
- **technical_indicators**: List of technical indicators to be used in the model.
- **params**: Dictionary of model parameters including:
  - `outputsize`: Number of data points to fetch.
  - `min_max_order`: Order of minimum/maximum points calculation.
  - `min_threshold`: Threshold for classifying minima.
  - `max_threshold`: Threshold for classifying maxima.
  - `window_size`: Window size for smoothing predictions.

### Example of Running the Main Script

The following is an example of how the `main.py` script is structured:

```python
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the parent directory to sys.path

from stock_prediction.stock_predictor_pipeline import StockPredictorPipeline

def main():
    training_symbols = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'TSLA','BRK.B',
        'META', 'UNH', 'XOM', 'LLY', 'JPM', 'JNJ', 'PG', 'MA', 'AVGO', 'HD',
        'CVX', 'MRK', 'COST', 'PEP', 'NFLX'
    ]
    testing_symbols = ['GOOG']
    technical_indicators = [
        "normalized_value", "2_reg", "3_reg", "5_reg", "10_reg", "20_reg", 
        "50_reg", "adx", "ema", "sma", "rsi", "percent_b"
    ]

    params = {
        'outputsize': 1200,
        'min_max_order': 5,
        'min_threshold': 0.0001,
        'max_threshold': 0.9999,
        'window_size': 10
    }

    pipeline = StockPredictorPipeline(training_symbols, testing_symbols, technical_indicators, params)
    
    # Choose one of the following:
    # To run the model and get predictions:
    # pipeline.run()
    
    # To run backtesting and evaluate the strategy:
    pipeline.run_backtest(initial_capital=10000, sell_perc=0.04, hold_till=5, stop_perc=0.0005)

if __name__ == "__main__":
    main()
```

In this script, the `StockPredictorPipeline` class is initialized with a set of training and testing symbols, technical indicators, and model parameters. You can choose to either run the model for predictions using `pipeline.run()` or perform backtesting using `pipeline.run_backtest()`.

### Additional Notes

- **Live Data Fetching**: If you skip the optional data-fetching step, the system will automatically pull live data for any unseen stock symbols during runtime.
- **Visualization**: The pipeline includes functionality to visualize predictions and backtesting results using Matplotlib, making it easier to analyze the model’s performance visually.
