# ML-based Stock Signal Prediction System
This repository contains a machine learning-based system designed to predict buy/sell signals by classifying stock price points as local minima or maxima. The system utilises historical stock data, technical indicators, and a logistic regression model to assist in making informed trading decisions.

## Features
- **Data Collection:** Fetches stock data using the Twelve Data API, using a custom API handler, with integrated caching and data processing capabilities.
- **Technical Indicators:** Computes and fetches over 10 technical indicators (e.g., EMA, RSI and n-day regressions) as features for model training. 
- **Stock Signal Classification:** Uses logistic regression to classify stock price points as local minima or maxima, generating actionable buy/sell signals.
- **Visualisation:** Provides tools to visualise stock trends and model predictions using Matplotlib and Seaborn, enhancing decision-making.
- **Extensibility:** Easily adaptable for different stocks and trading intervals.

## Example 

This image shows the predictions made by the machine learning model for GOOG (Google), where the machine learning model's predictions for buy/sell signals are overlaid on the actual stock price movement. 

The chart displays:
- **Black Line:** The daily closing prices.
- **Green Circles:** Predicted minima (buy signals)
- **Blue Circles:** Predicted maxima (sell signals)
- **Red Crosses:** True minima
- **Orange Crosses:** True maxima

<img width="1501" alt="Screenshot 2024-08-23 at 00 55 41" src="https://github.com/user-attachments/assets/4e96d5fa-1311-4b6c-a72f-7036b8155ced">

## Understanding My Project

### The Strategy

This project is centered around one of the most fundamental strategies in the stock market: **buying low and selling high**. The goal is to develop a model that predicts the low and high points of stock prices as accurately as possible. At a high level, the project can be broken down into two main steps:

1. **The ML Classification Model**: Identify price points as either local minima (buying opportunities), local maxima (selling opportunities), or neutral points.
2. **Backtesting and The Trading Strategy**: Use backtesting to devise a trading strategy that uses these predictions to automate tradesl; the objective being to maximise profits.

### The ML Classification Model

1. **Creating the Training Set:**

   - The training data is constructed using a variety of technical indicators, including n-day regressions, RSI, SMA, EMA, and normalised values. The model is trained on multiple inputted stocks, with the idea of finding an underlying relationship between all the stocks.
   - I then create the training set labels, by calculating the local minimums and maximums of the given stocks. For example, in the figure below of Google's stock chart, the green dots are the local minima and red dots are the local maxima.

<img width="1352" alt="Screenshot 2024-08-25 at 10 38 10" src="https://github.com/user-attachments/assets/667f8cae-8c4b-4d91-a87d-64fc6400c13b">

2. **Training the Model:**

   - I used a logistic regression model to classify the points.
   - The model is trained exclusively on points labeled as local minima or maxima.

   <insert matrix img>

3. **Using the Model for Prediction:**

   - The trained model is then used to predict whether an inputted stock's price on a given day is a local minimum or maximum.
   - Initially, I used a threshold of 0.5 - where points below 0.5 were classified as minima and points above 0.5 as maxima. However, this approach led to an excessive number of false positives, making the predictions completely unreliable:

   <insert 0.5 threshold img>

   - Therefore, to solve this issue, a much stricter threshold of 0.001 for minima and 0.999 for maxima was applied, significantly reducing the number of false positives and giving much more accurate results:

   <insert lower threshold img>

   - When the true local minima and maxima are overlaid on the predicted points, the model does a pretty good job of identifing these key turning points, despite a few false positives still remaining:

   <insert overlay true mins img>

### Backtesting and The Trading Strategy

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

<insert portfolio img>
<insert portfolio graph img>

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
