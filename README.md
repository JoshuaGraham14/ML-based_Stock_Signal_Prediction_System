# ML-based Stock Signal Prediction System
This repository contains a machine learning-based system designed to predict buy/sell signals by classifying stock price points as local minima or maxima. The system utilises historical stock data, technical indicators, and a logistic regression model to assist in making informed trading decisions.

## Features
- **Data Collection:** Fetches stock data using the Twelve Data API, using a custom API handler, with integrated caching and data processing capabilities.
- **Technical Indicators:** Computes and fetches over 10 technical indicators (e.g., EMA, RSI and n-day regressions) as features for model training. 
- **Stock Signal Classification:** Uses logistic regression to classify stock price points as local minima or maxima, generating actionable buy/sell signals.
- **Visualisation:** Provides tools to visualise stock trends and model predictions using Matplotlib and Seaborn, enhancing decision-making.
- **Extensibility:** Easily adaptable for different stocks and trading intervals.

## Example Usage

This image shows the predictions made by the machine learning model for GOOG (Google), where the machine learning model's predictions for buy/sell signals are overlaid on the actual stock price movement. 

The chart displays:
- **Black Line:** The daily closing prices.
- **Green Circles:** Predicted minima (buy signals)
- **Blue Circles:** Predicted maxima (sell signals)
- **Red Crosses:** True minima
- **Orange Crosses:** True maxima

<img width="1501" alt="Screenshot 2024-08-23 at 00 55 41" src="https://github.com/user-attachments/assets/4e96d5fa-1311-4b6c-a72f-7036b8155ced">


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

1. **Fetch and Analyze Stock Data:**
   - Use the `main.py` script to fetch stock data and calculate technical indicators.

2. **Train and Test the Model:**
   - Utilise the `StockPredictionModel` class to train a logistic regression model on selected stock symbols.

3. **Make Predictions:**
   - Apply the trained model to predict future minima and maxima points, guiding trading decisions.

## Project Structure
- `main.py`: Entry point of the application, managing data fetching and invoking the model.
- `StockPredictionModel.py`: Contains the logistic regression model for classifying stock signals.
- `api_handler.py`: Handles API calls to fetch stock data with error handling and caching.
- `stock_utils.py`: Provides utilities for processing stock data and computing technical indicators.
