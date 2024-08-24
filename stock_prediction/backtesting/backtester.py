import matplotlib.pyplot as plt
import numpy as np

from .simulator import Simulator
from stock_prediction.models.model_predictor import ModelPredictor
from stock_prediction.data_handling import StockUtils

class Backtester(Simulator):
    def __init__(self, model, scaler, testing_symbols, technical_indicators, params, capital, sell_perc=0.04, hold_till=5, stop_perc=0.005):
        super().__init__(capital)  # Initialize the simulator
        self.model_predictor = ModelPredictor(model, scaler, testing_symbols, technical_indicators, params)
        self.sell_perc = sell_perc
        self.hold_till = hold_till
        self.stop_perc = stop_perc
        self.portfolio_values = []  # To store portfolio value over time
        self.dates = []  # To store dates for portfolio value plotting
        self.trade_markers = []  # To store buy/sell markers on the portfolio value graph

    def run(self, show_graph=True):
        """
        Run the backtest on the provided data.
        """
        for symbol in self.model_predictor.testing_symbols:
            print(f"Running backtest for {symbol}")
            stock_utils = StockUtils(symbol=symbol)
            df_new_stock = stock_utils.get_indicators(self.model_predictor.technical_indicators, 
                                                      outputsize=self.model_predictor.outputsize, 
                                                      min_max_order=self.model_predictor.min_max_order)
            df_new_stock = df_new_stock.dropna(subset=self.model_predictor.technical_indicators)

            # Use ModelPredictor to evaluate the model and filter predictions
            df_predictions = self.model_predictor.evaluate_model(df_new_stock)
            df_predictions = self.model_predictor.filter_predictions(df_predictions)

            # Iterate over the DataFrame in reverse order to simulate the trading
            for index in df_predictions.index:
                row = df_predictions.loc[index]
                date = row['date']
                close_price = row['close']
                prediction = row['predicted_target']

                # Track portfolio value
                current_value = self.capital
                for stock, (buy_price, n_shares, _, _) in self.buy_orders.items():
                    current_value += n_shares * close_price
                self.portfolio_values.append(current_value)
                self.dates.append(date)

                # Buy signal: Buy if the prediction indicates a local minimum (0) and we don't already own the stock
                if prediction == 0 and symbol not in self.buy_orders:
                    self.buy(stock=symbol, buy_price=close_price, buy_date=date)
                    self.trade_markers.append(('B', date, current_value))  # Mark buy on portfolio value

                # Sell signal: Sell if the prediction indicates a local maximum (1)
                if symbol in self.buy_orders:
                    buy_date = self.buy_orders[symbol][3]  # Date of the buy
                    days_held = (date - buy_date).days
                    buy_price = self.buy_orders[symbol][0]

                    # Condition for taking profit at local maximum
                    if prediction == 1:
                        self.sell(stock=symbol, sell_price=close_price, n_shares_sell=self.buy_orders[symbol][1], sell_date=date)
                        self.trade_markers.append(('S', date, current_value))  # Mark sell on portfolio value

                    # Condition for holding period
                    elif days_held >= self.hold_till:
                        self.sell(stock=symbol, sell_price=close_price, n_shares_sell=self.buy_orders[symbol][1], sell_date=date)
                        self.trade_markers.append(('S', date, current_value))  # Mark sell on portfolio value

                    # Trailing Stop Loss: Protects against small dips
                    elif close_price <= buy_price * (1 - self.stop_perc):
                        self.sell(stock=symbol, sell_price=close_price, n_shares_sell=self.buy_orders[symbol][1], sell_date=date)
                        self.trade_markers.append(('S', date, current_value))  # Mark sell on portfolio value

        self.print_summary()

        # After the backtest, plot the results
        if show_graph:
            self.plot_results(df_new_stock, df_predictions, symbol)
        

    def plot_results(self, df_new_stock, df_predictions, symbol):
        """
        Plot the stock price, trades, true min/max, predicted min/max, and portfolio value.
        """
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # Plot stock price
        ax1.plot(df_new_stock['date'], df_new_stock['close'], color='black', label=f"{symbol} Daily Close Price")

        # Plot true min and max points
        ax1.scatter(df_new_stock['date'], df_new_stock['loc_min'], marker='x', color='red', label="True Minima", s=50)
        ax1.scatter(df_new_stock['date'], df_new_stock['loc_max'], marker='x', color='orange', label="True Maxima", s=50)

        # Plot predicted min and max points
        minima_points = df_predictions[df_predictions['predicted_target'] == 0]
        maxima_points = df_predictions[df_predictions['predicted_target'] == 1]
        ax1.scatter(minima_points['date'], minima_points['close'], marker='o', color='green', label="Predicted Minima", s=50)
        ax1.scatter(maxima_points['date'], maxima_points['close'], marker='o', color='blue', label="Predicted Maxima", s=50)

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Stock Price (USD)")
        ax1.legend(loc="upper left")

        # Create secondary y-axis for portfolio value
        ax2 = ax1.twinx()
        ax2.plot(self.dates, self.portfolio_values, color='blue', label='Portfolio Value', alpha=0.7)

        # Plot buy/sell markers on the portfolio value line
        for trade in self.trade_markers:
            label, trade_date, value = trade
            ax2.annotate(label, xy=(trade_date, value), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=10, color='black',
                        arrowprops=dict(facecolor='black', shrink=0.05))

        ax2.set_ylabel("Portfolio Value (USD)")
        ax2.legend(loc="upper right")

        plt.title(f"{symbol} Backtest with Trades, Predictions, and Portfolio Value")
        plt.show()

