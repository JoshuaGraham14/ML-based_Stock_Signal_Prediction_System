import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
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

                # Sell signal: Sell if the prediction indicates a local maximum (1) or if stop loss is triggered
                if symbol in self.buy_orders:
                    buy_price = self.buy_orders[symbol][0]

                    # Condition for taking profit at local maximum
                    if prediction == 1:
                        self.sell(stock=symbol, sell_price=close_price, n_shares_sell=self.buy_orders[symbol][1], sell_date=date)
                        self.trade_markers.append(('S', date, current_value))  # Mark sell on portfolio value

                    # Stop Loss: Protects against loss beyond stop_perc
                    elif close_price <= buy_price * (1 - self.stop_perc):
                        self.sell(stock=symbol, sell_price=close_price, n_shares_sell=self.buy_orders[symbol][1], sell_date=date)
                        self.trade_markers.append(('S', date, current_value))  # Mark sell on portfolio value

        self.print_summary()

        # After the backtest, plot the results
        if show_graph:
            self.plot_results(df_predictions, symbol)

    def plot_results(self, df_predictions, symbol):
        """
        Plot the stock price, trades, true min/max, predicted min/max, and portfolio value.
        """
        # Create the main figure and axis for the stock price
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # Base plot for stock price
        stock_line, = ax1.plot(df_predictions['date'], df_predictions['close'], color='black', label=f"{symbol} Daily Close Price", zorder=1)

        # True min and max points
        true_min_points = ax1.scatter(df_predictions['date'], df_predictions['loc_min'], marker='x', color='red', label="True Minima", s=50, zorder=2)
        true_max_points = ax1.scatter(df_predictions['date'], df_predictions['loc_max'], marker='x', color='orange', label="True Maxima", s=50, zorder=2)

        # Filter out neutral predictions
        minima_points = df_predictions[df_predictions['predicted_target'] == 0]
        maxima_points = df_predictions[df_predictions['predicted_target'] == 1]

        # Predicted min and max points
        predicted_minima_points = ax1.scatter(minima_points['date'], minima_points['close'], marker='o', color='green', label="Predicted Minima", s=50, zorder=3)
        predicted_maxima_points = ax1.scatter(maxima_points['date'], maxima_points['close'], marker='o', color='blue', label="Predicted Maxima", s=50, zorder=3)

        # Create secondary y-axis for portfolio value
        ax2 = ax1.twinx()

        # Initialize lists to store line objects and annotation objects for buy/sell markers
        trade_lines = []
        buy_annotations = []
        sell_annotations = []

        # Plot the portfolio value only between buys and sells with color coding for profit/loss
        buy_sell_pairs = [(self.trade_markers[i], self.trade_markers[i + 1])
                        for i in range(0, len(self.trade_markers) - 1, 2)]

        for buy, sell in buy_sell_pairs:
            buy_idx = self.dates.index(buy[1])
            sell_idx = self.dates.index(sell[1])

            # Determine if the trade was profitable
            trade_color = 'green' if self.portfolio_values[sell_idx] > self.portfolio_values[buy_idx] else 'red'

            # Highlight the stock price between buy and sell, overlayed on top of the base plot
            trade_line, = ax1.plot(self.dates[buy_idx:sell_idx+1], df_predictions['close'].iloc[buy_idx:sell_idx+1], color=trade_color, lw=2, zorder=4)
            trade_lines.append(trade_line)

            # Plot buy/sell markers on the stock price line
            buy_annotate = ax1.annotate('B', xy=(self.dates[buy_idx], df_predictions['close'].iloc[buy_idx]), xytext=(0, 10),
                                        textcoords='offset points', ha='center', fontsize=10, color='green', zorder=5,
                                        arrowprops=dict(facecolor='green', shrink=0.05, alpha=0.5, zorder=5))
            sell_annotate = ax1.annotate('S', xy=(self.dates[sell_idx], df_predictions['close'].iloc[sell_idx]), xytext=(0, 10),
                                        textcoords='offset points', ha='center', fontsize=10, color='red', zorder=5,
                                        arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.5, zorder=5))

            buy_annotations.append(buy_annotate)
            sell_annotations.append(sell_annotate)

        # Plot the portfolio value on the secondary axis
        portfolio_line, = ax2.plot(self.dates, self.portfolio_values, color='blue', label='Portfolio Value', alpha=0.7, zorder=2)

        # Add labels and legends
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Stock Price (USD)")
        ax1.legend(loc="upper left")
        ax2.set_ylabel("Portfolio Value (USD)")
        ax2.legend(loc="upper right")

        plt.title(f"{symbol} Backtest with Trades, Predictions, and Portfolio Value")

        # Create the CheckButtons for toggling elements
        rax = plt.axes([0.02, 0.5, 0.1, 0.15])  # Position the buttons
        check = CheckButtons(rax, ('Show Min/Max', 'Show Predicted', 'Show Buy/Sell Line', 'Show Buy/Sell Annotations', 'Show Portfolio'), 
                            (True, True, True, True, True))

        def toggle_visibility(label):
            if label == 'Show Min/Max':
                true_min_points.set_visible(not true_min_points.get_visible())
                true_max_points.set_visible(not true_max_points.get_visible())
            elif label == 'Show Predicted':
                predicted_minima_points.set_visible(not predicted_minima_points.get_visible())
                predicted_maxima_points.set_visible(not predicted_maxima_points.get_visible())
            elif label == 'Show Buy/Sell Line':
                for line in trade_lines:
                    line.set_visible(not line.get_visible())
            elif label == 'Show Buy/Sell Annotations':
                for annotation in buy_annotations + sell_annotations:
                    annotation.set_visible(not annotation.get_visible())
            elif label == 'Show Portfolio':
                portfolio_line.set_visible(not portfolio_line.get_visible())
            plt.draw()

        check.on_clicked(toggle_visibility)

        plt.show()