# backtester.py
import pandas as pd
from .simulator import Simulator
from stock_prediction.data_handling import StockUtils

class Backtester(Simulator):
    def __init__(self, model, scaler, testing_symbols, technical_indicators, params, capital, sell_perc=0.04, hold_till=5, stop_perc=0.005):
        super().__init__(capital)  # Initialize the simulator
        self.model = model
        self.scaler = scaler
        self.testing_symbols = testing_symbols
        self.technical_indicators = technical_indicators
        self.params = params
        self.sell_perc = sell_perc
        self.hold_till = hold_till
        self.stop_perc = stop_perc

    def run(self):
        """
        Run the backtest on the provided data.
        """
        for symbol in self.testing_symbols:
            print(f"Running backtest for {symbol}")
            stock_utils = StockUtils(symbol=symbol)
            df_new_stock = stock_utils.get_indicators(self.technical_indicators, outputsize=self.params['outputsize'], min_max_order=self.params['min_max_order'])
            df_new_stock = df_new_stock.dropna(subset=self.technical_indicators)
            df_new_stock['date'] = pd.to_datetime(df_new_stock['date'], format='%d/%m/%Y')

            # Scale the features
            X = self.scaler.transform(df_new_stock[self.technical_indicators])
            y_pred_proba = self.model.predict_proba(X)[:, 1]

            # Iterate over the DataFrame in reverse order
            for index in reversed(df_new_stock.index):
                row = df_new_stock.loc[index]
                date = row['date']
                close_price = row['close']
                prediction = y_pred_proba[index]

                # Buy signal: Buy if the prediction probability is below min_threshold and we don't already own the stock
                if prediction < self.params['min_threshold'] and symbol not in self.buy_orders:
                    self.buy(stock=symbol, buy_price=close_price, buy_date=date)

                # Sell signal: Sell if the prediction probability is above max_threshold
                if symbol in self.buy_orders:
                    buy_date = self.buy_orders[symbol][3]  # Date of the buy
                    days_held = (date - buy_date).days
                    buy_price = self.buy_orders[symbol][0]

                    # Condition for taking profit at local maximum
                    if prediction > self.params['max_threshold']:
                        self.sell(stock=symbol, sell_price=close_price, n_shares_sell=self.buy_orders[symbol][1], sell_date=date)

                    # Condition for holding period
                    elif days_held >= self.hold_till:
                        self.sell(stock=symbol, sell_price=close_price, n_shares_sell=self.buy_orders[symbol][1], sell_date=date)

                    # Trailing Stop Loss: Protects against small dips
                    elif close_price <= buy_price * (1 - self.stop_perc):
                        self.sell(stock=symbol, sell_price=close_price, n_shares_sell=self.buy_orders[symbol][1], sell_date=date)

        # Print final summary of the backtest
        self.print_summary()
