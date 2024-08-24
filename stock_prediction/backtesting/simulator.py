import numpy as np
import math
import pandas as pd

class Simulator:
    def __init__(self, capital):
        """
        initialize the simulator with the capital
        """
        self.capital = capital
        self.initial_capital = capital #keep a copy of the initial capital
        self.total_gain = 0
        self.buy_orders = {}
        self.history = []
        #create a pandas df to save history
        cols = ['stock', 'buy_price', 'n_shares', 'sell_price', 'net_gain', 'buy_date', 'sell_date']
        self.history_df = pd.DataFrame(columns = cols)
           
    def buy(self, stock, buy_price, buy_date):
        """
        function takes buy price and the number of shares and buy the stock
        """
        #calculate the procedure
        n_shares = self.buy_percentage(buy_price)
        self.capital = self.capital - buy_price * n_shares
        self.buy_orders[stock] = [buy_price, n_shares, buy_price * n_shares, buy_date]

    def sell(self, stock, sell_price, n_shares_sell, sell_date):
        """
        function to sell stock given the stock name and number of shares
        """
        buy_price, n_shares, _, buy_date = self.buy_orders[stock]
        sell_amount = sell_price * (n_shares_sell)

        self.capital = self.capital + sell_amount

        if (n_shares - n_shares_sell) == 0: #if sold all
            self.history.append([stock, buy_price, n_shares, sell_price, buy_date, sell_date])
            del self.buy_orders[stock]
        else:
            n_shares = n_shares - n_shares_sell
            self.buy_orders[stock][1] = n_shares
            self.buy_orders[stock][2] = buy_price * n_shares

    def buy_percentage(self, buy_price, buy_perc = 1):
        """
        this function determines how much capital to spend on the stock and returns the number of shares
        """
        stock_expenditure = self.capital * buy_perc
        n_shares = math.floor(stock_expenditure / buy_price)
        return n_shares

    def trailing_stop_loss(self):
        """
        activates a trailing stop loss
        """
        pass
    
    def print_bag(self):
        """
        print current stocks holding
        """
        print ("{:<10} {:<10} {:<10} {:<10}".format('STOCK', 'BUY PRICE', 'SHARES', 'TOTAL VALUE'))
        for key, value in self.buy_orders.items():
            print("{:<10} {:<10} {:<10} {:<10}".format(key, value[0], value[1], value[2]))
        print('\n')  

    def create_summary(self, print_results=False):
        """
        Create a summary of the trades.
        """
        if print_results:
            print("{:<10} {:<10} {:<10} {:<10} {:<10}".format('STOCK', 'BUY PRICE', 'SHARES', 'SELL PRICE', 'NET GAIN'))

        # Store rows to add to the history DataFrame
        rows_to_add = []

        for values in self.history:
            net_gain = (values[3] - values[1]) * values[2]
            self.total_gain += net_gain
            row = {
                'stock': values[0],
                'buy_price': values[1],
                'n_shares': values[2],
                'sell_price': values[3],
                'net_gain': net_gain,
                'buy_date': values[4],
                'sell_date': values[5]
            }
            rows_to_add.append(row)

            if print_results:
                print("{:<10} {:<10} {:<10} {:<10} {:<10}".format(values[0], values[1], values[2], values[3], np.round(net_gain, 2)))

        # Convert the rows to a DataFrame
        rows_df = pd.DataFrame(rows_to_add)

        # Drop columns that are entirely NA
        rows_df = rows_df.dropna(axis=1, how='all')

        # Drop columns from history_df that are entirely NA
        self.history_df = self.history_df.dropna(axis=1, how='all')

        # Only concatenate if both DataFrames are not empty
        if not rows_df.empty and not self.history_df.empty:
            self.history_df = pd.concat([self.history_df, rows_df], ignore_index=True)
        elif not rows_df.empty:  # If history_df is empty, assign rows_df directly
            self.history_df = rows_df

    def print_summary(self):
        """
        prints the summary of results
        """
        self.create_summary(print_results = True)
        print('\n')
        print(f'Initial Balance: {self.initial_capital:.2f}')
        print(f'Final Balance: {(self.initial_capital + self.total_gain):.2f}')
        print(f'Total gain: {self.total_gain:.2f}')
        print(f'P/L : {(self.total_gain/self.initial_capital)*100:.2f} %')
        print('\n')