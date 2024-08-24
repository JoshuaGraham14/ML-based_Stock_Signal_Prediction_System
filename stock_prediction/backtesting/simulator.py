import numpy as np
import math
import pandas as pd

class Simulator:
    def __init__(self, capital):
        """
        Initialize the simulator with the capital.
        """
        self.capital = capital
        self.initial_capital = capital  # Keep a copy of the initial capital
        self.total_gain = 0
        self.buy_orders = {}
        self.history = []
        # Create a pandas df to save history
        cols = ['stock', 'buy_price', 'n_shares', 'sell_price', 'net_gain', 'buy_date', 'sell_date', 'days_held']
        self.history_df = pd.DataFrame(columns=cols)
           
    def buy(self, stock, buy_price, buy_date):
        """
        Function takes buy price and the number of shares and buys the stock.
        """
        # Calculate the procedure
        n_shares = self.buy_percentage(buy_price)
        self.capital -= buy_price * n_shares  # Update capital after buying
        self.buy_orders[stock] = [buy_price, n_shares, buy_price * n_shares, buy_date]

    def sell(self, stock, sell_price, n_shares_sell, sell_date):
        """
        Function to sell stock given the stock name and number of shares.
        """
        buy_price, n_shares, _, buy_date = self.buy_orders[stock]
        sell_amount = sell_price * n_shares_sell

        self.capital += sell_amount  # Update capital after selling

        if (n_shares - n_shares_sell) == 0:  # If sold all
            days_held = (sell_date - buy_date).days
            net_gain = sell_amount - buy_price * n_shares
            self.total_gain += net_gain  # Update total gain
            self.history.append([stock, buy_price, n_shares, sell_price, net_gain, buy_date, sell_date, days_held])
            del self.buy_orders[stock]
        else:
            n_shares = n_shares - n_shares_sell
            self.buy_orders[stock][1] = n_shares
            self.buy_orders[stock][2] = buy_price * n_shares

    def buy_percentage(self, buy_price, buy_perc=1):
        """
        This function determines how much capital to spend on the stock and returns the number of shares.
        """
        stock_expenditure = self.capital * buy_perc
        n_shares = math.floor(stock_expenditure / buy_price)
        return n_shares

    def print_bag(self):
        """
        Print current stocks holding.
        """
        print("{:<10} {:<10} {:<10} {:<10}".format('STOCK', 'BUY PRICE', 'SHARES', 'TOTAL VALUE'))
        for key, value in self.buy_orders.items():
            print("{:<10} {:<10} {:<10} {:<10}".format(key, value[0], value[1], value[2]))
        print('\n')  

    def create_summary(self, print_results=False):
        """
        Create a summary of the trades.
        """
        if print_results:
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<15} {:<15} {:<10}".format('STOCK', 'BUY PRICE', 'SHARES', 'SELL PRICE', 'NET GAIN', 'BUY DATE', 'SELL DATE', 'DAYS HELD'))

        # Store rows to add to the history DataFrame
        rows_to_add = []

        for values in self.history:
            row = {
                'stock': values[0],
                'buy_price': values[1],
                'n_shares': values[2],
                'sell_price': values[3],
                'net_gain': values[4],
                'buy_date': values[5],
                'sell_date': values[6],
                'days_held': values[7]
            }
            rows_to_add.append(row)

            if print_results:
                print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<15} {:<15} {:<10}".format(
                    values[0], values[1], values[2], values[3], np.round(values[4], 2),
                    values[5].strftime('%Y-%m-%d'), values[6].strftime('%Y-%m-%d'), values[7]))

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
        Prints the summary of results.
        """
        self.create_summary(print_results=True)
        print('\n')
        print(f'Initial Balance: {self.initial_capital:.2f}')
        print(f'Final Balance: {self.capital:.2f}')
        print(f'Total gain: {self.total_gain:.2f}')
        print(f'P/L : {(self.total_gain/self.initial_capital)*100:.2f} %')
        print('\n')
