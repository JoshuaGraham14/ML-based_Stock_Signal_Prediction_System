import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns

from stock_utils import *

def create_train_data(df, technical_indicators):
    _data_ = df[(df['loc_min'] > 0) | (df['loc_max'] > 0)].reset_index(drop = True)

    #create a dummy variable for local_min (0) and max (1)
    _data_['target'] = [1 if x > 0 else 0 for x in _data_.loc_max]

    _data_ = _data_[technical_indicators+['target']]

    return _data_.dropna(axis = 0)


def display_coefficients(model, feature_names, show_graph=False):
    # Ensure feature_names does not include the target
    feature_names = [f for f in feature_names if f != 'target']
    coefficients = model.coef_[0]
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df['Coefficient'] = coef_df['Coefficient'].round(3)
    
    # Print the DataFrame
    print(coef_df[['Feature', 'Coefficient']])

    if show_graph:
        # Normalize coefficients for color mapping
        norm = plt.Normalize(coef_df['Coefficient'].min(), coef_df['Coefficient'].max())
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        coef_df['Color'] = coef_df['Coefficient'].apply(lambda x: sm.to_rgba(x))

        # Plotting the coefficients with colors
        fig, ax = plt.subplots(figsize=(10, 6))
        coef_df.sort_values(by='Coefficient', ascending=False, inplace=True)
        bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=coef_df['Color'])
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Feature Coefficients')

        # Add a color bar
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Coefficient Value')
        
        plt.show()

def train_and_test(training_df, technical_indicators, show_matrix=True, show_report=True, show_intercept=True, show_coefficients=True):
    X = training_df[technical_indicators]
    y = training_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # instantiate the model (using the default parameters)
    model = LogisticRegression(random_state=16)

    # fit the model with data
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if show_matrix:
        matrix = metrics.confusion_matrix(y_test, y_pred)
        print(matrix)

    if show_report:
        target_names = ['min', 'max']
        print(classification_report(y_test, y_pred, target_names=target_names))

    if show_intercept:
        print("Intercept:", round(model.intercept_[0], 3))

    if show_coefficients:
        display_coefficients(model, technical_indicators, show_graph=False)

'''
-----------********** MAIN **********-----------
'''

stock_symbol = 'AAPL'
technical_indicators = ["normalized_value", "2_reg", "3_reg", "5_reg", "10_reg", "20_reg", "50_reg", "adx", "ema", "percent_b", "rsi"]

stock_utils = StockUtils(symbol=stock_symbol, outputsize=2000)
df = stock_utils.get_indicators(technical_indicators)
print(df)

training_df = create_train_data(df, technical_indicators)

print(training_df)

train_and_test(training_df, technical_indicators)

# plot_graph(df, stock_symbol, show_maxs=True, show_mins=True)