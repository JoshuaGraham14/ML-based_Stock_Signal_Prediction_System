from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

from stock_prediction.data_handling import StockUtils

class ModelTrainer:
    def __init__(self, training_symbols, technical_indicators, params):
        """
        Initialize the ModelTrainer class.
        
        Args:
        - training_symbols: List of stock symbols for training.
        - technical_indicators: List of technical indicators to use.
        - params: Dictionary of parameters including 'outputsize', 'min_max_order', 'min_threshold',
                  'max_threshold', and 'window_size'.
        """
        self.training_symbols = training_symbols
        self.technical_indicators = technical_indicators
        
        # Parameters
        self.outputsize = params.get('outputsize', 5000)
        self.min_max_order = params.get('min_max_order', 5)

        self.model = None
        self.scaler = StandardScaler()

    def gather_stock_data(self, stock_symbols):
        """
        Combine stock data for all symbols into a single DataFrame.
        """
        combined_df = pd.DataFrame()
        for symbol in stock_symbols:
            stock_utils = StockUtils(symbol=symbol)
            df = stock_utils.get_indicators(self.technical_indicators, outputsize=self.outputsize, min_max_order=self.min_max_order)

            df = df.copy()
            df.loc[:, 'symbol'] = symbol
            
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        return combined_df

    def prepare_training_data(self, df):
        """
        Prepare the training data by selecting relevant rows and setting the target variable.
        """
        data = df[(df['loc_min'] > 0) | (df['loc_max'] > 0)].reset_index(drop=True)
        data['target'] = [1 if x > 0 else 0 for x in data['loc_max']]
        data = data[self.technical_indicators + ['target']]
        return data.dropna(axis=0)

    def train_model(self, training_df, show_intercept=True, show_coefficients=True):
        """
        Train a logistic regression model on the training data.
        """
        X = training_df[self.technical_indicators]
        y = training_df['target']

        # Apply scaling to the features
        X = self.scaler.fit_transform(X)

        # Train on all available data
        model = LogisticRegression(random_state=16)
        model.fit(X, y)
        
        if show_intercept:
            print("Intercept:", round(model.intercept_[0], 3))

        if show_coefficients:
            self.display_model_coefficients(model, self.technical_indicators, show_graph=False)

        self.model = model
        return model
    
    def display_model_coefficients(self, model, feature_names, show_graph=False):
        """
        Display the coefficients of the trained model, optionally with a graphical representation.
        """
        feature_names = [f for f in feature_names if f != 'target']
        coefficients = model.coef_[0]
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
        coef_df['Coefficient'] = coef_df['Coefficient'].round(3)
        print(coef_df[['Feature', 'Coefficient']])
        
        if show_graph:
            norm = plt.Normalize(coef_df['Coefficient'].min(), coef_df['Coefficient'].max())
            sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
            coef_df['Color'] = coef_df['Coefficient'].apply(lambda x: sm.to_rgba(x))

            fig, ax = plt.subplots(figsize=(10, 6))
            coef_df.sort_values(by='Coefficient', ascending=False, inplace=True)
            ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=coef_df['Color'])
            ax.set_xlabel('Coefficient Value')
            ax.setTitle('Feature Coefficients')

            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Coefficient Value')
            plt.show()

    def train(self):
        """
        High-level method to gather stock data, prepare training data, and train the model.
        """
        combined_df = self.gather_stock_data(self.training_symbols)
        training_df = self.prepare_training_data(combined_df)
        trained_model = self.train_model(training_df)
        return trained_model