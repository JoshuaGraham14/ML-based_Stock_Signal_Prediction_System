from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from stock_utils import StockUtils

class StockPredictionModel:
    def __init__(self, training_symbols, testing_symbols, technical_indicators, outputsize=5000):
        self.training_symbols = training_symbols
        self.testing_symbols = testing_symbols
        self.technical_indicators = technical_indicators
        self.outputsize = outputsize
        self.model = None
        self.scaler = StandardScaler()
    
    def gather_stock_data(self, stock_symbols):
        """
        Combine stock data for all symbols into a single DataFrame.
        """
        combined_df = pd.DataFrame()
        for symbol in stock_symbols:
            stock_utils = StockUtils(symbol=symbol)
            df = stock_utils.get_indicators(self.technical_indicators, outputsize=self.outputsize)
            df['symbol'] = symbol  # Add a column to identify the stock
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
            bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=coef_df['Color'])
            ax.set_xlabel('Coefficient Value')
            ax.set_title('Feature Coefficients')

            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Coefficient Value')
            plt.show()

    def train_model(self, training_df, threshold=0.5, show_intercept=True, show_coefficients=True):
        """
        Train a logistic regression model on the training data.
        """
        X = training_df[self.technical_indicators]
        y = training_df['target']

        # Apply scaling to the features
        X = self.scaler.fit_transform(X)

        # Train on all available data
        model = LogisticRegression(random_state=16)
        # model = RandomForestClassifier(random_state=16, n_estimators=100)
        model.fit(X, y)
        
        if show_intercept:
            print("Intercept:", round(model.intercept_[0], 3))

        if show_coefficients:
            self.display_model_coefficients(model, self.technical_indicators, show_graph=False)

        self.model = model
        return model

    def evaluate_model(self, df, min_threshold=0.00001, max_threshold=0.99999):
        """
        Evaluate the model on the entire DataFrame and add predictions.
        Predicts 0 (minima) if the smoothed predicted probability is less than min_threshold.
        Predicts 1 (maxima) if the smoothed predicted probability is greater than max_threshold.
        Anything in between is considered neutral (i.e., no prediction).
        """
        df_clean = df.dropna(subset=self.technical_indicators)
        X = df_clean[self.technical_indicators]

        # Apply the same scaling to the test data
        X = self.scaler.fit_transform(X)

        y_pred_proba = self.model.predict_proba(X)[:, 1]

        # Initialize predictions with a neutral value, such as None or -1
        y_pred = -1 * np.ones_like(y_pred_proba)

        # Predict minima
        y_pred[y_pred_proba < min_threshold] = 0

        # Predict maxima
        y_pred[y_pred_proba > max_threshold] = 1

        df_clean = df_clean.copy()  # Ensure we are working on a copy to avoid warnings
        df_clean['predicted_target'] = y_pred
        df_clean['predicted_probability'] = y_pred_proba

        df = df.merge(df_clean[['predicted_target', 'predicted_probability']], left_index=True, right_index=True, how='left')
        df = df.dropna(subset=['predicted_target', 'predicted_probability'])

        return df

    def plot_stock_predictions(self, axs, df, stock_symbol, index):
        """
        Plot the stock price and predictions for a given stock symbol.
        """
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        x = df['date']
        y = df["close"]
        axs[index].plot(x, y, ls='-', color="black", label=f"{stock_symbol} Daily")

        # Plot predicted minima
        minima_points = df[df['predicted_target'] == 0]
        axs[index].scatter(minima_points['date'], minima_points['close'], marker='o', color="green", label="Predicted Minima", s=50)

        # Plot predicted maxima
        maxima_points = df[df['predicted_target'] == 1]
        axs[index].scatter(maxima_points['date'], maxima_points['close'], marker='o', color="blue", label="Predicted Maxima", s=50)

        # # Plot neutral points (neither minima nor maxima)
        # neutral_points = df[df['predicted_target'] == -1]
        # axs[index].scatter(neutral_points['date'], neutral_points['close'], marker='o', color="gray", label="Neutral Points", s=50, alpha=0.5)

        # Plot true minima (if available in the df)
        axs[index].scatter(df['date'], df.get('loc_min', pd.Series()), marker='x', color="red", label="True Minima", s=20)

        # Plot true maxima (if available in the df)
        axs[index].scatter(df['date'], df.get('loc_max', pd.Series()), marker='x', color="orange", label="True Maxima", s=20)

        axs[index].set_xlabel("Date")
        axs[index].set_ylabel("Close (USD)")
        axs[index].legend()
        axs[index].set_title(f"Predictions for {stock_symbol}")
        axs[index].tick_params(axis='x', rotation=45)
        axs[index].grid(True)

    def filter_predictions(self, df, window=5):
        """
        Filter out predicted minima and maxima that are too close to each other.
        
        Parameters:
        - df: DataFrame containing at least ['date', 'predicted_target'].
        - window: The number of days within which only one prediction (minima or maxima) should be kept.

        Returns:
        - Filtered DataFrame with closely spaced predictions removed.
        """
        # Ensure that the DataFrame is sorted by date
        df = df.sort_values(by='date').reset_index(drop=True)

        # Initialize a list to store indices of rows to keep
        keep_indices = []

        # Last recorded minima/maxima positions
        last_minima_idx = -window - 1
        last_maxima_idx = -window - 1

        for idx, row in df.iterrows():
            if row['predicted_target'] == 0:  # Minima
                if idx - last_minima_idx > window:
                    keep_indices.append(idx)
                    last_minima_idx = idx

            elif row['predicted_target'] == 1:  # Maxima
                if idx - last_maxima_idx > window:
                    keep_indices.append(idx)
                    last_maxima_idx = idx

        # Filter the DataFrame
        filtered_df = df.loc[keep_indices].reset_index(drop=True)

        return filtered_df

    def run(self):
        """
        Run the complete pipeline: gather data, train the model, and visualize predictions.
        """
        combined_df = self.gather_stock_data(self.training_symbols)
        training_df = self.prepare_training_data(combined_df)
        self.train_model(training_df)

        fig, axs = plt.subplots(len(self.testing_symbols), 1, figsize=(12, 6 * len(self.testing_symbols)))

        if len(self.testing_symbols) == 1:
            axs = [axs]  # Ensure axs is iterable when there's only one subplot

        for index, symbol in enumerate(self.testing_symbols):
            print(f"Processing stock: {symbol}")
            stock_utils_new = StockUtils(symbol=symbol)
            df_new_stock = stock_utils_new.get_indicators(self.technical_indicators, outputsize=self.outputsize)
            df_predictions = self.evaluate_model(df_new_stock)  # Evaluate model with the new prediction logic
            # df_predictions = self.filter_predictions(df_predictions, window=0)
            self.plot_stock_predictions(axs, df_predictions, symbol, index)

        plt.tight_layout()
        plt.show()

        return df_predictions