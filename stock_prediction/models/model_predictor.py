import numpy as np
import pandas as pd

class ModelPredictor:
    def __init__(self, model, scaler, testing_symbols, technical_indicators, params):
        """
        Initialize the ModelPredictor class.
        
        Args:
        - model: Trained ML model.
        - testing_symbols: List of stock symbols for testing.
        - technical_indicators: List of technical indicators to use.
        - params: Dictionary of parameters including 'outputsize', 'min_max_order', 'min_threshold',
                  'max_threshold', and 'window_size'.
        """
        self.model = model
        self.scaler = scaler
        self.testing_symbols = testing_symbols
        self.technical_indicators = technical_indicators
        
        self.min_threshold = params.get('min_threshold', 0.00001)
        self.max_threshold = params.get('max_threshold', 0.99999)
        self.window_size = params.get('window_size', 5)

    def evaluate_model(self, df):
        """
        Evaluate the model on the entire DataFrame and add predictions.
        """
        df_clean = df.dropna(subset=self.technical_indicators)
        X = df_clean[self.technical_indicators]

        # Apply the same scaling to the test data
        X = self.scaler.fit_transform(X)

        y_pred_proba = self.model.predict_proba(X)[:, 1]

        y_pred = -1 * np.ones_like(y_pred_proba)

        y_pred[y_pred_proba < self.min_threshold] = 0
        y_pred[y_pred_proba > self.max_threshold] = 1

        df_clean = df_clean.copy()
        df_clean['predicted_target'] = y_pred
        df_clean['predicted_probability'] = y_pred_proba

        df = df.merge(df_clean[['predicted_target', 'predicted_probability']], left_index=True, right_index=True, how='left')
        df = df.dropna(subset=['predicted_target', 'predicted_probability'])

        return df

    def filter_predictions(self, df):
        """
        Filter out predicted minima and maxima that are too close to each other.
        """
        df = df.sort_values(by='date').reset_index(drop=True)
        keep_indices = []

        last_minima_idx = -self.window_size - 1
        last_maxima_idx = -self.window_size - 1

        for idx, row in df.iterrows():
            if row['predicted_target'] == 0:
                if idx - last_minima_idx > self.window_size:
                    keep_indices.append(idx)
                    last_minima_idx = idx
            elif row['predicted_target'] == 1:
                if idx - last_maxima_idx > self.window_size:
                    keep_indices.append(idx)
                    last_maxima_idx = idx

        filtered_df = df.loc[keep_indices].reset_index(drop=True)

        return filtered_df

    def plot_stock_predictions(self, axs, df_new_stock, df, stock_symbol, index):
        """
        Plot the stock price and predictions for a given stock symbol.
        """
        df_new_stock['date'] = pd.to_datetime(df_new_stock['date'], format='%d/%m/%Y')
        x = df_new_stock['date']
        y = df_new_stock["close"]
        axs[index].plot(x, y, ls='-', color="black", label=f"{stock_symbol} Daily")

        minima_points = df[df['predicted_target'] == 0]
        axs[index].scatter(minima_points['date'], minima_points['close'], marker='o', color="green", label="Predicted Minima", s=50)

        maxima_points = df[df['predicted_target'] == 1]
        axs[index].scatter(maxima_points['date'], maxima_points['close'], marker='o', color="blue", label="Predicted Maxima", s=50)

        axs[index].scatter(df['date'], df.get('loc_min', pd.Series()), marker='x', color="red", label="True Minima", s=20)
        axs[index].scatter(df['date'], df.get('loc_max', pd.Series()), marker='x', color="orange", label="True Maxima", s=20)

        axs[index].set_xlabel("Date")
        axs[index].set_ylabel("Close (USD)")
        axs[index].legend()
        axs[index].set_title(f"Predictions for {stock_symbol}")
        axs[index].tick_params(axis='x', rotation=45)
        axs[index].grid(True)