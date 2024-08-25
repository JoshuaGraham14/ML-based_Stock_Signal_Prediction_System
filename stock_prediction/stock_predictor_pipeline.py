import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_prediction.models import ModelTrainer, ModelPredictor
from stock_prediction.backtesting import Backtester

class StockPredictorPipeline:
    def __init__(self, training_symbols, testing_symbols, technical_indicators, params):
        """
        Initialize the StockPredictorPipeline class.

        Args:
        - training_symbols: List of stock symbols for training.
        - testing_symbols: List of stock symbols for testing.
        - technical_indicators: List of technical indicators to use.
        - params: Dictionary of parameters including 'outputsize', 'min_max_order', 'min_threshold',
                  'max_threshold', and 'window_size'.
        """
        self.training_symbols = training_symbols
        self.testing_symbols = testing_symbols
        self.technical_indicators = technical_indicators
        self.params = params

    def run(self, plot_graph=True):
        """
        Run the complete pipeline: train the model and test it with optional plotting.
        """
        model_trainer = ModelTrainer(
            self.training_symbols, 
            self.technical_indicators, 
            self.params)
        
        trained_model = model_trainer.train()

        model_predictor = ModelPredictor(
            trained_model,
            model_trainer.scaler,
            self.testing_symbols,
            self.technical_indicators,
            self.params
        )

        df_predictions = model_predictor.test(plot_graph=plot_graph)

        return df_predictions
    
    def run_backtest(self, initial_capital=10000, sell_perc=0.04, hold_till=5, stop_perc=0.005):
        """
        Run the backtesting process with the trained model.
        """
        model_trainer = ModelTrainer(self.training_symbols, self.technical_indicators, self.params)
        trained_model = model_trainer.train()

        backtester = Backtester(
            model=trained_model,
            scaler=model_trainer.scaler,
            testing_symbols=self.testing_symbols,
            technical_indicators=self.technical_indicators,
            params=self.params,
            capital=initial_capital,
            sell_perc=sell_perc,
            hold_till=hold_till,
            stop_perc=stop_perc
        )

        backtester.run()
        