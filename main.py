from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from stock_utils import *

def create_train_data_2_way(df, technical_indicators):
    _data_ = df[(df['loc_min'] > 0) | (df['loc_max'] > 0)].reset_index(drop=True)

    # Create a dummy variable for local_min (0) and max (1)
    _data_['target'] = [1 if x > 0 else 0 for x in _data_['loc_max']]

    _data_ = _data_[technical_indicators + ['target']]

    return _data_.dropna(axis=0)

def create_train_data_3_way(df, technical_indicators):
    _data_ = df

    _data_['target'] = [
        0 if loc_min > 0 else 2 if loc_max > 0 else 1
        for loc_min, loc_max in zip(_data_['loc_min'], _data_['loc_max'])
    ]

    _data_ = _data_[technical_indicators + ['target']]

    return _data_.dropna(axis=0)

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

def train_and_test_2_way(training_df, technical_indicators, threshold=0.5, show_matrix=True, show_report=True, show_intercept=True, show_coefficients=True):
    X = training_df[technical_indicators]
    y = training_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the model (using the default parameters)
    model = LogisticRegression(random_state=16)

    # Fit the model with data
    model.fit(X_train, y_train)

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Apply the custom threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    if show_matrix:
        matrix = metrics.confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", matrix)

    if show_report:
        target_names = ['min', 'max']
        print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

    if show_intercept:
        print("Intercept:", round(model.intercept_[0], 3))

    if show_coefficients:
        display_coefficients(model, technical_indicators, show_graph=False)

    return model

def train_and_test_3_way(training_df, technical_indicators, show_matrix=True, show_report=True, show_intercept=True, show_coefficients=True):
    X = training_df[technical_indicators]
    y = training_df['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the logistic regression model for multi-class classification
    model = LogisticRegression(random_state=16, multi_class='multinomial', solver='lbfgs')

    # Fit the model with data
    model.fit(X_train, y_train)

    # Predict class probabilities
    y_pred_proba = model.predict_proba(X_test)

    # Predict classes based on the highest probability
    y_pred = np.argmax(y_pred_proba, axis=1)

    if show_matrix:
        matrix = metrics.confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", matrix)

    if show_report:
        target_names = ['min', 'none', 'max']
        print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

    if show_intercept:
        print("Intercept:", model.intercept_)

    if show_coefficients:
        display_coefficients(model, technical_indicators, show_graph=False)

    return model

def test_on_entire_df_2_way(model, df, technical_indicators, threshold=0.001):
    """
    Test the trained model on the entire DataFrame, including neutral points, after dropping NaN values.
    """
    # Drop rows with NaN values in the technical indicators
    df_clean = df.dropna(subset=technical_indicators)

    # Extract features for prediction
    X = df_clean[technical_indicators]

    # Predict probabilities for the 'max' class
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Apply the threshold to determine the class
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Use .loc to assign values explicitly, avoiding SettingWithCopyWarning
    df_clean.loc[:, 'predicted_target'] = y_pred
    df_clean.loc[:, 'predicted_probability'] = y_pred_proba

    # Merge the predictions back into the original DataFrame
    df = df.merge(df_clean[['predicted_target', 'predicted_probability']], left_index=True, right_index=True, how='left')

    df = df.dropna(subset=['predicted_target', 'predicted_probability'])

    return df

# Example usage
stock_symbol = 'WMT'
technical_indicators = ["normalized_value", "2_reg", "3_reg", "5_reg", "10_reg", "20_reg", "50_reg"]

# Assuming stock_utils and df have been correctly defined elsewhere in your code
stock_utils = StockUtils(symbol=stock_symbol, outputsize=5000)
df = stock_utils.get_indicators(technical_indicators)

training_df_2_way = create_train_data_2_way(df, technical_indicators)

print(training_df_2_way)

# Train and test the model
model = train_and_test_2_way(training_df_2_way, technical_indicators)

# Test the model on the entire DataFrame (including neutral points)
df_with_predictions = test_on_entire_df_2_way(model, df, technical_indicators)

# Display the results
print(df_with_predictions)

# Ensure 'date' is in datetime format in the original DataFrame
df_with_predictions['date'] = pd.to_datetime(df_with_predictions['date'], format='%d/%m/%Y')

# Plot the close prices over time
plt.figure(figsize=(10, 5))
x = df_with_predictions['date']
y = df_with_predictions["close"]
plt.plot(x, y, ls='-', color="black", label="Daily")

# Identify points where predicted_target is 0 (local minima)
minima_points = df_with_predictions[df_with_predictions['predicted_target'] == 0]

# Ensure 'date' is in datetime format in the minima_points DataFrame
minima_points['date'] = pd.to_datetime(minima_points['date'], format='%d/%m/%Y')

# Plot the minima points with green dots
plt.scatter(minima_points['date'], minima_points['close'], marker='o', color="green", label="Predicted Minima (0)", s=50)
plt.scatter(x, df_with_predictions['loc_min'], marker='o', color="red", label="Predicted Minima (0)", s=50)

# Customize the plot
plt.xlabel("Date")
plt.ylabel("Close (USD)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

# Show the plot
plt.show()