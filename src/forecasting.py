import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecast_utils import *

# Load the data
file_path = "data/sales_data.csv"  # Cambia esto si tu archivo está en otra ubicación
df = pd.read_csv(file_path)

# Configurate the index as a datetime object
df["Week"] = pd.date_range(start="2023-01-01", periods=len(df), freq="W")
df.set_index("Week", inplace=True)
df.index.freq = "W"  # Establecer la frecuencia como semanal

# Split the data into training and test sets
test_size = 4
train = df.iloc[:-test_size]  # All except the last 4 weeks for training
test = df.iloc[-test_size:]  # Last 4 weeks for testing

exog_vars = ["Promotion", "Holiday"]

# Randomized Grid Search to find the best hyperparameters
best_score, best_order, best_seasonal_order, crossval_metrics = randomized_grid_search(df, exog_vars, n_iter=50)

# Train best model on the entire training set
best_model = SARIMAX(train["Sales"], order=best_order, seasonal_order=best_seasonal_order, exog=train[exog_vars])
best_model_fitted = best_model.fit(disp=False)

# Predictions for training and test set
train_predictions = best_model_fitted.predict(start=0, end=len(train) - 1, exog=train[exog_vars])
test_predictions = best_model_fitted.forecast(steps=len(test), exog=test[exog_vars])

# Calculate additional metrics for the training set
train_metrics = calculate_metrics(train["Sales"], train_predictions)

# Calculate additional metrics for the test set
test_metrics = calculate_metrics(test["Sales"], test_predictions)

# Plot the predictions and metrics
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the real sales vs the predictions with a vertical line marking the start of the test set
plot_predictions_with_test_marker(ax, df, train, train_predictions, test, test_predictions)

# Plot train, test, and cross-validation metrics as text
plot_metrics_text(ax, train_metrics, test_metrics, crossval_metrics, df, test.index)

# Finalize the plot with title, labels, legend, grid, and save the plot
finalize_plot(ax, df,  best_order, best_seasonal_order, filename="best_forecast_plot.png")


# To finalize the challenge, define future exogenous variables for week 29 and 30
future = pd.DataFrame({
    'Product': 'Product_1',
    'Sales': [0, 0],
    'Promotion': [1, 0],
    'Holiday': [0, 1]
}, index=pd.date_range(start=df.index[-1] + pd.Timedelta(weeks=1), periods=2, freq='W'))

# Format the index to show week numbers
future.index = future.index.strftime('%U')

# Make predictions for week 29 and 30
future_predictions = best_model_fitted.forecast(steps=2, exog=future[exog_vars])

# Ensure the indices match
future_predictions.index = future.index

# Add the predictions to the exogenous variables DataFrame
future['Sales'] = future_predictions

# Reset the index and rename the column to 'Week'
future = future.reset_index().rename(columns={'index': 'Week'})

# Save the predictions to a CSV file
future.to_csv('data/predictions_week_29_30.csv', index=False)

print("Predictions for week 29 and 30 saved to 'data/predictions_week_29_30.csv'")