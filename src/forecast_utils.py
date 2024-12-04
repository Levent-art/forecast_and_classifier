import os
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Evaluate the model using time series cross-validation since the data is small
def evaluate_sarimax_model(data, arima_order, seasonal_order, exog, test_size=4, n_splits=5):
    tscv = TimeSeriesSplit(test_size=test_size, n_splits=n_splits)
    metrics_list = []

    for train_index, test_index in tscv.split(data):
        train = data.iloc[train_index]
        test = data.iloc[test_index]

        try:
            # Train the SARIMAX model
            model = SARIMAX(train["Sales"], order=arima_order, seasonal_order=seasonal_order, exog=train[exog])
            model_fitted = model.fit(disp=False)
            # Make predictions
            test_predictions = model_fitted.forecast(steps=len(test), exog=test[exog])
            # Calculate metrics for the current split
            metrics = calculate_metrics(test["Sales"], test_predictions)
            metrics_list.append(metrics)
        except Exception as e:
            print(f"Failed to fit SARIMAX{arima_order}x{seasonal_order} on split: {e}")
            metrics_list.append({"MAE": np.inf, "MSE": np.inf, "RMSE": np.inf, "MAPE": np.inf})

    # Calculate average metrics across all splits
    avg_metrics = {
        "MAE": np.mean([m["MAE"] for m in metrics_list]),
        "MSE": np.mean([m["MSE"] for m in metrics_list]),
        "RMSE": np.mean([m["RMSE"] for m in metrics_list]),
        "MAPE": np.mean([m["MAPE"] for m in metrics_list])
    }

    return avg_metrics

def randomized_grid_search(df, exog_vars, metric='MAE', n_iter=50):
    """
    Perform a randomized grid search to find the best hyperparameters for a SARIMAX model.

    Parameters:
    df (DataFrame): The DataFrame with the data.
    exog_vars (list): List of exogenous variables.
    metric (str): The metric to use for evaluation. One of 'MAE', 'MSE', 'RMSE', 'MAPE'.
    n_iter (int): Number of iterations for the random search.

    Returns:
    tuple: Best score, best order, and best seasonal order.
    """
    best_metric, best_order, best_seasonal_order = float("inf"), None, None

    # Range of hyperparameters to try
    p_values = range(0, 2)
    d_values = range(0, 2)
    q_values = range(0, 2)
    P_values = range(0, 2)  # Seasonal AR terms
    D_values = range(0, 2)  # Seasonal differencing terms
    Q_values = range(0, 2)  # Seasonal MA terms
    m_values = [2, 4, 8]  # Seasonal period 

    for _ in range(n_iter):
        p = random.choice(p_values)
        d = random.choice(d_values)
        q = random.choice(q_values)
        P = random.choice(P_values)
        D = random.choice(D_values)
        Q = random.choice(Q_values)
        m = random.choice(m_values)

        order = (p, d, q)
        seasonal_order = (P, D, Q, m)
        metric_value = evaluate_sarimax_model(df, order, seasonal_order, exog_vars)[metric]

        if metric_value < best_metric:
            best_metric, best_order, best_seasonal_order = metric_value, order, seasonal_order
            print(f"SARIMAX{order}x{seasonal_order} {metric}={metric_value:.3f}")

    print(f"\nBest Hyperparameters: SARIMAX{best_order}x{best_seasonal_order} with {metric}={best_metric:.3f}")
    crossval_metrics = evaluate_sarimax_model(df, best_order, best_seasonal_order, exog_vars)
    return best_metric, best_order, best_seasonal_order, crossval_metrics

def calculate_metrics(y_true, y_pred):
    """
    Calculate additional metrics for the training set.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    dict: Dictionary containing MAE, MSE, and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # MAPE in percentage
    rmse = np.sqrt(mse)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape
    }


def plot_predictions_with_test_marker(ax, df, train, train_predictions, test, test_predictions):
    """
    Plot the real sales vs the predictions with a vertical line marking the start of the test set.

    Parameters:
    ax (Axes): The matplotlib Axes object to plot on.
    df (DataFrame): The DataFrame with the actual sales data.
    train (DataFrame): The DataFrame with the training data.
    train_predictions (array-like): The predictions for the training set.
    test (DataFrame): The DataFrame with the test data.
    test_predictions (array-like): The predictions for the test set.
    """
    ax.plot(df.index, df["Sales"], label="Real Sales", color="blue", linestyle='-')
    ax.plot(train.index, train_predictions, label="Training Predictions", color="green", linestyle='--')
    ax.plot(test.index, test_predictions, label="Test Predictions", color="red", linestyle='-.')

    # Add a vertical line to mark the start of the test set
    ax.axvline(x=test.index[0], color='black', linestyle='-', linewidth=2, label='Test Start')


def plot_metrics_text(ax, train_metrics, test_metrics, crossval_metrics, df, test_index):
    """
    Plot the metrics text on the graph.

    Parameters:
    ax (Axes): The matplotlib Axes object to plot on.
    train_metrics (dict): Dictionary containing training metrics.
    test_metrics (dict): Dictionary containing test metrics.
    df (DataFrame): The DataFrame with the data.
    test_index (Index): The index of the test set.
    """
    train_metric_text = (
        f"Training\nMAE: {train_metrics['MAE']:.2f}\n"
        f"MSE: {train_metrics['MSE']:.2f}\n"
        f"RMSE: {train_metrics['RMSE']:.2f}\n"
        f"MAPE: {train_metrics['MAPE']:.2f}%"
    )

    test_metric_text = (
        f"Test\nMAE: {test_metrics['MAE']:.2f}\n"
        f"MSE: {test_metrics['MSE']:.2f}\n"
        f"RMSE: {test_metrics['RMSE']:.2f}\n"
        f"MAPE: {test_metrics['MAPE']:.2f}%"
    )

    crossval_metric_text = (
        f"Cross-Validation\nMAE: {crossval_metrics['MAE']:.2f}\n"
        f"MSE: {crossval_metrics['MSE']:.2f}\n"
        f"RMSE: {crossval_metrics['RMSE']:.2f}\n"
        f"MAPE: {crossval_metrics['MAPE']:.2f}%"
    )

    ax.text(
        test_index[-3],  # x-coordinate: start of the test set
        100,
        train_metric_text,
        fontsize=6,
        color="black",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    ax.text(
        test_index[-3],  # x-coordinate: start of the test set
        -100,
        test_metric_text,
        fontsize=6,
        color="black",
        bbox=dict(facecolor="white", alpha=0.8)
    )
    ax.text(
        test_index[-3],  # x-coordinate: start of the test set
        0,
        crossval_metric_text,
        fontsize=6,
        color="black",
        bbox=dict(facecolor="white", alpha=0.8)
    )

def finalize_plot(ax, df,  best_order, best_seasonal_order, filename="best_forecast_plot.png"):
    """
    Finalize the plot with title, labels, legend, grid, and save the plot to the images folder.

    Parameters:
    ax (Axes): The matplotlib Axes object to plot on.
    best_order (tuple): The best order for the SARIMAX model.
    best_seasonal_order (tuple): The best seasonal order for the SARIMAX model.
    filename (str): The filename to save the plot as.
    """
    ax.set_title(f"Best SARIMAX Model {best_order}x{best_seasonal_order} vs Real Data")
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid()

    # Set x-axis ticks to show weeks with labels "W1", "W2", etc.
    xticks = df.index
    xticklabels = [f"W{i}" for i in range(0, len(df.index))]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45)

    # Ensure the images directory exists
    if not os.path.exists("images"):
        os.makedirs("images")

    # Save the plot to the images folder
    plt.savefig(os.path.join("images", filename))
    plt.show()