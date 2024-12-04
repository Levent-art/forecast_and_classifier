import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.base import BaseEstimator, RegressorMixin
import random


# Cargar los datos desde el archivo CSV
file_path = "data/sales_data.csv"  # Cambia esto si tu archivo está en otra ubicación
df = pd.read_csv(file_path)

# Configurar las fechas como índice y establecer la frecuencia explícitamente
df["Week"] = pd.date_range(start="2023-01-01", periods=len(df), freq="W")
df.set_index("Week", inplace=True)
df.index.freq = "W"  # Establecer la frecuencia como semanal

# Dividir los datos en entrenamiento y prueba
#test_proportion = 0.2
#test_size = int(test_proportion*len(df))
test_size = 4
train = df.iloc[:-test_size]  # Todas las semanas excepto las últimas 12
test = df.iloc[-test_size:]  # Últimas 12 semanas para prueba

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Función para evaluar una combinación de hiperparámetros usando TimeSeriesSplit
def evaluate_sarimax_model(data, arima_order, seasonal_order, exog, test_size=4, n_splits=5):
    tscv = TimeSeriesSplit(test_size=test_size, n_splits=n_splits)
    errors = []

    for train_index, test_index in tscv.split(data):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        
        # # Check if the training set has at least 4 points
        # if len(train) < 4:
        #     continue
        
        try:
            # Entrenar el modelo
            model = SARIMAX(train["Sales"], order=arima_order, seasonal_order=seasonal_order, exog=train[exog])
            model_fitted = model.fit(disp=False)
            # Hacer predicciones
            test_predictions = model_fitted.forecast(steps=len(test), exog=test[exog])
            # Calcular el error para el split actual
            error = mean_absolute_error(test["Sales"], test_predictions)
            errors.append(error)
        except Exception as e:
            print(f"Failed to fit SARIMAX{arima_order}x{seasonal_order} on split: {e}")
            errors.append(np.inf)

    # Calcular el error promedio a través de todos los splits
    if errors:
        average_error = np.mean(errors)
    else:
        average_error = np.inf  # If no valid splits, return a high error
    return average_error

# Randomized Grid Search para encontrar los mejores hiperparámetros
best_score, best_order, best_seasonal_order = float("inf"), None, None
exog_vars = ["Promotion", "Holiday"]
n_iter = 50  # Numero de muestras aleatorias

# Rango de hiperparámetros a probar
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
    score = evaluate_sarimax_model(df, order, seasonal_order, exog_vars)
    if score < best_score:
        best_score, best_order, best_seasonal_order = score, order, seasonal_order
        print(f"SARIMAX{order}x{seasonal_order} MAE={score:.3f}")

print(f"\nMejores Hiperparámetros: SARIMAX{best_order}x{best_seasonal_order} con MAE={best_score:.3f}")

# Entrenar el modelo óptimo
best_model = SARIMAX(train["Sales"], order=best_order, seasonal_order=best_seasonal_order, exog=train[exog_vars])
best_model_fitted = best_model.fit(disp=False)

# Hacer predicciones con el mejor modelo
train_predictions = best_model_fitted.predict(start=0, end=len(train) - 1, exog=train[exog_vars])
test_predictions = best_model_fitted.forecast(steps=len(test), exog=test[exog_vars])

# Calcular métricas adicionales para el conjunto de entrenamiento
train_mae = mean_absolute_error(train["Sales"], train_predictions)
train_mse = mean_squared_error(train["Sales"], train_predictions)
train_mape = np.mean(np.abs((train["Sales"] - train_predictions) / train["Sales"])) * 100  # MAPE en porcentaje

# Calcular métricas adicionales para el conjunto de prueba
test_mae = mean_absolute_error(test["Sales"], test_predictions)
test_mse = mean_squared_error(test["Sales"], test_predictions)
test_mape = np.mean(np.abs((test["Sales"] - test_predictions) / test["Sales"])) * 100  # MAPE en porcentaje

# Visualizar las predicciones vs los datos reales con métricas
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Sales"], label="Ventas reales", color="blue", linestyle='-')
plt.plot(train.index, train_predictions, label="Predicciones de entrenamiento", color="green", linestyle='--')
plt.plot(test.index, test_predictions, label="Predicciones de prueba", color="red", linestyle='-.')

# Añadir una línea vertical para marcar el inicio del conjunto de prueba
plt.axvline(x=test.index[0], color='black', linestyle='-', linewidth=2, label='Inicio de prueba')

# Añadir las métricas en el gráfico
train_metric_text = f"Entrenamiento\nMAE: {train_mae:.2f}\nMSE: {train_mse:.2f}\nRMSE: {np.sqrt(train_mse):.2f}\nMAPE: {train_mape:.2f}%"
test_metric_text = f"Prueba\nMAE: {test_mae:.2f}\nMSE: {test_mse:.2f}\nRMSE: {np.sqrt(test_mse):.2f}\nMAPE: {test_mape:.2f}%"

plt.text(
    test.index[-4],  # Coordenada x: inicio del conjunto de prueba
    max(df["Sales"]) * 0.0,  # Coordenada y: un poco debajo del valor máximo de las ventas
    train_metric_text,
    fontsize=8,
    color="black",
    bbox=dict(facecolor="white", alpha=0.8)
)

plt.text(
    test.index[-4],  # Coordenada x: inicio del conjunto de prueba
    max(df["Sales"]) * 0.4,  # Coordenada y: un poco más abajo para no superponer
    test_metric_text,
    fontsize=8,
    color="black",
    bbox=dict(facecolor="white", alpha=0.8)
)

plt.title(f"Mejor Modelo SARIMAX{best_order}x{best_seasonal_order} vs Datos Reales")
plt.xlabel("Semana")
plt.ylabel("Ventas")
plt.legend()
plt.grid()
plt.show()