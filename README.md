# Two Projects

Este repositorio contiene dos proyectos principales: uno para la predicción de ventas y otro para la clasificación de textos.

## Proyectos

### 1. Predicción de Ventas

Este proyecto utiliza modelos SARIMAX para predecir las ventas futuras basándose en datos históricos y variables exógenas como promociones y días festivos.

#### Archivos Principales

- `src/forecasting.py`: Script principal para la predicción de ventas.
- `src/forecast_utils.py`: Funciones auxiliares utilizadas en el script de predicción.
- `data/sales_data.csv`: Archivo CSV con los datos históricos de ventas.
- `data/predictions_week_29_30.csv`: Archivo CSV con las predicciones de ventas para las semanas 29 y 30.

#### Uso

1. Asegúrate de tener `poetry` instalado. Si no lo tienes, puedes instalarlo siguiendo las instrucciones en [Poetry](https://python-poetry.org/docs/#installation).

2. Instala las dependencias del proyecto:
    ```bash
    poetry install
    ```

3. Ejecuta el script de predicción:
    ```bash
    poetry run python src/forecasting.py
    ```

### 2. Clasificación de Textos

Este proyecto utiliza un clasificador de textos para asignar etiquetas a textos basándose en un conjunto de etiquetas candidatas.

#### Archivos Principales

- `src/classifying.py`: Script principal para la clasificación de textos.
- `data/text_sample_data.csv`: Archivo CSV con los textos a clasificar.
- `data/text_samples_classified.csv`: Archivo CSV con los resultados de la clasificación.

#### Uso

1. Asegúrate de tener `poetry` instalado. Si no lo tienes, puedes instalarlo siguiendo las instrucciones en [Poetry](https://python-poetry.org/docs/#installation).

2. Instala las dependencias del proyecto:
    ```bash
    poetry install
    ```

3. Ejecuta el script de clasificación:
    ```bash
    poetry run python src/classifying.py
    ```

## Estructura del Repositorio
mercado_libre_project/ │ ├── data/ │ ├── sales_data.csv │ ├── predictions_week_29_30.csv │ └── text_sample_data.csv │ ├── src/ │ ├── forecasting.py │ ├── forecast_utils.py │ └── classifying.py │ ├── images/ │ └── best_model_plot.png │ ├── README.md └── pyproject.toml

## Requisitos

- Python 3.11
- Poetry

## Contacto

Para cualquier consulta, por favor contacta a [levent.arturo@im.unam.mx](mailto:levent.arturo@im.unam.mx).