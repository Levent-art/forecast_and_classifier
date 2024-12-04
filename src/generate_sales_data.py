import pandas as pd

# Crear los datos como un diccionario
data = {
    "Week": list(range(1, 29)),
    "Product": ["Product_1"] * 28,
    "Sales": [
        152, 485, 398, 320, 156, 121, 238, 70, 152, 171, 264, 380, 137, 422, 149, 409, 
        201, 180, 199, 358, 307, 393, 463, 343, 435, 241, 493, 326
    ],
    "Promotion": [
        1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0
    ],
    "Holiday": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
    ]
}

# Crear un DataFrame con los datos
df = pd.DataFrame(data)

# Guardar el DataFrame como CSV
file_path = "data/sales_data.csv"
df.to_csv(file_path, index=False)