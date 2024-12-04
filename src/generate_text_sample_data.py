import pandas as pd

# Datos proporcionados
texts = [
    (
        "The axolotl is more than a peculiar amphibian; in its natural environment, it plays an "
        "essential role in the ecological stability of the Xochimilco canals."
    ),
    (
        "Geoffrey Hinton, Yann LeCun and Yoshua Bengio are considered the ‘godfathers’ of "
        "an essential technique in artificial intelligence, called ‘deep learning’."
    ),
    (
        "Greenland is about to open up to adventure-seeking visitors. How many tourists will "
        "come is yet to be seen, but the three new airports will bring profound change."
    ),
    (
        "GitHub Copilot is an AI coding assistant that helps you write code faster and with less "
        "effort, allowing you to focus more energy on problem solving and collaboration."
    ),
    (
        "I have a problem with my laptop that needs to be resolved asap!!"
    )
]

# Etiquetas reales
true_labels = ["animal", "artificial intelligence", "travel", "artificial intelligence", "urgent"]

# Crear un DataFrame con los textos y las etiquetas reales
data = pd.DataFrame({
    "text": texts,
    "label": true_labels
})

# Guardar en un archivo CSV
file_path = "data/text_sample_data.csv"
data.to_csv(file_path, index=False)