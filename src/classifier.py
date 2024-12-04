import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report

# Crear el pipeline para zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", return_all_scores=True)
file_path = "data/test_data.csv"  
sample_texts = pd.read_csv(file_path)

candidate_labels = ["urgent", "artificial intelligence", "computer", "travel", "animal", "fiction"]
true_labels = sample_texts['label']
texts = list(sample_texts['text'])
results = classifier(texts, candidate_labels=candidate_labels)
predicted_labels = [result["labels"][0] for result in results]

results_as_list = []
for result in results:
    row = {
        "text": result["sequence"],
        "predicted_label": result["labels"][0]
    }
    for label, score in zip(result["labels"], result["scores"]):
        row[f"probabilities_{label}"] = score
    results_as_list.append(row)

results_df = pd.DataFrame(results_as_list)

# Guardar el DataFrame como un archivo CSV
output_file_path = "data/test_predictions_with_probabilities.csv"
results_df.to_csv(output_file_path, index=False)


# # Calcular métricas de clasificación
print("\nReporte de clasificación:")
print(classification_report(true_labels, predicted_labels, zero_division=1, labels=candidate_labels, target_names=candidate_labels))
