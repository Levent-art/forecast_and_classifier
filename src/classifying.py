import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report
from classify_utils import classify_texts

# Create pipeline for zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", return_all_scores=True)

candidate_labels = ["urgent", "artificial intelligence", "computer", "travel", "animal", "fiction"]

# We want to understand better the performance of the classifier over synthetic data. Just for curiosity
file_path = "data/text_test_data.csv"
output_file_path = "data/text_test_classified.csv"
true_labels, predicted_labels = classify_texts(file_path, output_file_path, classifier, candidate_labels)

# Then we calculate and print the classification report
print("\nReporte de clasificación:")
print(classification_report(true_labels, predicted_labels, zero_division=1, labels=candidate_labels, target_names=candidate_labels))

# To finish the challenge, we classify the texts in text_stext_samples_classified.csv using the pipeline and save the results in a CSV file.
file_path = "data/text_sample_data.csv"
output_file_path = "data/text_samples_classified.csv"
true_labels, predicted_labels = classify_texts(file_path, output_file_path, classifier, candidate_labels)