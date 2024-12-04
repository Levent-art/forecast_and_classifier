import pandas as pd

def classify_texts(file_path, output_file_path, classifier, candidate_labels):
    """
    Classify texts from a CSV file and save the results to another CSV file.

    Parameters:
    file_path (str): Path to the input CSV file containing texts.
    output_file_path (str): Path to the output CSV file to save the classified results.
    classifier (function): The classifier function to use for text classification.
    candidate_labels (list): List of candidate labels for classification.

    Returns:
    None
    """
    # Load the texts from the CSV file
    sample_texts = pd.read_csv(file_path)
    texts = list(sample_texts['text'])

    # Classify the texts
    results = classifier(texts, candidate_labels=candidate_labels)
    predicted_labels = [result["labels"][0] for result in results]
    true_labels = sample_texts['label']

    # Save the results as a CSV file
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
    results_df.to_csv(output_file_path, index=False)

    return true_labels, predicted_labels