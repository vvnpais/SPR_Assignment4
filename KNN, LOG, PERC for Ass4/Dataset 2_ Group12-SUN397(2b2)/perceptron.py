import json
import os
import ast
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multiclass import OneVsOneClassifier

# Load dataset
def read_list_from_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read().strip()
        python_list = ast.literal_eval(text)
    return python_list

def stack_lists_from_subdirs(main_directory):
    stacked_lists = []
    labels = []

    # Enumerate over subdirectories
    for label, subdir in enumerate(sorted(os.listdir(main_directory))):  # Sorted for consistent ordering
        subdir_path = os.path.join(main_directory, subdir)
        if os.path.isdir(subdir_path):  # Check if it is a subdirectory
            for file_name in os.listdir(subdir_path):
                if file_name.endswith('.txt'):  # Check for .txt files
                    file_path = os.path.join(subdir_path, file_name)
                    file_list = read_list_from_file(file_path)
                    stacked_lists.append(file_list)
                    labels.append(label)  # Label based on the subdirectory index
    return np.array(stacked_lists), np.array(labels)

X_train, y_train = stack_lists_from_subdirs("./train")
X_test, y_test = stack_lists_from_subdirs("./test")

X_train = np.vstack(X_train)
X_test = np.vstack(X_test)

model = Perceptron(max_iter=1000, eta0=0.01, random_state=42)
ovo_model = OneVsOneClassifier(model)
ovo_model.fit(X_train, y_train)

def evaluate_model(y_true, y_pred, class_labels):
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=class_labels)

    # Convert confusion matrix to a labeled DataFrame
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=[f"Actual {cls}" for cls in class_labels],
        columns=[f"Predicted {cls}" for cls in class_labels]
    )
    class_report = classification_report(y_true, y_pred, labels=class_labels, output_dict=True)
    f1_scores = {cls: class_report[str(cls)]["f1-score"] for cls in class_labels}

    return conf_matrix_df, class_report

unique_classes = np.unique(y_train)
y_pred = ovo_model.predict(X_test)

conf_matrix_df, class_report = evaluate_model(y_test, y_pred, unique_classes)

# Save Confusion Matrix to JSON
conf_matrix_dict = conf_matrix_df.to_dict()
with open('./Perceptron Result/confusion_matrix.json', 'w') as f:
    json.dump(conf_matrix_dict, f, indent=4)

# Save Classification Report to JSON
with open('./Perceptron Result/classification_report.json', 'w') as f:
    json.dump(class_report, f, indent=4)

# Print Accuracy and Results
print(f"Accuracy: {np.mean(y_pred == y_test):.2f}\n")
print("Confusion Matrix:")
print(conf_matrix_df)
print("\nClassification Report:")
for cls, metrics in class_report.items():
    if cls not in ['accuracy', 'macro avg', 'weighted avg']:  # Exclude averages
        print(f"Class {cls}: F1-score = {metrics['f1-score']:.2f}")

print("\nConfusion Matrix and Classification Report saved as JSON.")
