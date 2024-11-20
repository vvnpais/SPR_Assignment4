import numpy as np
from util import euclidean_distance, read_file, split_data
import os
import ast
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json

def knn_density_estimation(data, query_point, K):
    distances = euclidean_distance(data, query_point)
    if K > len(distances):
        K = len(distances)
    kth_distance = np.sort(distances)[K-1]
    volume = (np.pi ** (data.shape[1] / 2) / np.math.gamma(data.shape[1] / 2 + 1)) * (kth_distance**data.shape[1])
    density = K / (len(data) * volume + 1e-8)  # To prevent division by zero
    return density

# Bayes classifier
def bayes_classifier(X_train, y_train, X_test, K):
    classes = np.unique(y_train)
    priors = {c: np.mean(y_train == c) for c in classes}
    predictions = []
    
    for query in X_test:
        posteriors = {}
        for c in classes:
            class_data = X_train[y_train == c]
            likelihood = knn_density_estimation(class_data, query, K)
            posterior = likelihood * priors[c]
            posteriors[c] = posterior
        
        predictions.append(max(posteriors, key=posteriors.get))
    return np.array(predictions)

# Evaluation
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Read data from files
def read_list_from_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read().strip()
        python_list = ast.literal_eval(text)
    return python_list

def stack_lists_from_subdirs(main_directory):
    stacked_lists = []
    labels = []

    for label, subdir in enumerate(sorted(os.listdir(main_directory))):  # Sorted for consistent ordering
        subdir_path = os.path.join(main_directory, subdir)
        if os.path.isdir(subdir_path): 
            for file_name in os.listdir(subdir_path):
                if file_name.endswith('.txt'):  # Check for .txt files
                    file_path = os.path.join(subdir_path, file_name)
                    file_list = read_list_from_file(file_path)
                    stacked_lists.append(file_list)
                    labels.append(label)  # Label based on the subdirectory index
    return np.array(stacked_lists), np.array(labels)

# Main workflow
X_train, y_train = stack_lists_from_subdirs("./train")
X_test, y_test = stack_lists_from_subdirs("./test")

X_train = np.vstack(X_train)
X_test = np.vstack(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

K_vals = np.concatenate((np.array([1, 3, 5, 7, 9, 15, 25, 35, 45]), np.arange(51, 351, 50)))
class_labels = np.unique(y_train)

# Initialize dictionaries to save results
confusion_matrices = {}
classification_reports = {}

for K in K_vals:
    print(f"\nK = {K}")
    y_pred = bayes_classifier(X_train, y_train, X_test, K)
    acc = accuracy(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    confusion_matrices[int(K)] = conf_matrix.tolist()
    
    classification_reports[int(K)] = class_report

# Save confusion matrices to JSON
with open('./KNN Result/confusion_matrices.json', 'w') as f:
    json.dump(confusion_matrices, f, indent=4)

# Save classification reports (with all metrics) to JSON
with open('./KNN Result/classification_reports.json', 'w') as f:
    json.dump(classification_reports, f, indent=4)
