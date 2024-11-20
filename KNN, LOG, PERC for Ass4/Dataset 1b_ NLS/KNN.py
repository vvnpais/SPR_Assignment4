import numpy as np
from util import euclidean_distance, read_file, split_data
import os
import ast
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
    density = K / (len(data) * volume+0.00000001)
    return density

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

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def plot_decision_boundary(X_train, y_train, classifier, K, resolution=0.02):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = classifier(X_train, y_train, grid_points, K)
    predictions = predictions.reshape(xx.shape)

    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    plt.contourf(xx, yy, predictions, alpha=0.8, cmap=cmap)

    colors = ['red', 'green', 'blue']
    for idx, label in enumerate(np.unique(y_train)):
        plt.scatter(
            X_train[y_train == label, 0],
            X_train[y_train == label, 1],
            c=colors[idx],
            label=f"Class {label}",
            edgecolor='k'
        )

    plt.legend()
    plt.title(f"KNN Decision Boundary (K={K})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(f"./KNN Result/KNN Decision Boundary (K={K})", dpi=300) 
    plt.close() 
    print(f"Plot saved as 'KNN Decision Boundary (K={K})'")

class1 = read_file("./train/class1.txt")
class2 = read_file("./train/class2.txt")
class3 = read_file("./train/class3.txt")

test1 = read_file("./test/class1.txt")
test2 = read_file("./test/class2.txt")
test3 = read_file("./test/class3.txt")

combined_list = class1 + class2 + class3
combined_array = np.array(combined_list)

X_train = np.array(class1 + class2 + class3)
y_train = np.array([0] * len(class1) + [1] * len(class2) + [2] * len(class3))
X_test = np.array(test1 + test2 + test3)
y_test = np.array([0] * len(test1) + [1] * len(test2) + [2] * len(test3))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

arr = np.arange(51, 351, 50)
pre = np.array([1, 3, 5, 7, 9, 15, 25, 35, 45])
K_vals = np.concatenate((pre, arr))

confusion_matrices = {}
classification_reports = {}

for K in K_vals:
    print(f"\nK = {K}")
    y_pred = bayes_classifier(X_train, y_train, X_test, K)
    acc = accuracy(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    
    plot_decision_boundary(X_train, y_train, bayes_classifier, K)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    confusion_matrices[int(K)] = conf_matrix.tolist()

    classification_reports[int(K)] = classification_report(y_test, y_pred, output_dict=True)

with open(f'./KNN Result/confusion_matrices K={K}.json', 'w') as f:
    json.dump(confusion_matrices, f, indent=4)
with open(f'./KNN Result/classification_reports K={K}.json', 'w') as f:
    json.dump(classification_reports, f, indent=4)
