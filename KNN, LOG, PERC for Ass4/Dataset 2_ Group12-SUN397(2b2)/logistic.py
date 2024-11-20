import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from matplotlib.colors import ListedColormap
import os, ast
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Logistic Regression OvR class definition
class LogisticRegressionOvR:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.classes = np.unique(y)
        self.weights = np.zeros((len(self.classes), num_features + 1))  # One weight vector per class

        X = np.c_[np.ones(num_samples), X]  # Add bias term
        for i, cls in enumerate(self.classes):
            y_binary = (y == cls).astype(int)  # Convert to binary problem for class cls
            for _ in range(self.epochs):
                predictions = self.sigmoid(X @ self.weights[i])
                gradient = X.T @ (predictions - y_binary) / num_samples
                self.weights[i] -= self.learning_rate * gradient

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        scores = X @ self.weights.T  # Compute scores for all classes
        probabilities = softmax(scores, axis=1)  # Convert to probabilities
        return np.argmax(probabilities, axis=1)  # Return class with highest probability

# Function to plot decision boundaries
def plot_decision_boundary_lr(X_train, y_train, model, resolution=0.02):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    # Create grid of points
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # Combine into (N, 2)

    # Predict for the entire grid
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)

    # Overlay training data
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
    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Read data from files
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

# Function to evaluate the model
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

# Main workflow
X_train, y_train = stack_lists_from_subdirs("./train")
X_test, y_test = stack_lists_from_subdirs("./test")

# Convert lists to NumPy arrays for computation
X_train = np.vstack(X_train)
X_test = np.vstack(X_test)

# Unique classes in the dataset
unique_classes = np.unique(y_train)

# Train and evaluate the model
model = LogisticRegressionOvR(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate performance
conf_matrix_df, class_report = evaluate_model(y_test, y_pred, unique_classes)

# Save Confusion Matrix to JSON
conf_matrix_dict = conf_matrix_df.to_dict()
with open('./Logistic Result/confusion_matrix.json', 'w') as f:
    json.dump(conf_matrix_dict, f, indent=4)

# Save Classification Report to JSON
with open('./Logistic Result/classification_report.json', 'w') as f:
    json.dump(class_report, f, indent=4)

# Display results
print(f"Accuracy: {np.mean(y_pred == y_test):.2f}\n")
print("Confusion Matrix:")
print(conf_matrix_df)
print("\nClassification Report:")
for cls, metrics in class_report.items():
    if cls not in ['accuracy', 'macro avg', 'weighted avg']:  # Exclude averages
        print(f"Class {cls}: F1-score = {metrics['f1-score']:.2f}")

print("\nConfusion Matrix and Classification Report saved as JSON.")
