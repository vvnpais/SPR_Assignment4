import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, classification_report
from util import euclidean_distance, read_file, split_data
from matplotlib.colors import ListedColormap
import json

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
    plt.savefig(f"./Logistic Result/Logistic Decision Boundary", dpi=300) 
    plt.close() 
    print(f"Plot saved as \'Logistic Decision Boundary\'")

# Load Dataset-1
class1 = np.array(read_file("./train/class1.txt"))
class2 = np.array(read_file("./train/class2.txt"))
class3 = np.array(read_file("./train/class3.txt"))

test1 = np.array(read_file("./test/class1.txt"))
test2 = np.array(read_file("./test/class2.txt"))
test3 = np.array(read_file("./test/class3.txt"))

X_train = np.vstack((class1, class2, class3))
y_train = np.array([0] * len(class1) + [1] * len(class2) + [2] * len(class3))
X_test = np.vstack((test1, test2, test3))
y_test = np.array([0] * len(test1) + [1] * len(test2) + [2] * len(test3))

# Train and visualize Logistic Regression OvR
model = LogisticRegressionOvR(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)
plot_decision_boundary_lr(X_train, y_train, model)

# Evaluate
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

with open('./Logistic Result/confusion_matrix.json', 'w') as f:
    json.dump(conf_matrix.tolist(), f)

# Classification Report (F-score, Precision, Recall)
class_report = classification_report(y_test, y_pred, output_dict=True)
print("\nClassification Report:")
print(class_report)

with open('./Logistic Result/classification_report.json', 'w') as f:
    json.dump(class_report, f, indent=4)