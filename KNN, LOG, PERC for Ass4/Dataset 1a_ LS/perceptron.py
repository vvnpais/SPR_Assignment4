import numpy as np
from util import read_file, split_data
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, classification_report
import json

class PerceptronBinary:
    def __init__(self, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def fit(self, X_train, y_train):
        self.weights = np.random.randn(X_train.shape[1])
        self.bias = np.random.randn()

        epoch = 0
        while epoch < self.max_epochs:
            Nm = 0
            misclassified_points = []
            for i in range(len(X_train)):
                linear_output = np.dot(X_train[i], self.weights) + self.bias
                y_pred = self.step_function(linear_output)

                if y_pred != y_train[i]:
                    misclassified_points.append((y_train[i], X_train[i]))
                    Nm += 1

            if Nm == 0:  # Convergence criterion
                break

            # Update weights and bias
            if Nm > 0:
                delta_w = (self.learning_rate / Nm) * np.sum(
                    [y_true * x for y_true, x in misclassified_points], axis=0
                )
                self.weights += delta_w
                self.bias += (self.learning_rate / Nm) * np.sum(
                    [y_true for y_true, _ in misclassified_points]
                )

            epoch += 1

    def predict(self, X_test):
        linear_output = np.dot(X_test, self.weights) + self.bias
        return np.array([self.step_function(x) for x in linear_output])

    def step_function(self, x):
        return 1 if x >= 0 else -1


class PerceptronOvO:
    def __init__(self, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.classifiers = {}

    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        classes = np.unique(y_train)

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                class_i, class_j = classes[i], classes[j]

                # Create a boolean mask for filtering
                indices = (y_train == class_i) | (y_train == class_j)
                if not isinstance(indices, np.ndarray):  # Ensure `indices` is a boolean array
                    indices = np.array(indices, dtype=bool)

                X_subset = X_train[indices] 
                y_subset = y_train[indices]  

                # Relabel to +1 and -1 for binary classification
                y_binary = np.where(y_subset == class_i, 1, -1)

                perceptron = PerceptronBinary(
                    learning_rate=self.learning_rate, max_epochs=self.max_epochs
                )
                perceptron.fit(X_subset, y_binary)

                self.classifiers[(class_i, class_j)] = perceptron


    def predict(self, X_test):
        # Initialize vote counters
        votes = np.zeros((X_test.shape[0], len(self.classifiers)))

        # Predict using each binary classifier
        for idx, ((class_i, class_j), perceptron) in enumerate(self.classifiers.items()):
            preds = perceptron.predict(X_test)

            # Increment votes based on predictions
            votes[:, idx] = np.where(preds == 1, class_i, class_j)

        # Get final prediction by majority vote
        y_pred = [np.bincount(votes[i].astype(int)).argmax() for i in range(len(X_test))]
        return np.array(y_pred)

def plot_decision_boundaries(X_train, y_train, perceptron_ovo, resolution=0.01):
    # Define a grid over the feature space
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))

    # Predict for each point in the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron_ovo.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundaries
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)

    # Plot the training points
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, edgecolor='k', cmap=plt.cm.Paired)

    # Adding legend for class labels
    handles, labels = scatter.legend_elements()
    plt.legend(handles, [f"Class {int(label)}" for label in np.unique(y_train)])
    plt.title("Decision Boundaries between Classes")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(f"./Perceptron Result/Perceptron Decision Boundary", dpi=300) 
    plt.close() 
    print(f"Plot saved as 'Perceptron Decision Boundary'")

# Load Dataset-1
class1 = read_file("./train/Class1.txt")
class2 = read_file("./train/Class2.txt")
class3 = read_file("./train/Class3.txt")

test1 = read_file("./test/Class1.txt")
test2 = read_file("./test/Class2.txt")
test3 = read_file("./test/Class3.txt")
combined_list = class1 + class2 + class3
combined_array = np.array(combined_list)

X_train = np.array(class1 + class2 + class3)
y_train = np.array([0] * len(class1) + [1] * len(class2) + [2] * len(class3))
X_test = np.array(test1 + test2 + test3)
y_test = np.array([0] * len(test1) + [1] * len(test2) + [2] * len(test3))

perceptron_ovo = PerceptronOvO(learning_rate=0.01, max_epochs=1000)
perceptron_ovo.fit(X_train, y_train)

y_pred = perceptron_ovo.predict(X_test)
accuracy = np.mean(y_test == y_pred)
print(f"Accuracy of OvO Perceptron: {accuracy:.2f}")

# Plot Decision Boundaries
plot_decision_boundaries(X_train, y_train, perceptron_ovo)

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

# Save confusion matrix and full classification report to JSON
conf_matrix_dict = conf_matrix.tolist()
with open('./Perceptron Result/confusion_matrix.json', 'w') as f:
    json.dump(conf_matrix_dict, f, indent=4)

with open('./Perceptron Result/classification_report.json', 'w') as f:
    json.dump(class_report, f, indent=4)

print("Confusion matrix and classification report saved in JSON format.")
