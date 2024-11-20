import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import os
import json_tricks as json

DATASET_DIR = 'DATASET_ASSIGNMENT4'
dataset_name = 'Dataset 2: Group12-SUN397(2b2)'
dataset = os.path.join(DATASET_DIR, dataset_name)

X_train = []
X_test = []
y_train = []
y_test = []

datafile = open('pca_gmm.txt', 'w')

for i, dir in enumerate(os.listdir(os.path.join(dataset, 'train'))):
    for filename in os.listdir(os.path.join(dataset, 'train', dir)):
        traindata_sample = open(os.path.join(
            dataset, 'train', dir, filename), 'r').read()
        X_train.append(eval(traindata_sample))
        y_train.append(i)
X_train = np.array(X_train)
y_train = np.array(y_train)

for i, dir in enumerate(os.listdir(os.path.join(dataset, 'test'))):
    for filename in os.listdir(os.path.join(dataset, 'test', dir)):
        testdata_sample = open(os.path.join(
            dataset, 'test', dir, filename), 'r').read()
        X_test.append(eval(testdata_sample))
        y_test.append(i)
X_test = np.array(X_test)
y_test = np.array(y_test)


def calculate_metrics(y_true, y_pred, k):
    confusion_matrix = np.zeros((k, k))
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true, pred] += 1

    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    precision = np.zeros(k)
    recall = np.zeros(k)
    f_measure = np.zeros(k)

    for i in range(k):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        precision[i] = tp / (tp + fp) if tp + fp != 0 else 0
        recall[i] = tp / (tp + fn) if tp + fn != 0 else 0
        f_measure[i] = 2 * (precision[i] * recall[i]) / (precision[i] +
                                                         recall[i]) if precision[i] + recall[i] != 0 else 0

    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_f_measure = np.mean(f_measure)

    return {
        'accuracy': accuracy,
        'precision_per_class': precision.tolist(),
        'mean_precision': mean_precision.tolist(),
        'recall_per_class': recall.tolist(),
        'mean_recall': mean_recall.tolist(),
        'f_measure_per_class': f_measure.tolist(),
        'mean_f_measure': mean_f_measure.tolist(),
        'confusion_matrix': confusion_matrix.tolist()
    }


def pca_gmm_bayes(X_train, X_test, y_train, y_test, l_values, n_mixtures, output_dir):
    results = []
    for l in l_values:
        pca = PCA(n_components=l)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        eigenvalues = pca.explained_variance_

        full_eigenvalues = np.linalg.svd(
            X_train)[1]**2 / (X_train.shape[0] - 1)

        plt.figure(figsize=(8, 6))
        plt.plot(np.sort(full_eigenvalues),
                 marker='o', linestyle='-', color='b')
        plt.title(f"Eigenvalues in Ascending Order (All Eigenvalues)")
        plt.xlabel("Component Index")
        plt.ylabel("Eigenvalue (Explained Variance)")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"all_eigenvalues_pca.png"))
        plt.close()

        for n in n_mixtures:
            gmm_models = {}
            for label in np.unique(y_train):
                gmm = GaussianMixture(
                    n_components=n, random_state=42, covariance_type='full')
                gmm.fit(X_train_pca[y_train == label])
                gmm_models[label] = gmm

            y_pred = []
            for x in X_test_pca:
                log_likelihoods = {label: gmm.score_samples(
                    [x])[0] for label, gmm in gmm_models.items()}
                y_pred.append(max(log_likelihoods, key=log_likelihoods.get))

            metrics = calculate_metrics(
                y_test, y_pred, len(np.unique(y_train)))
            results.append((l, n, metrics))
            print(json.dumps(metrics, indent=4), file=datafile)
            print(
                f"PCA Components: {l}, GMM Mixtures: {
                    n}, Accuracy: {metrics['accuracy']:.4f}",
                file=datafile
            )

        if l == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                        c=y_train, cmap='viridis', edgecolors='k', marker='o')
            plt.title(f"PCA Reduced Dimensionality (PCA Components: {l})")
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.colorbar(label='Class')
            plt.savefig(os.path.join(
                output_dir, f"pca_2d_representation_{l}_components.png"))
            plt.close()

    return results


l_values = [1, 2, 4, 8]
n_mixtures = [1, 2, 4, 8]

output_dir = 'pca_gmm_results'
os.makedirs(output_dir, exist_ok=True)

results = pca_gmm_bayes(X_train, X_test, y_train, y_test,
                        l_values, n_mixtures, output_dir)
