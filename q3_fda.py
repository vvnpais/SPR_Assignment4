import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal
from itertools import combinations
from collections import Counter
import os


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


def perform_pairwise_fda_and_bayes(X_train, X_test, y_train, y_test, n_classes, gmm_components=1):
    class_pairs = list(combinations(range(n_classes), 2))
    lda_models = {}
    pairwise_predictions_gaussian = []
    pairwise_predictions_gmm = []

    for c1, c2 in class_pairs:
        idx_train = np.logical_or(y_train == c1, y_train == c2)
        idx_test = np.logical_or(y_test == c1, y_test == c2)

        X_train_pair = X_train[idx_train]
        y_train_pair = y_train[idx_train]
        X_test_pair = X_test[idx_test]
        y_test_pair = y_test[idx_test]

        y_train_pair_binary = np.where(y_train_pair == c1, 0, 1)
        y_test_pair_binary = np.where(y_test_pair == c1, 0, 1)

        lda = LDA(n_components=1)
        X_train_fda = lda.fit_transform(X_train_pair, y_train_pair_binary)
        X_test_fda = lda.transform(X_test_pair)
        lda_models[(c1, c2)] = lda

        means = []
        covariances = []
        for binary_class in [0, 1]:
            X_c = X_train_fda[y_train_pair_binary == binary_class]
            means.append(np.mean(X_c, axis=0))
            covariances.append(np.cov(X_c, rowvar=False) if len(
                X_c) > 1 else np.eye(X_c.shape[1]))

        y_pred_gaussian = []
        for x in X_test_fda:
            posteriors = [
                multivariate_normal.pdf(
                    x, mean=means[binary_class], cov=covariances[binary_class])
                for binary_class in [0, 1]
            ]
            y_pred_gaussian.append(c1 if np.argmax(posteriors) == 0 else c2)

        gmm_0 = GaussianMixture(n_components=gmm_components, random_state=0).fit(
            X_train_fda[y_train_pair_binary == 0])
        gmm_1 = GaussianMixture(n_components=gmm_components, random_state=0).fit(
            X_train_fda[y_train_pair_binary == 1])

        y_pred_gmm = []
        for x in X_test_fda:
            posterior_0 = gmm_0.score_samples(x.reshape(1, -1))[0]
            posterior_1 = gmm_1.score_samples(x.reshape(1, -1))[0]
            y_pred_gmm.append(c1 if posterior_0 > posterior_1 else c2)

        pairwise_predictions_gaussian.append((idx_test, y_pred_gaussian))
        pairwise_predictions_gmm.append((idx_test, y_pred_gmm))

    def perform_max_voting(pairwise_preds, y_test):
        votes = np.zeros((len(y_test), n_classes))
        for idx_test, preds in pairwise_preds:
            for i, pred in zip(np.where(idx_test)[0], preds):
                votes[i, pred] += 1
        return np.argmax(votes, axis=1)

    y_pred_gaussian = perform_max_voting(pairwise_predictions_gaussian, y_test)
    y_pred_gmm = perform_max_voting(pairwise_predictions_gmm, y_test)

    metrics_gaussian = calculate_metrics(y_test, y_pred_gaussian, n_classes)
    metrics_gmm = calculate_metrics(y_test, y_pred_gmm, n_classes)

    print(f"Unimodal Gaussian Bayes Accuracy: {metrics_gaussian['accuracy']}")
    print(f"GMM Bayes Accuracy: {metrics_gmm['accuracy']}")

    return metrics_gaussian['accuracy'], metrics_gmm['accuracy']


n_classes = 3
gmm_components = 3

# DATASET 1
for dataset_name in ['Dataset 1a: LS', 'Dataset 1b: NLS']:
    DATASET_DIR = 'DATASET_ASSIGNMENT4'
    dataset = os.path.join(DATASET_DIR, dataset_name)
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i, file in enumerate(os.listdir(os.path.join(dataset, 'train'))):
        data = open(os.path.join(dataset, 'train', file),
                    'r').read().splitlines()
        while '' in data:
            data.remove('')
        for idx in range(len(data)):
            data[idx] = [float(x) for x in data[idx].strip(' ').split(' ')]
        X_train.extend(data)
        y_train.extend([i for x in range(len(data))])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(X_train.shape)
    print(y_train.shape)

    for i, file in enumerate(os.listdir(os.path.join(dataset, 'test'))):
        data = open(os.path.join(dataset, 'test', file),
                    'r').read().splitlines()
        while '' in data:
            data.remove('')
        for idx in range(len(data)):
            data[idx] = [float(x) for x in data[idx].strip(' ').split(' ')]
        X_test.extend(data)
        y_test.extend([i for x in range(len(data))])
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(X_test.shape)
    print(y_test.shape)

    acc_gaussian, acc_gmm = perform_pairwise_fda_and_bayes(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_classes=n_classes,
        gmm_components=gmm_components)

# DATASET 2
for dataset_name in ['Dataset 2: Group12-SUN397(2b2)']:
    DATASET_DIR = 'DATASET_ASSIGNMENT4'
    dataset = os.path.join(DATASET_DIR, dataset_name)
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i, dir in enumerate(os.listdir(os.path.join(dataset, 'train'))):
        for filename in os.listdir(os.path.join(dataset, 'train', dir)):
            traindata_sample = open(os.path.join(
                dataset, 'train', dir, filename), 'r').read()
            X_train.append(eval(traindata_sample))
            y_train.append(i)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(X_train.shape)
    print(y_train.shape)

    for i, dir in enumerate(os.listdir(os.path.join(dataset, 'test'))):
        for filename in os.listdir(os.path.join(dataset, 'test', dir)):
            testdata_sample = open(os.path.join(
                dataset, 'test', dir, filename), 'r').read()
            X_test.append(eval(testdata_sample))
            y_test.append(i)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(X_test.shape)
    print(y_test.shape)
    acc_gaussian, acc_gmm = perform_pairwise_fda_and_bayes(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_classes=n_classes,
        gmm_components=gmm_components)
