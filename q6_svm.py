from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import accuracy_score
import numpy as np
import os
import matplotlib.pyplot as plt
import json_tricks as json


def plot_decision_boundaries(X, y, model, title, filename, param_details=None):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True)

    if param_details is not None:
        caption = f"Kernel: {param_details.get('kernel', 'N/A')}\n" + \
            f"C={param_details.get('C', 'N/A')}\n" + \
            f"gamma={param_details.get('gamma', 'N/A')}\n" + \
            f"degree={param_details.get('degree', 'N/A')}"
        plt.text(0.5, 0.05, caption, transform=plt.gca().transAxes, fontsize=8,
                 verticalalignment='bottom', horizontalalignment='center',
                 bbox=dict(facecolor='white', alpha=0.7))

    plt.savefig(filename)
    plt.close()


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


def svm_classification(X_train, X_test, y_train, y_test, datafile, kernel_type='linear', params=None, dataset_name='.', plot=True):
    svm_model = SVC(kernel=kernel_type)

    all_metrics = {}

    if params is not None:
        for param_set in ParameterGrid(params):
            model = SVC(kernel=kernel_type, **param_set)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = calculate_metrics(y_test, y_pred, len(np.unique(y_test)))
            all_metrics[str(param_set)] = metrics

            print(f"Parameters for {kernel_type} kernel: {
                  param_set}", file=datafile)
            print(f"Accuracy for {kernel_type} kernel: {
                  metrics['accuracy']}", file=datafile)
            print(
                f"Classification Report for {kernel_type} kernel:\n{json.dumps(metrics, indent=4)}", file=datafile)

            if plot:
                plot_decision_boundaries(
                    X_train,
                    y_train,
                    model,
                    f"{dataset_name}/SVM Decision Boundary ({kernel_type} kernel) {
                        param_set}",
                    f"{dataset_name}/svm_{kernel_type}_decision_boundary_{param_set}.png",
                    param_details=param_set | {'kernel': kernel_type})

    else:
        model = svm_model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred, len(np.unique(y_test)))
        all_metrics[str({})] = metrics

        print(
            f"Accuracy for {kernel_type} kernel: {metrics['accuracy']}", file=datafile)
        print(
            f"Classification Report for {kernel_type} kernel:\n{metrics}",
            file=datafile)

        if plot:
            plot_decision_boundaries(
                X_train,
                y_train,
                model,
                f"{dataset_name}/SVM Decision Boundary ({kernel_type} kernel)",
                f"{dataset_name}/svm_{kernel_type}_decision_boundary.png",
                param_details={})

    return all_metrics


def svm_experiments(X_train, X_test, y_train, y_test, datafile, kernel_types=['linear', 'poly', 'rbf'], dataset_name='.', plot=True):
    param_grids = {
        'linear': {'C': [0.1, 1, 10]},
        'poly': {'C': [0.1, 1, 10], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto']},
        'rbf': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    }

    results = {}
    for kernel in kernel_types:
        print(f"\nTraining SVM with {kernel} kernel:", file=datafile)
        metrics = svm_classification(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            kernel_type=kernel,
            params=param_grids.get(kernel, None),
            dataset_name=dataset_name,
            plot=plot,
            datafile=datafile)
        results[kernel] = metrics

    return results


# DATASET 1
for dataset_name in ['Dataset 1a: LS', 'Dataset 1b: NLS']:
    os.makedirs(dataset_name, exist_ok=True)
    datafile = open(f'svm_{dataset_name}.txt', 'w')
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

    accuracies = svm_experiments(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        dataset_name=dataset_name,
        datafile=datafile)

# DATASET 2
for dataset_name in ['Dataset 2: Group12-SUN397(2b2)']:
    os.makedirs(dataset_name, exist_ok=True)
    datafile = open(f'svm_{dataset_name}.txt', 'w')
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

    accuracies = svm_experiments(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        dataset_name=dataset_name,
        plot=False, datafile=datafile)
