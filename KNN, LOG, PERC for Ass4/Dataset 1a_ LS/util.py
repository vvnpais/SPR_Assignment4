import random
import numpy as np

def read_file(filename):
    with open(filename, 'r') as file:
        data = []
        for line in file:
            x, y = list(map(float, line.strip().split()))
            data.append((x, y))
    return data

def split_data(data, train_size = 0.7):
    random.seed(10)
    random.shuffle(data)
    split_index = int(len(data) * train_size)
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data
    
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2, axis=1))