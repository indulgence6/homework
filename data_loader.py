import numpy as np
import pickle
import os

def load_cifar10(path):
    def _load_batch(filename):
        with open(filename, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
        return batch['data'], batch['labels']
    
    # Load training data
    train_data = []
    train_labels = []
    for i in range(1, 6):
        data, labels = _load_batch(os.path.join(path, f'data_batch_{i}'))
        train_data.append(data)
        train_labels.append(labels)
    X_train = np.concatenate(train_data).astype(np.float32) / 255.0
    y_train = np.concatenate(train_labels)
    
    # Load test data
    X_test, y_test = _load_batch(os.path.join(path, 'test_batch'))
    X_test = X_test.astype(np.float32) / 255.0
    y_test = np.array(y_test)
    
    # Split validation set
    val_size = 5000
    X_train, X_val = X_train[:-val_size], X_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test