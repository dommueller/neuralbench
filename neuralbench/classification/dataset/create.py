import numpy as np

def createDataSet(choice, seed = 0):
    if choice == "mnist":
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
        # rescale the data, use the traditional train/test split
        X, y = mnist.data / 255., mnist.target
        test_size = 60000
        X_train, X_test = X[:test_size], X[60000:]
        y_train, y_test = y[:test_size], y[60000:]
        return (X_train, y_train, X_test, y_test)
    else:
        import tsPlaygroundDatasets
        from sklearn.cross_validation import train_test_split
        if choice == "spiral":
            data = tsPlaygroundDatasets.spiralData(500, 0.25, seed)
            train, test = train_test_split(data, test_size = 0.5, random_state=seed)
            X_train = train.as_matrix(['x', 'y'])
            y_train = train.as_matrix(['label'])
            X_test = test.as_matrix(['x', 'y'])
            y_test = test.as_matrix(['label'])
            return (X_train, y_train, X_test, y_test)
        elif choice == "xor":
            data = tsPlaygroundDatasets.xorData(500, 0.0, seed)
            train, test = train_test_split(data, test_size = 0.5, random_state=seed)
            X_train = train.as_matrix(['x', 'y'])
            y_train = train.as_matrix(['label'])
            X_test = test.as_matrix(['x', 'y'])
            y_test = test.as_matrix(['label'])
            return (X_train, y_train, X_test, y_test)
        elif choice == "circle":
            data = tsPlaygroundDatasets.circleData(500, 0.0, seed)
            train, test = train_test_split(data, test_size = 0.5, random_state=seed)
            X_train = train.as_matrix(['x', 'y'])
            y_train = train.as_matrix(['label'])
            X_test = test.as_matrix(['x', 'y'])
            y_test = test.as_matrix(['label'])
            return (X_train, y_train, X_test, y_test)
        elif choice == "gaussian":
            data = tsPlaygroundDatasets.gaussianData(500, 0.1, seed)
            train, test = train_test_split(data, test_size = 0.5, random_state=seed)
            X_train = train.as_matrix(['x', 'y'])
            y_train = train.as_matrix(['label'])
            X_test = test.as_matrix(['x', 'y'])
            y_test = test.as_matrix(['label'])
            return (X_train, y_train, X_test, y_test)
        else:
            print "Bad luck no known dataset"

def run_test_validate_splits(callback, X, y, folds=10):
    from sklearn.cross_validation import StratifiedKFold
    from tqdm import tqdm
    skf_test = StratifiedKFold(y.reshape(-1), n_folds=folds, shuffle=True)
    for i, (train_validate_index, test_index) in tqdm(enumerate(skf_test)):
        X_train_validate = X[train_validate_index]
        y_train_validate = y[train_validate_index]
        X_test = X[test_index]
        y_test = y[test_index]
        skf_validate = StratifiedKFold(y[train_validate_index].reshape(-1), n_folds=folds, shuffle=True)
        for j, (train_index, validate_index) in tqdm(enumerate(skf_validate)):
            X_train = X_train_validate[train_index]
            y_train = y_train_validate[train_index]
            X_validate = X_train_validate[validate_index]
            y_validate = y_train_validate[validate_index]

            callback(X_train, y_train, X_validate, y_validate, X_test, y_test, test_split=i, validate_split=j)

def run_validate_splits(callback, X_train_validate, y_train_validate, X_test, y_test, folds=10, seed=0):
    from sklearn.cross_validation import StratifiedKFold
    from tqdm import tqdm

    skf_validate = StratifiedKFold(y_train_validate.reshape(-1), n_folds=folds, shuffle=True, random_state=seed)
    for j, (train_index, validate_index) in tqdm(enumerate(skf_validate)):
        X_train = X_train_validate[train_index]
        y_train = y_train_validate[train_index]
        X_validate = X_train_validate[validate_index]
        y_validate = y_train_validate[validate_index]

        callback(X_train, y_train, X_validate, y_validate, X_test, y_test, test_split=0, validate_split=j)


# Taken from
# https://github.com/tensorflow/tensorflow/blob/1d76583411038767f673a0c96174c80eaf9ff42f/tensorflow/g3doc/tutorials/mnist/input_data.py
def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  labels_dense = np.array(labels_dense, dtype="int32")
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot




