import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters

    def f_softmax(self, data, W):
            exp = np.exp(data @ W)
            return exp / np.sum(exp, axis=1).reshape(-1, 1)

    def gradient(self, data, labels_onehot, W):
        return data.T @ (self.f_softmax(data, W) - labels_onehot)

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        D = training_data.shape[1]
        C = get_n_classes(training_labels)
        labels_onehot = label_to_onehot(training_labels, C)
        self.W = np.random.normal(0, 0.1, (D, C))
        for _ in range(self.max_iters):
            self.W = self.W - self.lr * self.gradient(training_data, labels_onehot, self.W)
        pred_labels = onehot_to_label(self.f_softmax(training_data, self.W))
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        pred_labels = onehot_to_label(self.f_softmax(test_data, self.W))
        return pred_labels
