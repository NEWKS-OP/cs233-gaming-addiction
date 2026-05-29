import numpy as np

class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        ### WRITE YOUR CODE HERE
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def gradient(y_true, y_pred):
        ### WRITE YOUR CODE HERE
        return 2 * (y_pred - y_true) / y_true.shape[0]
    
class CrossEntropy:

    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One-hot encoded truth vector, shape (N, C).
        :param y_pred: (array) Predicted probabilities (after softmax), shape (N, C).
        :return: (flt) Average cross-entropy over the batch.
        """
        # eps to avoid log(0) and numerical instability
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

    @staticmethod
    def gradient(y_true, y_pred):
        """
        Returns the simplified gradient (y_pred - y_true) / N, valid only when
        y_pred is the output of a Softmax. This relies on the chain rule
        simplification between softmax and cross-entropy.
        """
        N = y_true.shape[0]
        return (y_pred - y_true) / N
