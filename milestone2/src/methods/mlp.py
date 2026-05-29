import numpy as np

class MLP:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: list of dimensions of the neural net. (input, hidden layer, ... ,hidden layer, output)
        :param activations: list of activation functions. Must contain N-1 activation function, where N = len(dimensions).

        Example of one hidden layer with
        - 2 inputs
        - 10 hidden nodes
        - 5 outputs
        layers -->    [0,        1,          2]
        ----------------------------------------
        
        """

        ### WRITE YOUR CODE HERE
        self.n_layers = len(dimensions)
        self.loss = None
        self.learning_rate = None

        # Weights/biases indexed by layer l (from 1 to n_layers-1).
        self.w = {}
        self.b = {}
        self.activations = {}

        for l in range(1, self.n_layers):
            self.w[l] = np.random.randn(dimensions[l], dimensions[l - 1]) / np.sqrt(dimensions[l - 1])
            self.b[l] = np.zeros(dimensions[l])
            self.activations[l] = activations[l - 1]


    def feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """

        ### WRITE YOUR CODE HERE
        a = {}
        z = {0: x}  # layer 0 := input

        for l in range(1, self.n_layers):
            a[l] = z[l - 1] @ self.w[l].T + self.b[l]
            z[l] = self.activations[l].forward(a[l])

        return a, z


    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """

        ### WRITE YOUR CODE HERE
        _, z = self.feed_forward(x)
        return z[self.n_layers - 1]


    def back_prop(self, z, a, y_true, loss):
        """
        The input dicts keys represent the layers of the net.
        a = { 0: x,
              1: f(w1(x) + b1)
              2: f(w2(a2) + b2)
              }
        :param a: (dict) w^T@x + b
        :param z: (dict) f(a)
        :param y_true: (array) One hot encoded truth vector.
        :param loss: Loss class with a static .gradient(y_true, y_pred) method.
        :return:
        """

        ### WRITE YOUR CODE HERE
        L = self.n_layers - 1

        # Output layer: delta^(L) = dL/dz^(L) ⊙ sigma'(a^(L))
        delta = loss.gradient(y_true, z[L])

        # Walk back through the layers.
        for l in range(L, 0, -1):
            # Gradients for layer l.
            dw = delta.T @ z[l - 1]          # (dim[l], dim[l-1]), same shape as w[l]
            db = np.sum(delta, axis=0)        # (dim[l],), same shape as b[l]

            # Propagate delta to the previous layer (before overwriting w[l]).
            if l > 1:
                delta = (delta @ self.w[l]) * self.activations[l - 1].gradient(a[l - 1])

            self.update_w_b(l, dw, db)


    def update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        ### WRITE YOUR CODE HERE
        self.w[index] = self.w[index] - self.learning_rate * dw
        self.b[index] = self.b[index] - self.learning_rate * delta

    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3, x_val=None, y_val=None):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        """

        ### WRITE YOUR CODE HERE
        self.loss = loss
        self.learning_rate = learning_rate
        self.train_losses = []
        self.val_losses = []

        N = x.shape[0]

        for epoch in range(epochs):
            # Shuffle the data each epoch (stochasticity helps escape bad regions).
            perm = np.random.permutation(N)
            x_shuffled = x[perm]
            y_shuffled = y_true[perm]

            for start in range(0, N, batch_size):
                end = start + batch_size  # last batch may be smaller; slicing handles it
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                a, z = self.feed_forward(x_batch)
                self.back_prop(z, a, y_batch, loss)

            train_pred = self.predict(x)
            self.train_losses.append(loss.loss(y_true, train_pred))

            if x_val is not None and y_val is not None:
                val_pred = self.predict(x_val)
                self.val_losses.append(loss.loss(y_val, val_pred))
