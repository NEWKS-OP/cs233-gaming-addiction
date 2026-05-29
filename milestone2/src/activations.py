import numpy as np

class Sigmoid:
    @staticmethod
    def forward(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def gradient(z):
        return Sigmoid.forward(z)*(1 - Sigmoid.forward(z))

class ReLU:
    @staticmethod
    def forward(z):
        return np.maximum(z , 0)

    @staticmethod
    def gradient(z):
        return (z > 0).astype(float)

class Identity:
    @staticmethod
    def forward(z):
        return z

    @staticmethod
    def gradient(z):
        return np.ones_like(z)