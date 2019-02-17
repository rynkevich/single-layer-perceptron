import numpy as np


class AlternativePerceptron:
    BIAS = 1.0

    def __init__(self, vector_size, learning_rate=0.1):
        self.vector_size = vector_size + 1
        self.learning_rate = learning_rate
        self._weights = np.zeros(self.vector_size)

    def guess(self, vector):
        input_vector = (self.BIAS, ) + vector
        activation_value = 0.0
        for weight, component in zip(self._weights, input_vector):
            activation_value += weight * component

        return self._activate(activation_value)

    def train(self, training_data):
        has_classification_errors = False
        for vector, target_class in training_data:
            guess = self.guess(vector)
            error = target_class - guess

            input_vector = (self.BIAS,) + vector
            if error != 0:
                has_classification_errors = True

                for i in range(self.vector_size):
                    self._weights[i] += error * self.learning_rate * input_vector[i]

        return has_classification_errors


    @staticmethod
    def _activate(activation_value):
        return 1 if activation_value >= 0.0 else 0
