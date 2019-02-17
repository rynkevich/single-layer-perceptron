import numpy as np


class Perceptron:
    BIAS = 1.0

    def __init__(self, vector_size, class_count=2, learning_rate=0.1):
        self.vector_size = vector_size + 1
        self.class_count = class_count
        self.learning_rate = learning_rate
        self._weights = np.zeros((class_count, self.vector_size))

    def guess(self, vector):
        input_vector = (self.BIAS, ) + vector
        activation_values = self._get_activation_values(input_vector)
        return self._activate(activation_values)

    def train(self, training_data):
        has_classification_errors = False
        for vector, target_class in training_data:
            input_vector = np.array((self.BIAS, ) + vector)
            activation_values = self._get_activation_values(input_vector)
            guess = self._activate(activation_values)
            error = target_class - guess

            if error != 0:
                has_classification_errors = True

                self._weights[target_class] += self.learning_rate * input_vector

                for i in range(self.class_count):
                    if i != target_class and activation_values[i] >= activation_values[target_class]:
                        self._weights[i] -= self.learning_rate * input_vector

        return has_classification_errors

    def _get_activation_values(self, vector):
        activation_values = []
        input_vector = np.array(vector)
        for i in range(self.class_count):
            activation_values.append(self._weights[i].dot(input_vector))

        return activation_values

    @staticmethod
    def _activate(activation_values):
        return np.argmax(activation_values)
