from sys import argv
from perceptron import Perceptron
from perceptron_alt import AlternativePerceptron
import plotter


def main():
    algorithm_number = int(argv[1])
    training_data_file = open(argv[2], 'r')
    test_data_file = open(argv[3], 'r')

    training_data = []
    for line in training_data_file.readlines():
        split_line = line.split()
        target_class = int(split_line[-1])
        training_data.append((tuple(float(x) for x in split_line[:-1]), target_class))

    test_data = []
    for line in test_data_file.readlines():
        test_data.append(tuple(float(x) for x in line.split()))

    if algorithm_number == 1:
        perceptron = Perceptron(vector_size=2)
    else:
        perceptron = AlternativePerceptron(vector_size=2)

    has_classification_errors = True
    while has_classification_errors:
        has_classification_errors = perceptron.train(training_data)

    classified_data = []
    for vector in test_data:
        classified_data.append((vector, perceptron.guess(vector)))

    plotter.draw_classes(training_data, 'Perceptron Classifier: Training Data', 1)
    plotter.draw_classes(classified_data, 'Perceptron Classifier: Classified Data', 2)
    plotter.show()


if __name__ == '__main__':
    main()
