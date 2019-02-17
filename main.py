from sys import argv
from random import randint
from perceptron import Perceptron
from perceptron_alt import AlternativePerceptron
import plotter


def random_color():
    return '#%06x' % randint(0, 0xFFF000)


def main():
    training_data_file = open(argv[1], 'r')
    test_data_file = open(argv[2], 'r')
    algorithm_number = int(argv[3])

    training_data = []
    for line in training_data_file.readlines():
        split_line = line.split()
        target_class = int(split_line[-1])
        training_data.append((tuple(float(x) for x in split_line[:-1]), target_class))

    test_data = []
    for line in test_data_file.readlines():
        test_data.append(tuple(float(x) for x in line.split()))

    if algorithm_number == 1:
        class_count = int(argv[4])
        perceptron = Perceptron(vector_size=2, class_count=class_count)
    else:
        class_count = 2
        perceptron = AlternativePerceptron(vector_size=2)

    has_classification_errors = True
    while has_classification_errors:
        has_classification_errors = perceptron.train(training_data)

    classified_data = []
    for vector in test_data:
        classified_data.append((vector, perceptron.guess(vector)))

    colors = [random_color() for _ in range(class_count)]
    plotter.draw_classes(training_data, 'Perceptron Classifier: Training Data', 1, colors)
    plotter.draw_classes(classified_data, 'Perceptron Classifier: Classified Data', 2, colors)
    plotter.show()


if __name__ == '__main__':
    main()
