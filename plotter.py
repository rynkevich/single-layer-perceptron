import matplotlib.pyplot as plt


def draw_classes(classified_data, title, figure_number):
    colors = ['orange', 'blue']

    plt.figure(figure_number)

    for vector, label in classified_data:
        plt.scatter(vector[0], vector[1], c=colors[label], s=60)

    plt.title(title)


def show():
    plt.show()
