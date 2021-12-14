from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def get_formated_data(bins):
    dataset = load_iris()
    data = dataset.data

    for i in range(len(data[0])):
        column = [data[j][i] for j in range(len(data))]
        max_val = max(column)
        min_val = min(column)
        for j in range(len(data)):
            if data[j][i] < min_val + ((max_val - min_val)/bins):
                data[j][i] = 0
            elif data[j][i] >= min_val + ((bins-1) * (max_val - min_val)/bins):
                data[j][i] = bins-1
            else:
                for k in range(1, bins-1):
                    if data[j][i] < min_val + ((k+1) * (max_val - min_val)/bins) and data[j][i] >= min_val + (k * (max_val - min_val)/bins):
                        data[j][i] = k
                        break

    labels = dataset.target
    classes = list(set(labels))
    attributes = range(len(data[0]))
    full_data = tuple(zip(data, labels))
    shuffled = shuffle(full_data)
    res = list(zip(*shuffled))
    data = res[0]
    labels = res[1]
    model_data, test_data, model_labels, test_labels = train_test_split(data, labels, test_size=0.15)
    train_data, validation_data, train_labels, validation_labels = train_test_split(model_data, model_labels, test_size=(3/17))
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels, classes, attributes
