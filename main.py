from data_loader import get_formated_data
from decision_tree import DecisionTree


def main():
    depth = int(input('Enter max tree depth: '))
    bins = int(input('Enter discretizer bins number: '))
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels, classes, attributes = get_formated_data(bins)
    tree = DecisionTree(depth)
    tree.fit(attributes, classes, train_data, train_labels, validation_data, validation_labels)

    test_acc = tree.evaluate(test_data, test_labels)
    print('\nTest accuracy: ', test_acc)


if __name__ == "__main__":
    main()
