from data_loader import get_formated_data
from decision_tree import DecisionTree

DEPTH = 4

def main():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels, classes, attributes = get_formated_data()
    tree = DecisionTree(DEPTH)
    tree.fit(attributes, classes, train_data, train_labels, validation_data, validation_labels)


if __name__ == "__main__":
    main()
