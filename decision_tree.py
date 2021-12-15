from node import Node


class DecisionTree:
    def __init__(self, max_depth) -> None:
        if max_depth > 0:
            self.max_depth = max_depth
        else:
            raise Exception('Negative tree depth exeption')
        self.current_depth = 0
        self.root = None
        self.classes = []
        self.attributes = []
        self.train_data = []
        self.train_labels = []
        self.validation_data = []
        self.validation_labels = []
        self.leaves = []


    def train(self):
        next_leaves = []
        for leaf in self.leaves:
            new_leaves = leaf.induce()
            if new_leaves is None:
                continue
            next_leaves.extend(new_leaves)
        self.leaves.clear()
        self.leaves.extend(next_leaves)


    def fit(self, attributes, classes, train_data, train_labels, validation_data = None, validation_labels = None, max_depth=None) :
        if max_depth is not None:
            self.max_depth = max_depth

        self.attributes.clear()
        self.attributes.extend(attributes)

        self.classes.clear()
        self.classes.extend(classes)

        self.train_data.clear()
        self.train_data.extend(train_data)

        self.train_labels.clear()
        self.train_labels.extend(train_labels)

        if validation_data is not None:
            self.validation_data.clear()
            self.validation_data.extend(validation_data)

        if validation_labels is not None:
            self.validation_labels.clear()
            self.validation_labels.extend(validation_labels)

        self.root = Node(self.train_data, self.train_labels, None, None, self.classes, self.attributes, 0)
        self.leaves.clear()
        self.leaves.append(self.root)

        train_acc = self.evaluate(self.train_data, self.train_labels)
        print("Max depth:  0", ";    Train accuracy: ", train_acc, end='')
        if self.validation_data is not None:
            val_acc = self.evaluate(self.validation_data, self.validation_labels)
            print(";    Validation accuracy: ", val_acc)
        else:
            print("\n")
        for i in range(self.max_depth):
            self.train()
            train_acc = self.evaluate(self.train_data, self.train_labels)
            print("Max depth: ", i+1, ";    Train accuracy: ", train_acc ,end='')
            if self.validation_data is not None:
                val_acc = self.evaluate(self.validation_data, self.validation_labels)
                print(";    Validation accuracy: ", val_acc)
            else:
                print("\n")


    def evaluate(self, data, labels):
        correct = 0
        predictions = self.predict(data)
        for i in range(len(labels)):
            if predictions[i] == labels[i]:
                correct += 1
        return correct / len(data)


    def predict(self, data):
        return [self.root.predict(dat) for dat in data]


