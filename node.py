from math import log


class Node:
    def __init__(self, train_data, train_labels, attribute=None,
                 attribute_value=None, classes=None, attributes=None, depth = 0) -> None:

        self.attribute = attribute
        self.attribute_value = attribute_value
        self.kids = []
        self.best_class = None
        self.classes = []
        self.attributes = []
        self.depth = depth
        self.train_data = []
        self.train_labels = []
        self.isleaf = True

        if classes is not None:
            for classe in classes:
                self.classes.append(classe)

        if attributes is not None:
            for attribute in attributes:
                self.attributes.append(attribute)

        if train_data is not None:
            if self.attributes is None or self.classes is None or train_labels is None:
                raise Exception("Can't load train data to node without loading classes, lables and attributes")
            samp_numb = [0 for _ in range(len(self.classes))]
            for i in range(len(train_data)):
                self.train_data.append(train_data[i])
                cl = train_labels[i]
                self.train_labels.append(cl)
                for i in range(len(self.classes)):
                    if cl == self.classes[i]:
                        samp_numb[i] += 1
                        break
                best = max(samp_numb)
                for i in range(len(samp_numb)):
                    if samp_numb[i] == best:
                        self.best_class = self.classes[i]
                        break


    def predict(self, data):
        if self.isleaf:
            return self.best_class
        else:
            for kid in self.kids:
                if data[kid.attribute] == kid.attribute_value:
                    return kid.predict(data)


    def induce(self):
        if len(self.kids) == 0:
            unique_lable = set(self.train_labels)
            if len(unique_lable) == 1 or len(self.attributes) == 0:
                return None
            best_attribute = self.get_best_attribute()
            unique_attr = set([self.train_data[i][best_attribute] for i in range(len(self.train_data))])
            self.isleaf = False
            for attr in unique_attr:
                train_data = []
                train_labels = []
                for i in range(len(self.train_labels)):
                    if self.train_data[i][best_attribute] == attr:
                        train_data.append(self.train_data[i])
                        train_labels.append(self.train_labels[i])
                if len(train_data) == 0:
                    break
                attributes = self.attributes.copy()
                attributes.remove(best_attribute)
                kid = Node(train_data, train_labels, best_attribute, attr, self.classes, attributes, self.depth+1)
                self.kids.append(kid)
            return self.kids
        raise Exception('Trying to induce already induced node')


    def get_best_attribute(self):
        best_attr = None
        best_infGain = -1
        for attribute in self.attributes:
            curr_infGain = self.infGain(attribute)
            if curr_infGain > best_infGain:
                best_infGain = curr_infGain
                best_attr = attribute
        return best_attr


    def infGain(self, attribute):
        def entropy(lables):
            unique = list(set(lables))
            amounts = [0 for _ in range(len(unique))]
            for lab in lables:
                for i in range(len(unique)):
                    if lab == unique[i]:
                        amounts[i] += 1
                        break
            return -sum([(amounts[i] / len(lables)) * log((amounts[i] / len(lables)))])
        infgain = 0
        unique_attr_val = set([self.train_data[i][attribute] for i in range(len(self.train_data))])
        for val in unique_attr_val:
            lables = []
            for i in range(len(self.train_labels)):
                if self.train_data[i][attribute] == val:
                    lables.append(self.train_labels[i])
            if len(lables) == 0:
                break
            infgain += (len(lables) / len(self.train_labels) * entropy(lables))
        return entropy(self.train_labels) - infgain


