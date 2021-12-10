from node import Node


def test_create_node():
    train_dat = [[1, 2, 4],
                 [1, 4, 5],
                 [1, 4, 3],
                 [1, 5, 3]]
    train_lab = [2, 3, 1, 1]
    classes = [1, 2, 3]
    attr = [1, 2]

    nd = Node(train_dat, train_lab, 0, 1, classes, attr, 1)
    assert nd.attribute == 0
    assert nd.attribute_value == 1
    assert nd.attributes == attr
    assert nd.best_class == 1
    assert nd.classes == classes
    assert nd.depth == 1
    assert nd.isleaf == True
    assert nd.train_data == train_dat
    assert nd.train_labels == train_lab


def test_predict():
    train_dat = [[1, 2, 4],
                 [1, 4, 5],
                 [2, 4, 3],
                 [2, 5, 3]]
    train_lab = [2, 2, 1, 1]
    classes = [1, 2]
    attr = [0, 1, 2]

    nd = Node(train_dat, train_lab, None, None, classes, attr, 0)

    attrk = [1, 2]
    kid1 = Node([[1, 2, 4], [1, 4, 5]], [2, 2], 0, 1, classes, attrk, 1)
    kid2 = Node([[2, 4, 3], [2, 5, 3]], [1, 1], 0, 2, classes, attrk, 1)
    nd.kids.append(kid1)
    nd.kids.append(kid2)
    nd.isleaf = False

    assert nd.predict([2, 5, 3]) == 1
    assert nd.predict([1, 2, 4]) == 2


def test_get_best_attribute():
    train_dat = [[1, 2, 4],
                 [1, 4, 5],
                 [1, 4, 3],
                 [1, 5, 3]]
    train_lab = [2, 3, 1, 1]
    classes = [1, 2, 3]
    attr = [1, 2]

    nd = Node(train_dat, train_lab, 0, 1, classes, attr, 1)
    assert nd.get_best_attribute() == 2


def test_induce():
    train_dat = [[1, 2, 4],
                 [1, 4, 5],
                 [1, 4, 3],
                 [1, 5, 3]]
    train_lab = [2, 3, 1, 1]
    classes = [1, 2, 3]
    attr = [1, 2]

    nd = Node(train_dat, train_lab, 0, 1, classes, attr, 1)
    nd.induce()
    assert len(nd.kids) == 3
