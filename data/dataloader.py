import numpy as np


def generate_label(data: np.ndarray, label: int):
    """
    Assign label to certain class.
    :param data: The data.
    :param label: the index of class, e.g., 1,2,3
    :return: [data,label]
    """
    label = np.ones((len(data), 1)) * label
    return np.hstack((label, data))


def preprecess(data: np.ndarray, label: int):
    return generate_label(label=label, data=data)


def split_dataset(data: np.ndarray, ratio=0.2, num_class: int = 3):
    np.random.shuffle(data)
    split_idx = int(len(data) * (1 - ratio))
    train_data, train_label = data[:split_idx, 1:], data[:split_idx, 0]
    train_label = generate_one_hot(train_label, num_class=num_class)
    train_data = train_data.T
    train_label = train_label.T

    test_data, test_label = data[split_idx:, 1:], data[split_idx:, 0]
    test_label = generate_one_hot(test_label, num_class=num_class)
    test_data = test_data.T
    test_label = test_label.T

    return train_data, train_label, test_data, test_label


def generate_single_one_hot(one_hot_label, label):
    label = int(label) - 1
    one_hot_label[label] = 1
    return one_hot_label


def generate_one_hot(labels: np.ndarray, num_class: int):
    one_hot_labels = np.zeros(shape=(labels.shape[0], num_class))
    for i in range(labels.shape[0]):
        one_hot_labels[i] = generate_single_one_hot(one_hot_labels[i], labels[i])
    return one_hot_labels


class DataLoader:
    """
    My dataloader to generate batch data.
    """

    def __init__(self, data: np.ndarray, label: np.ndarray, batch_size: int = 1, num_class: int = 3):
        self.data = data  # nums x dim
        self.label = label  # nums x dim
        self.length = self.label.shape[0]
        self.batch_size = batch_size
        self.num_class = num_class
        self.cursor = -1

    def __len__(self):
        return self.length // self.batch_size

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __next__(self):
        if self.cursor + 1 + self.batch_size <= self.length:
            data, label = self[self.cursor + 1:self.cursor + 1 + self.batch_size]
            self.cursor += self.batch_size
            return data, label
        else:
            self.cursor = -1
            raise StopIteration

    def __iter__(self):
        return self
