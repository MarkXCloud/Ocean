import numpy as np
import nn
from ops import Variable
from optimizer import SGD
from data import DataLoader, generate_one_hot
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class MLP(nn.NodeAdder):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim=784, output_dim=200),
            nn.Sigmoid(),
            nn.Linear(input_dim=200, output_dim=10),
            nn.Softmax()
        )

    def forward(self, X):
        return self.fc(X)

def load_data(path: str = 'data/mnist_dataset/mnist_train.csv'):
    """Load mnist"""
    data_file = open(path, 'r')
    data_list = data_file.readlines()
    data_file.close()
    data_list = np.array([data.strip().split(',') for data in data_list], dtype=float)  # nums x (1+dim), 0~255
    data = data_list[:, 1:] / 255.0
    label = data_list[:, 0]
    data = data[..., np.newaxis]
    label = label[..., np.newaxis]
    label = generate_one_hot(label, num_class=10)
    label = label[..., np.newaxis]
    return data, label


if __name__ == '__main__':
    E = 10
    batch_size = 64

    train_data, train_label = load_data('../data/mnist_dataset/mnist_train.csv')
    train_loader = DataLoader(data=train_data, label=train_label, batch_size=batch_size)
    test_data, test_label = load_data('../data/mnist_dataset/mnist_test.csv')
    test_loader = DataLoader(data=test_data, label=test_label, batch_size=1)
    print(f"Successfully load data!")

    x = Variable()
    m = MLP()
    pred = m(x)
    y = Variable()
    loss = nn.MSE()
    error = loss(pred=pred, target=y)
    optim = SGD(graph=m.model_graph, loss=error, lr=0.1)

    train_log = []
    error_log = []
    for i in range(E):
        m.set_train_mode()
        for batch_data, batch_label in tqdm(train_loader, desc=f'epoch {i}'):
            optim.zero_gradient()
            for data, label in zip(batch_data, batch_label):
                x.set_value(data)
                y.set_value(label)
                optim.calculate_grad()
                train_log.append(error.value)

            optim.step()
        error_log.append(np.mean(train_log))
        train_log = []

        pred_list = []
        m.set_eval_mode()
        for batch_data, batch_label in tqdm(test_loader):
            for data, label in zip(batch_data, batch_label):
                x.set_value(data)
                y.set_value(label)
                error.forward()
                pred_list.append(pred.value)
        print(accuracy_score(np.argmax(pred_list,axis=1).flatten(),np.argmax(test_label,axis=1).flatten()))

    print(error_log)
