import numpy as np
import nn
from ops import Variable
from optimizer import SGD
from data import DataLoader, generate_one_hot
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class FE(nn.NodeAdder):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Sigmoid(),
            nn.MaxPooling(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.MaxPooling(kernel_size=2, stride=2)
        )

    def forward(self, X):
        return self.conv(X)


class MLP(nn.NodeAdder):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim=16 * 4 * 4, output_dim=120),
            nn.Sigmoid(),
            nn.Linear(input_dim=120, output_dim=84),
            nn.Sigmoid(),
            nn.Linear(input_dim=84, output_dim=10),
            nn.Softmax()
        )

    def forward(self, X):
        return self.fc(X)


class LeNet(nn.NodeAdder):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            FE(),
            nn.Reshape(input_shape=(16, 4, 4), target_shape=(-1, 1)),
            MLP()
        )

    def forward(self, X):
        return self.net(X)


def load_data(path: str = 'data/mnist_dataset/mnist_train.csv'):
    """Load mnist"""
    data_file = open(path, 'r')
    data_list = data_file.readlines()
    data_file.close()
    data_list = np.array([data.strip().split(',') for data in data_list], dtype=float)  # nums x (1+dim), 0~255
    data = data_list[:, 1:] / 255.0
    label = data_list[:, 0]
    label = label[..., np.newaxis]
    label = generate_one_hot(label, num_class=10)

    data = data.reshape(-1, 1, 28, 28)  # nums x C x H x W
    label = label[..., np.newaxis]
    return data, label


if __name__ == '__main__':
    E = 10
    batch_size = 10

    train_data, train_label = load_data('../data/mnist_dataset/mnist_train.csv')
    train_loader = DataLoader(data=train_data, label=train_label, batch_size=batch_size)
    teat_data, test_label = load_data('../data/mnist_dataset/mnist_test.csv')
    test_loader = DataLoader(data=teat_data, label=test_label, batch_size=1)

    x = Variable()
    m = LeNet()
    pred = m(x)
    y = Variable()
    # loss = MSE()
    loss = nn.CELoss()
    error = loss(pred=pred, target=y)
    optim = SGD(graph=m.model_graph, loss=error, lr=0.1)

    train_log = []
    error_log = []
    for i in range(E):
        for batch_data, batch_label in tqdm(train_loader, desc=f'epoch {i}'):
            optim.zero_gradient()
            for data, label in zip(batch_data, batch_label):
                x.set_value(data)
                y.set_value(label)
                optim.calculate_grad()
                train_log.append(error.value)

            optim.step()
        error_log.append(np.mean(train_log))
        # print(f' training error {error_log[-1]}')
        train_log = []

        pred_list = []
        for batch_data, batch_label in tqdm(test_loader):
            for data, label in zip(batch_data, batch_label):
                x.set_value(data)
                y.set_value(label)
                error.forward()
                pred_list.append(pred.value)
        print(accuracy_score(np.argmax(pred_list,axis=0),np.argmax(test_label,axis=0)))

    print(error_log)
