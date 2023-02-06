import numpy as np
import nn
from ops import Variable
from optimizer import Adam
from sklearn.metrics import accuracy_score
from data import DataLoader, generate_one_hot
from tqdm import tqdm


class Residual(nn.NodeAdder):
    def __init__(self,in_channels,out_channels,use_1x1conv=False,stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d()
        self.bn2 = nn.BatchNorm2d()

        self.relu1 = nn.Relu()
        self.relu2 = nn.Relu()

    def forward(self, X):
        Y1 = self.relu1(self.bn1(self.conv1(X)))
        Y2 = self.bn2(self.conv2(Y1))
        if self.conv3:
            X = self.conv3(X)
        return self.relu2(Y2+X)

class FlattenLayer(nn.NodeAdder):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Reshape(input_shape=(128,),target_shape=(-1,1))

    def forward(self, X):
        return self.flatten(X)

def resnet_block(in_channels,out_channels,num_residuals,first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

class ResNet(nn.NodeAdder):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(),
            nn.Relu(),
            nn.MaxPooling(kernel_size=3, stride=2, padding=1)
        )
        self.res_blocks = nn.Sequential(
            resnet_block(64, 64, 2, first_block=True),
            resnet_block(64,128,2)
        )
        self.fc = nn.Sequential(
            nn.GlobalAveragePolling(),
            FlattenLayer(),
            nn.Linear(128,10)
        )

    def forward(self, X):
        return self.fc(self.res_blocks(self.head(X)))


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
    batch_size = 64

    train_data, train_label = load_data('data/mnist_dataset/mnist_train.csv')
    train_loader = DataLoader(data=train_data, label=train_label, batch_size=batch_size)
    teat_data, test_label = load_data('data/mnist_dataset/mnist_test.csv')
    test_loader = DataLoader(data=teat_data, label=test_label, batch_size=1)

    x = Variable()
    m = ResNet()
    pred = m(x)
    y = Variable()
    loss = nn.MSE()
    error = loss(pred=pred, target=y)
    optim = Adam(graph=m.model_graph, loss=error, lr=0.001)

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
        # print(f' training error {error_log[-1]}')
        train_log = []

        pred_list = []
        m.set_eval_mode()
        for batch_data, batch_label in tqdm(test_loader):
            for data, label in zip(batch_data, batch_label):
                x.set_value(data)
                y.set_value(label)
                error.forward()
                pred_list.append(pred.value)
        print(accuracy_score(np.argmax(pred_list, axis=1).flatten(), np.argmax(test_label, axis=1).flatten()))

    print(error_log)
