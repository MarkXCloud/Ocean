from ops import Node, MatMul, Add, Variable, sigmoid, Convolve, MaxPool, AveragePool, ReshapeValue, softmax,batchnorm2d,Multiply,relu
from computation_graph import global_graph
import numpy as np


class NodeAdder:
    def __init__(self):
        self.model_graph = global_graph

    def forward(self, X: Node):
        raise NotImplementedError

    def __call__(self, X, *args, **kwargs):
        return self.forward(X)


class Linear(NodeAdder):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(Linear, self).__init__()
        self.W = Variable(is_Train=True)
        self.W.name = 'W_' + self.W.name
        self.W.set_value(np.random.normal(scale=0.1, size=(output_dim, input_dim)))
        self.use_bias = use_bias
        if use_bias:
            self.B = Variable(is_Train=True)
            self.B.set_value(value=np.random.normal(size=(output_dim, 1)))

    def forward(self, X):
        return Add(MatMul(self.W, X), self.B) if self.use_bias else MatMul(self.W, X)


class Sigmoid(NodeAdder):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, X):
        return sigmoid(X)

class Relu(NodeAdder):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, X):
        return relu(X)


class Softmax(NodeAdder):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, X):
        return softmax(X)


class Conv2d(NodeAdder):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel = Variable(is_Train=True)
        self.kernel.set_value(np.random.normal(scale=0.1, size=(out_channels, in_channels, kernel_size, kernel_size)))

    def forward(self, X):
        return Convolve(self.kernel, X, kernel_size=self.kernel_size, stride=self.stride,
                        padding=self.padding)


class MaxPooling(NodeAdder):
    def __init__(self, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, X):
        return MaxPool(X, kernel_size=self.kernel_size, stride=self.stride,
                       padding=self.padding)


class AveragePooling(NodeAdder):
    def __init__(self, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, X):
        return AveragePool(X, kernel_size=self.kernel_size, stride=self.stride,
                           padding=self.padding)


class GlobalAveragePolling(NodeAdder):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, X):
        return AveragePool(X, kernel_size=self.kernel_size, stride=1,
                           padding=0)


class Reshape(NodeAdder):

    def __init__(self, input_shape, target_shape):
        super().__init__()
        self.origin_shape = input_shape
        self.target_shape = target_shape

    def forward(self, X):
        return ReshapeValue(X, img_shape=self.origin_shape, target_shape=self.target_shape)


class Sequential(NodeAdder):
    def __init__(self, *Adders):
        super().__init__()
        self.adders = list(Adders)
        assert len(self.adders) != 0, 'Empty Sequential!'

    def forward(self, X):
        temp_var_list = []
        temp_var_list.append(self.adders[0](X))
        for adders in self.adders[1:]:
            temp_var_list.append(adders(temp_var_list[-1]))
        return temp_var_list[-1]

class BatchNorm2d(NodeAdder):
    def __init__(self):
        super().__init__()
        self.gamma = Variable(is_Train=True)
        self.gamma.set_value(np.ones(1,))
        self.beta = Variable(is_Train=True)
        self.beta.set_value(np.zeros(1,))
    def forward(self, X: Node):
        return Add(Multiply(self.gamma,batchnorm2d(X)),self.beta)