from ops import Node, Variable
import ops
from computation_graph import global_graph
import numpy as np


class NodeAdder:
    def __init__(self):
        self.model_graph = global_graph

    def forward(self, X: Node):
        raise NotImplementedError

    def __call__(self, X, *args, **kwargs):
        return self.forward(X)

    def set_train_mode(self):
        for node in self.model_graph.nodes:
            if isinstance(node, ops.Norm):
                node.set_train_mode()

    def set_eval_mode(self):
        for node in self.model_graph.nodes:
            if isinstance(node, ops.Norm):
                node.set_eval_mode()


class Linear(NodeAdder):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(Linear, self).__init__()
        self.W = Variable(is_Train=True)
        self.W.name = 'W_' + self.W.name
        self.W.set_value(np.random.normal(scale=0.1, size=(output_dim, input_dim)))
        self.use_bias = use_bias
        if use_bias:
            self.B = Variable(is_Train=True)
            self.B.set_value(value=np.random.normal(size=(output_dim, 1),scale=0.1))

    def forward(self, X):
        return ops.Add(ops.MatMul(self.W, X), self.B) if self.use_bias else ops.MatMul(self.W, X)


class Sigmoid(NodeAdder):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, X):
        return ops.sigmoid(X)


class Relu(NodeAdder):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, X):
        return ops.relu(X)


class Softmax(NodeAdder):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, X):
        return ops.softmax(X)


class Conv2d(NodeAdder):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel = Variable(is_Train=True)
        self.kernel.set_value(np.random.normal(scale=0.1, size=(out_channels, in_channels, kernel_size, kernel_size)))

    def forward(self, X):
        return ops.Convolve(self.kernel, X, kernel_size=self.kernel_size, stride=self.stride,
                            padding=self.padding)


class MaxPooling(NodeAdder):
    def __init__(self, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, X):
        return ops.MaxPool(X, kernel_size=self.kernel_size, stride=self.stride,
                           padding=self.padding)


class AveragePooling(NodeAdder):
    def __init__(self, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, X):
        return ops.AveragePool(X, kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding)


class GlobalAveragePolling(NodeAdder):

    def forward(self, X):
        return ops.GlobalAveragePool(X)


class Reshape(NodeAdder):

    def __init__(self, input_shape, target_shape):
        super().__init__()
        self.origin_shape = input_shape
        self.target_shape = target_shape

    def forward(self, X):
        return ops.ReshapeValue(X, img_shape=self.origin_shape, target_shape=self.target_shape)


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
        self.gamma.set_value(np.ones(1, ))
        self.beta = Variable(is_Train=True)
        self.beta.set_value(np.zeros(1, ))

    def forward(self, X: Node):
        return ops.Add(ops.Multiply(self.gamma, ops.batchnorm2d(X)), self.beta)


class Dropout(NodeAdder):
    def __init__(self, drop_rate: float):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, X: Node):
        return ops.dropout(self.drop_rate, X)
