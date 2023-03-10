from computation_graph import Node
import numpy as np
from cupyx.scipy.special import expit as sigmoid_gpu


class Activation(Node):
    def __init__(self, *parents, **kwargs):
        super(Activation, self).__init__(*parents, **kwargs)
        assert len(self.parents) == 1

    def backward(self, current_node):
        """
        Recursively gather the gradient of children nodes, and multiplied with current gradient according to chain rule
        :return:grad, pass to the previous node
        """
        self.gather_grad()
        return self.grad

    def gather_grad(self):
        if self.judge_nan(self.grad):
            self.calculate_grad()
            self.grad = self.P.sum(self.P.asarray([child.backward(self) for child in self.children]),
                                   axis=0)  # gather grad
            self.grad = self.current_grad * self.grad
        else:
            pass


class sigmoid(Activation):

    def calculate(self):
        if not self.use_cuda:
            self.value = sigmoid.efficient_sigmoid(self.parents[0].value)
        else:
            self.value = sigmoid.efficient_sigmoid_gpu(self.parents[0].value)

    @staticmethod
    def efficient_sigmoid(x):
        """partly calculate sigmoid of x>=0 and x<0 to avoid overflow of exp"""
        y = x.copy()
        y[x >= 0] = 1 / (1 + np.exp(-x[x >= 0]))
        y[x < 0] = np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))
        return y

    @staticmethod
    def efficient_sigmoid_gpu(x):
        """partly calculate sigmoid of x>=0 and x<0 to avoid overflow of exp"""
        return sigmoid_gpu(x)

    def calculate_grad(self):
        self.current_grad = self.value * (1 - self.value)


class tanh(Activation):

    def calculate(self):
        self.value = self.P.tanh(self.parents[0].value)

    def calculate_grad(self):
        self.current_grad = 1 - self.value ** 2


class relu(Activation):

    def calculate(self):
        self.value = self.P.clip(self.parents[0].value, a_min=0, a_max=None)

    def calculate_grad(self):
        self.current_grad = self.P.where(self.value > 0, 1, 0)


class softmax(Activation):
    def calculate(self):
        y = self.P.exp(self.parents[0].value)
        y_sum = self.P.sum(y)
        self.value = y / y_sum

    def calculate_grad(self):
        grad = -self.value * self.value.T
        grad += self.P.diag(self.value.flatten())
        self.current_grad = grad

    def gather_grad(self):
        if self.judge_nan(self.grad):
            self.calculate_grad()
            # gather grad
            self.grad = self.P.sum(self.P.asarray([child.backward(self) for child in self.children]), axis=0)
            self.grad = self.current_grad @ self.grad
        else:
            pass
