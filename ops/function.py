from computation_graph import Node
import numpy as np
import cupy as cp
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
        if self.judge_nan(self.grad):
            self.calculate_grad()
            if self.graph.cuda_device == 'cpu':
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)  # gather grad
            else:
                self.grad = cp.sum(cp.asarray([child.backward(self) for child in self.children]), axis=0)  # gather grad
            self.grad = self.current_grad * self.grad

        return self.grad


class sigmoid(Activation):

    def calculate(self):
        if self.graph.cuda_device == 'cpu':
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
        if self.graph.cuda_device == 'cpu':
            self.value = tanh.efficient_tanh(self.parents[0].value)
        else:
            self.value = tanh.efficient_tanh_gpu(self.parents[0].value)

    @staticmethod
    def efficient_tanh(x):
        y = x.copy()
        y = np.tanh(y)
        return y

    @staticmethod
    def efficient_tanh_gpu(x):
        y = x.copy()
        y = cp.tanh(y)
        return y

    def calculate_grad(self):
        self.current_grad = 1 - self.value ** 2


class relu(Activation):

    def calculate(self):
        if self.graph.cuda_device == 'cpu':
            self.value = relu.efficient_relu(self.parents[0].value)
        else:
            self.value = relu.efficient_relu_gpu(self.parents[0].value)

    @staticmethod
    def efficient_relu(x):
        y = x.copy()
        return np.clip(y, a_min=0, a_max=None)

    @staticmethod
    def efficient_relu_gpu(x):
        y = x.copy()
        return cp.clip(y, a_min=0, a_max=None)

    def calculate_grad(self):
        if self.graph.cuda_device == 'cpu':
            self.current_grad = np.where(self.value > 0, 1, 0)
        else:
            self.current_grad = cp.where(self.value > 0, 1, 0)


class softmax(Activation):
    def calculate(self):
        if self.graph.cuda_device == 'cpu':
            self.value = softmax.efficient_softmax(self.parents[0].value)
        else:
            self.value = softmax.efficient_softmax_gpu(self.parents[0].value)

    @staticmethod
    def efficient_softmax(x):
        y = x.copy()
        y = np.exp(y)
        y_sum = np.sum(y)
        return y / y_sum

    @staticmethod
    def efficient_softmax_gpu(x):
        y = x.copy()
        y = cp.exp(y)
        y_sum = cp.sum(y)
        return cp.divide(y, y_sum)

    def calculate_grad(self):
        grad = -self.value * self.value.T
        if self.graph.cuda_device == 'cpu':
            grad += np.diag(self.value.flatten())
        else:
            grad = cp.add(grad, cp.diag(self.value.flatten()))
        self.current_grad = grad

    def backward(self, current_node):
        if self.judge_nan(self.grad):
            self.calculate_grad()
            if self.graph.cuda_device == 'cpu':
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)  # gather grad
            else:
                self.grad = cp.sum(cp.asarray([child.backward(self) for child in self.children]), axis=0)  # gather grad
            self.grad = self.current_grad @ self.grad

        return self.grad
