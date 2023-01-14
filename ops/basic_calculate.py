import numpy as np
import cupy as cp
from computation_graph import Node


class Operator(Node):
    def __init__(self, *parents, **kwargs):
        super(Operator, self).__init__(*parents, **kwargs)
        assert len(self.parents) == 2


class Add(Operator):

    def calculate(self):
        if self.graph.cuda_device == 'cpu':
            self.value = self.parents[0].value + self.parents[1].value
        else:
            self.value = cp.add(self.parents[0].value, self.parents[1].value)

    def backward(self, current_node):
        if self.judge_nan(self.grad):
            if self.graph.cuda_device == 'cpu':
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)  # gather grad
            else:
                self.grad = cp.sum(cp.array([child.backward(self) for child in self.children]), axis=0)  # gather grad
        return self.grad


class Multiply(Operator):

    def calculate(self):
        if self.graph.cuda_device == 'cpu':
            self.value = self.parents[0].value * self.parents[1].value
        else:
            self.value = cp.multiply(self.parents[0].value, self.parents[1].value)

    def backward(self, parent):
        if self.graph.cuda_device == 'cpu':
            if self.judge_nan(self.grad):
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)
            if parent.name == self.parents[0].name:
                return self.parents[1].value * self.grad
            else:
                return self.parents[0].value * self.grad
        else:
            if self.judge_nan(self.grad):
                self.grad = cp.sum(cp.asarray([child.backward(self) for child in self.children]), axis=0)
            if parent.name == self.parents[0].name:
                return cp.multiply(self.parents[1].value, self.grad)
            else:
                return cp.multiply(self.parents[0].value, self.grad)


class MatMul(Operator):
    def calculate(self):
        assert self.parents[0].value.shape[1] == self.parents[1].value.shape[0],f'shape mismatch {self.parents[0].value.shape}, {self.parents[1].value.shape}'
        if self.graph.cuda_device == 'cpu':
            self.value = self.parents[0].value @ self.parents[1].value
        else:
            self.value = cp.matmul(self.parents[0].value, self.parents[1].value)

    def backward(self, parent):
        if self.graph.cuda_device == 'cpu':
            if self.judge_nan(self.grad):
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)
            if parent is self.parents[0]:
                return self.grad @ self.parents[1].value.T
            else:
                return self.parents[0].value.T @ self.grad
        else:
            if self.judge_nan(self.grad):
                self.grad = cp.sum(cp.array([child.backward(self) for child in self.children]), axis=0)  # gather grad
            if parent is self.parents[0]:
                return cp.matmul(self.grad, self.parents[1].value.T)
            else:
                return cp.matmul(self.parents[0].value.T, self.grad)


class Subtract(Operator):

    def calculate(self):
        if self.graph.cuda_device == 'cpu':
            self.value = self.parents[0].value - self.parents[1].value
        else:
            self.value = cp.subtract(self.parents[0].value, self.parents[1].value)

    def backward(self, parent):
        if self.graph.cuda_device == 'cpu':
            if self.judge_nan(self.grad):
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)
            if parent.name == self.parents[0].name:
                return self.grad
            else:
                return -self.grad
        else:
            if self.judge_nan(self.grad):
                self.grad = cp.sum(cp.array([child.backward(self) for child in self.children]), axis=0)  # gather grad
            if parent.name == self.parents[0].name:
                return self.grad
            else:
                return cp.multiply(-1, self.grad)


class Divide(Operator):

    def calculate(self):
        if self.graph.cuda_device == 'cpu':
            self.value = self.parents[0].value / self.parents[1].value
        else:
            self.value = cp.divide(self.parents[0].value, self.parents[1].value)

    def backward(self, parent):
        if self.graph.cuda_device == 'cpu':
            if self.judge_nan(self.grad):
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)
            if parent.name == self.parents[0].name:
                return (1 / self.parents[1].value) * self.grad
            else:
                return (-self.parents[0].value / self.parents[1].value ** 2) * self.grad
        else:
            if self.judge_nan(self.grad):
                self.grad = cp.sum(cp.array([child.backward(self) for child in self.children]), axis=0)  # gather grad
            if parent.name == self.parents[0].name:
                return cp.multiply(cp.divide(1, self.parents[1].value), self.grad)
            else:
                return cp.multiply(cp.divide(-self.parents[0].value, cp.power(self.parents[1].value, 2)), self.grad)
