import numpy as np
import cupy as cp
from computation_graph import Node


class Operator(Node):
    def __init__(self, *parents, **kwargs):
        super(Operator, self).__init__(*parents, **kwargs)
        assert len(self.parents) == 2


class Add(Operator):

    def calculate(self):
        self.value = self.parents[0].value + self.parents[1].value

    def backward(self, current_node):
        if self.judge_nan(self.grad):
            if self.graph.cuda_device == 'cpu':
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)  # gather grad
            else:
                self.grad = cp.sum(cp.array([child.backward(self) for child in self.children]), axis=0)  # gather grad
        return self.grad


class Multiply(Operator):

    def calculate(self):
        self.value = self.parents[0].value * self.parents[1].value

    def backward(self, parent):
        if self.judge_nan(self.grad):
            if self.graph.cuda_device == 'cpu':
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)
            else:
                self.grad = cp.sum(cp.asarray([child.backward(self) for child in self.children]), axis=0)
        if parent.name == self.parents[0].name:
            return self.parents[1].value * self.grad
        else:
            return self.parents[0].value * self.grad


class MatMul(Operator):
    def calculate(self):
        assert self.parents[0].value.shape[1] == self.parents[1].value.shape[
            0], f'shape mismatch {self.parents[0].value.shape}, {self.parents[1].value.shape}'

        self.value = self.parents[0].value @ self.parents[1].value

    def backward(self, parent):
        if self.judge_nan(self.grad):
            if self.graph.cuda_device == 'cpu':
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)
            else:
                self.grad = cp.sum(cp.array([child.backward(self) for child in self.children]), axis=0)  # gather grad
        if parent is self.parents[0]:
            return self.grad @ self.parents[1].value.T
        else:
            return self.parents[0].value.T @ self.grad


class Subtract(Operator):

    def calculate(self):
        self.value = self.parents[0].value - self.parents[1].value

    def backward(self, parent):
        if self.judge_nan(self.grad):
            if self.graph.cuda_device == 'cpu':
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)
            else:
                self.grad = cp.sum(cp.array([child.backward(self) for child in self.children]), axis=0)  # gather grad

        if parent.name == self.parents[0].name:
            return self.grad
        else:
            return -self.grad


class Divide(Operator):

    def calculate(self):
        self.value = self.parents[0].value / self.parents[1].value

    def backward(self, parent):
        if self.judge_nan(self.grad):
            if self.graph.cuda_device == 'cpu':
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)
            else:
                self.grad = cp.sum(cp.array([child.backward(self) for child in self.children]), axis=0)  # gather grad

        if parent.name == self.parents[0].name:
            return (1 / self.parents[1].value) * self.grad
        else:
            return (-self.parents[0].value / self.parents[1].value ** 2) * self.grad
