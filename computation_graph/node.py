from .graph import global_graph
import numpy as np
import cupy as cp
import ops

_name = 1


def fresh_name():
    global _name
    name = f'Variable{_name}'
    _name += 1
    return name


class Node:
    """
    Node in the computation graph, play a part as basic computation functions such as linear, conv,
    activation and so on.
    """

    def __init__(self, *parents, **kwargs):
        """
        Create a new node, attach it to its parent when calling.
        """
        self.kargs = kwargs
        self.graph = kwargs.get('graph', global_graph)
        self.name = fresh_name()
        self.parents = list(parents)
        self.children = []
        self.value = np.nan
        self.grad = np.nan  # grad from loss to current node
        self.current_grad = np.nan  # grad from the output of current node to current input
        self.is_Train = False
        self.use_cuda = False
        self.P = np

        for parent in self.parents:
            parent.children.append(self)
        self.graph.insert_node(self)

    def calculate(self):
        """
        Basic function to calculate values.
        """
        pass

    def calculate_grad(self):
        """
        Basic function to calculate gradients of current node.
        """
        pass

    def forward(self):
        for node in self.parents:
            if self.judge_nan(node.value):
                node.forward()
        self.calculate()

    def backward(self, parent):
        """
        Recursively gather the gradient of children nodes, and multiplied with current gradient according to chain rule
        :return:grad, pass to the previous node
        e.g.:
        if self.judge_nan(self.grad):
            self.calculate_grad()
            if self.graph.cuda_device == 'cpu':
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)  # gather grad
                self.grad = self.current_grad @ self.grad
            else:
                self.grad = cp.sum(cp.asarray([child.backward(self) for child in self.children]), axis=0)  # gather grad
                self.grad = cp.matmul(self.current_grad, self.grad)

        return self.grad
        """

        raise NotImplementedError

    def zero_gradient(self):
        self.grad = self.P.nan

    def zero_value(self, recursively=True):
        """reset the value, and recursively reset the value of its children"""
        self.value = self.P.nan
        if recursively:
            for child in self.children:
                child.zero_value()

    def judge_nan(self, value):
        return self.P.isnan(value).all()

    def set_gpu(self, index):
        self.value = cp.asarray(self.value)
        self.grad = cp.asarray(self.grad)
        self.current_grad = cp.asarray(self.current_grad)
        self.P = cp
        self.use_cuda = True

    def __add__(self, other):
        return ops.Add(self, other)

    def __sub__(self, other):
        return ops.Subtract(self, other)

    def __mul__(self, other):
        return ops.Multiply(self, other)

    def __truediv__(self, other):
        return ops.Divide(self, other)

    def __matmul__(self, other):
        return ops.MatMul(self,other)
