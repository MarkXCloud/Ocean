import numpy as np
import cupy as cp
from computation_graph import Graph
from ops import Loss, Variable, batchnorm2d


class Optimizer:
    def __init__(self, graph: Graph, loss: Loss, lr: float, weight_decay: float = 0.0):
        self.graph = graph
        self.loss_node = loss
        self.lr = lr
        self.wd = weight_decay
        self.accumulated_grad = {}

    def zero_gradient(self):
        self.accumulated_grad = {}

    def calculate_grad(self):
        """
        count gradient in one step, which means perform forward and backward once, with only one data sample.
        """
        self.loss_node.graph.zero_gradient()
        self.loss_node.forward()
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.is_Train:
                node.backward(node)
                if node not in self.accumulated_grad:
                    self.accumulated_grad[node] = node.grad
                else:
                    self.accumulated_grad[node] += node.grad

    def step(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.is_Train:
                node.set_value(self.refresh(node, self.accumulated_grad[node]))
            elif isinstance(node, batchnorm2d):
                node.refresh()

    def refresh(self, W, accumulated_grad):
        pass


class SGD(Optimizer):

    def refresh(self, W, accumulated_grad):
        return W.value - self.lr * accumulated_grad - self.lr * self.wd * W.value


class Adam(Optimizer):
    def __init__(self, betas: tuple = (0.9, 0.999), **kwargs):
        super(Adam, self).__init__(**kwargs)
        self.m = {}
        self.s = {}
        self.eps = 1e-8
        self.beta1, self.beta2 = betas
        self.t = 0

    def refresh(self, W, accumulated_grad):
        if W not in self.m:
            self.m[W] = (1 - self.beta1) * accumulated_grad
            self.s[W] = (1 - self.beta2) * accumulated_grad ** 2
        else:
            self.m[W] = self.beta1 * self.m[W] + (1 - self.beta1) * accumulated_grad
            self.s[W] = self.beta2 * self.s[W] + (1 - self.beta2) * accumulated_grad ** 2
        self.t += 1
        m_hat = self.m[W] / (1 - self.beta1 ** self.t)
        s_hat = self.s[W] / (1 - self.beta2 ** self.t)
        return W.value - self.lr * m_hat / (np.sqrt(s_hat) + self.eps) - self.lr * self.wd * W.value
