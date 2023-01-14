import numpy as np
import cupy as cp
from computation_graph import Graph
from ops import Loss,Variable

class Optimizer:
    def __init__(self,graph:Graph,loss:Loss,lr:float):
        self.graph = graph
        self.loss_node = loss
        self.lr = lr
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
            if isinstance(node,Variable) and node.is_Train:
                node.backward(node)
                if node not in self.accumulated_grad:
                    self.accumulated_grad[node] = node.grad
                else:
                    if self.graph.cuda_device=='cpu':
                        self.accumulated_grad[node]+=node.grad
                    else:
                        self.accumulated_grad[node] = cp.add(self.accumulated_grad[node], node.grad)
        return self.loss_node.value

    def step(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.is_Train:
                node.set_value(self.refresh(node.value, self.accumulated_grad[node]))

    def refresh(self,W,accumulated_grad)->np.ndarray:
        pass

class SGD(Optimizer):

    def refresh(self,W,accumulated_grad)->np.ndarray:
        if self.graph.cuda_device=='cpu':
            return W-self.lr*accumulated_grad
        else:
            return cp.subtract(W,cp.multiply(self.lr,accumulated_grad))
