from computation_graph import Node
import numpy as np
import cupy as cp


class Loss(Node):
    pass


class MSELoss(Loss):
    def calculate(self):
        self.value = ((self.parents[0].value - self.parents[1].value) ** 2).mean() / 2

    def backward(self, parent):
        result = (self.parents[0].value - self.parents[1].value) / self.parents[0].value.flatten().shape[0]
        return result if parent.name == self.parents[0].name else -result


class CrossEntropyLoss(Loss):
    def calculate(self):
        self.value = -(self.parents[1].value * np.log(self.parents[0].value + 1e-16)).sum()

    def backward(self, parent):
        if parent.name == self.parents[0].name:
            return -self.parents[1].value / (self.parents[0].value + 1e-16)
        else:
            return -self.P.log(self.parents[0].value + 1e-16)