from computation_graph import Node
import numpy as np
import cupy as cp


class Loss(Node):
    pass


class MSELoss(Loss):
    def calculate(self):
        if self.graph.cuda_device == 'cpu':
            self.value = np.mean((self.parents[0].value - self.parents[1].value) ** 2) / 2
        else:
            self.value = cp.mean((self.parents[0].value - self.parents[1].value) ** 2) / 2

    def backward(self, parent):
        if self.graph.cuda_device == 'cpu':
            if parent.name == self.parents[0].name:
                return (self.parents[0].value - self.parents[1].value) /self.parents[0].value.flatten().shape[0]
            elif parent.name == self.parents[1].name:
                return (self.parents[1].value - self.parents[0].value) / self.parents[0].value.flatten().shape[0]
        else:
            if parent.name == self.parents[0].name:
                return cp.divide(cp.subtract(self.parents[0].value, self.parents[1].value),self.parents[0].value.flatten().shape[0])
            elif parent.name == self.parents[1].name:
                return cp.divide(cp.subtract(self.parents[1].value, self.parents[0].value), self.parents[0].value.flatten().shape[0])


class CrossEntropyLoss(Loss):
    def calculate(self):
        self.value = -np.sum(self.parents[1].value * np.log(self.parents[0].value + 1e-16))

    def backward(self,parent):
        if self.graph.cuda_device == 'cpu':
            if parent.name== self.parents[0].name:
                return -self.parents[1].value / (self.parents[0].value + 1e-16)
            else:
                return -np.log(self.parents[0].value + 1e-16)
        else:
            if parent.name== self.parents[0].name:
                return cp.divide( cp.multiply(-1,self.parents[1].value) / cp.add(self.parents[0].value , 1e-16))
            else:
                return cp.multiply(-1,cp.log(cp.add(self.parents[0].value , 1e-16)))
