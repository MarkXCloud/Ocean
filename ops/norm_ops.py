from computation_graph import Node
import numpy as np
import cupy as cp


class Norm(Node):
    def __init__(self, *parents, **kwargs):
        super(Norm, self).__init__(*parents, **kwargs)
        assert len(self.parents) == 1


class batchnorm2d(Norm):
    def __init__(self, *parents, **kwargs):
        super(Norm, self).__init__(*parents, **kwargs)
        self.batch_list = []
        self.batch_mean = 0
        self.batch_std = 1
        self.eps = 1e-8
        self.momentum = 0.1

    def calculate(self):
        self.batch_list.append(self.parents[0].value)
        if self.graph.cuda_device == 'cpu':
            self.value = (self.parents[0].value - self.batch_mean) / (self.batch_std + self.eps)
        else:
            self.value = cp.divide(cp.subtract(self.parents[0].value , self.batch_mean),cp.add(self.batch_std + self.eps))
    def backward(self, parent):
        if self.graph.cuda_device == 'cpu':
            if self.judge_nan(self.grad):
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)  # gather grad
            return self.grad*(self.batch_std + self.eps)
        else:
            if self.judge_nan(self.grad):
                self.grad = cp.sum(cp.asarray([child.backward(self) for child in self.children]), axis=0)  # gather grad
            return cp.multiply((self.batch_std + self.eps),self.grad)
    def refresh(self):
        if self.graph.cuda_device == 'cpu':
            new_mean = np.mean(self.batch_list,axis=0)
            new_std = np.std(self.batch_list,axis=0)
            self.batch_mean = (1-self.momentum)*self.batch_mean+self.momentum*new_mean
            self.batch_std = (1-self.momentum)*self.batch_std+self.momentum*new_std
            self.batch_list = []
        else:
            new_mean = cp.mean(cp.asarray(self.batch_list), axis=0)
            new_std = cp.std(cp.asarray(self.batch_list), axis=0)
            self.batch_mean =cp.add( cp.multiply((1 - self.momentum) ,self.batch_mean ),cp.multiply(self.momentum ,new_mean))
            self.batch_std =cp.add( cp.multiply((1 - self.momentum) ,self.batch_std ),cp.multiply(self.momentum ,new_std))
            self.batch_list = []

