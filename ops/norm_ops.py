from computation_graph import Node
import numpy as np
import cupy as cp


class Norm(Node):
    def __init__(self, *parents, **kwargs):
        super(Norm, self).__init__(*parents, **kwargs)
        assert len(self.parents) == 1
        self.train_mode = True
    def set_train_mode(self):
        self.train_mode=True
    def set_eval_mode(self):
        self.train_mode=False

class batchnorm2d(Norm):
    def __init__(self, *parents, **kwargs):
        super(Norm, self).__init__(*parents, **kwargs)
        self.batch_list = []
        self.batch_mean = 0
        self.batch_std = 1
        self.eps = 1e-8
        self.momentum = 0.1



    def calculate(self):
        if self.train_mode:
            self.batch_list.append(self.parents[0].value)
        self.value = (self.parents[0].value - self.batch_mean) / (self.batch_std + self.eps)

    def backward(self, parent):
        if self.graph.cuda_device == 'cpu':
            if self.judge_nan(self.grad):
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)  # gather grad

        else:
            if self.judge_nan(self.grad):
                self.grad = cp.sum(cp.asarray([child.backward(self) for child in self.children]), axis=0)  # gather grad
        return self.grad * (self.batch_std + self.eps)

    def refresh(self):
        if self.graph.cuda_device == 'cpu':
            new_mean = np.mean(self.batch_list, axis=0)
            new_std = np.std(self.batch_list, axis=0)
        else:
            new_mean = cp.mean(cp.asarray(self.batch_list), axis=0)
            new_std = cp.std(cp.asarray(self.batch_list), axis=0)
        self.batch_mean = (1 - self.momentum) * self.batch_mean + self.momentum * new_mean
        self.batch_std = (1 - self.momentum) * self.batch_std + self.momentum * new_std
        self.batch_list = []




class dropout(Norm):
    def __init__(self, drop_rate: float = 0.2, *parents, **kwargs):
        super(dropout, self).__init__(*parents, **kwargs)
        self.mask = 0
        self.retain_rate = 1 - drop_rate

    def calculate(self):
        if self.train_mode:
            if self.graph.cuda_device == 'cpu':
                self.mask = np.random.binomial(n=1, p=self.retain_rate, size=self.parents[0].value.shape)
            else:
                self.mask = cp.random.binomial(n=1, p=self.retain_rate, size=self.parents[0].value.shape)
            self.value = self.mask * self.parents[0].value/self.retain_rate
        else:
            self.value = self.parents[0].value

    def backward(self, parent):
        if self.judge_nan(self.grad):
            if self.graph.cuda_device == 'cpu':
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)  # gather grad
            else:
                self.grad = cp.sum(cp.asarray([child.backward(self) for child in self.children]), axis=0)  # gather grad
        return self.mask * self.grad
