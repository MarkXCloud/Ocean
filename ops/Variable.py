from computation_graph import Node
import numpy as np
import cupy as cp


class Variable(Node):
    def __init__(self, is_Train=False, **kwargs):
        super(Variable, self).__init__(**kwargs)
        self.is_Train = is_Train

    def set_value(self, value):
        """
        set value of variable
        """
        self.zero_value()
        self.value = value

    def backward(self, current_node):
        if self.judge_nan(self.grad):
            self.grad = self.P.sum(self.P.asarray([child.backward(self) for child in self.children]),
                                   axis=0)  # gather grad
        return self.grad

    def set_gpu(self, index):
        self.value = cp.asarray(self.value)
        self.P = cp
        self.use_cuda = True
