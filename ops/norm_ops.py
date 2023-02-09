from computation_graph import Node


class Norm(Node):
    def __init__(self, *parents, **kwargs):
        super(Norm, self).__init__(*parents, **kwargs)
        assert len(self.parents) == 1
        self.train_mode = True

    def set_train_mode(self):
        self.train_mode = True

    def set_eval_mode(self):
        self.train_mode = False


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
        self.gather_grad()
        return self.grad * (self.batch_std + self.eps)

    def refresh(self):
        new_mean = self.P.mean(self.P.asarray(self.batch_list), axis=0)
        new_std = self.P.std(self.P.asarray(self.batch_list), axis=0)
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
            self.mask = self.P.random.binomial(n=1, p=self.retain_rate, size=self.parents[0].value.shape)
            self.value = self.mask * self.parents[0].value / self.retain_rate
        else:
            self.value = self.parents[0].value

    def backward(self, parent):
        self.gather_grad()
        return self.mask * self.grad
