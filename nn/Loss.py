from computation_graph import Node
from ops import MSELoss, CrossEntropyLoss
from computation_graph import global_graph


class LossNodeAdder:
    def __init__(self):
        self.model_graph = global_graph

    def forward(self, pred: Node, target: Node):
        raise NotImplementedError

    def __call__(self, pred: Node, target: Node, *args, **kwargs):
        return self.forward(pred, target)


class MSE(LossNodeAdder):
    def forward(self, pred: Node, target: Node):
        return MSELoss(pred, target)


class CELoss(LossNodeAdder):
    def forward(self, pred: Node, target: Node):
        return CrossEntropyLoss(pred, target)
