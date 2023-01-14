class Graph:
    """
    Computation graph
    """

    def __init__(self):
        # nodes are stored in a list
        self.nodes = []
        self.cuda_device = 'cpu'

    def insert_node(self, node):
        """
        Insert a node into the computation graph at the end
        :param node: Node
        :return: None
        """
        self.nodes.append(node)

    def zero_gradient(self):
        """
        Set the gradient of all nodes as 0.
        """
        for node in self.nodes:
            node.zero_gradient()

    def set_gpu(self, index: str):
        self.cuda_device = index
        for node in self.nodes:
            node.set_gpu(index)


global_graph = Graph()
