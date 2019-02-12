class Node():

    def __init__(self, id, nodeType='leaf'):
        self.id = id
        self.type = nodeType
        self.attr = None

class Graph():

    def __init__(self):
        self.nodes = []
        self.edges = []
        self._initialized = False

    def addNode(self, node, parentId):
        if isinstance(node, Node) or hasattr(node, 'id'):
            if isinstance(parentId, int):
                self._addEdge_((parentId, node.id))
                self.nodes.append(node)

    def setRootNode(self, node):
        if isinstance(node, Node) or hasattr(node, 'id'):
            if node.id != 0:
                node.id = 0
            self.nodes.append(node)
            self._initialized = True

    def getNode(self, id):
        for node in self.nodes:
            if node.id == id:
                return node

    def _next_id(self):
        if len(self.nodes) == 0:
            return 0
        else:
            max_id = max([node.id for node in self.nodes])
            return max_id + 1

    def _addEdge_(self, edge):
        if edge is not None:
            if isinstance(edge, list) or isinstance(edge, tuple):
                if len(edge) == 2:
                    self.edges.append(tuple(edge))
