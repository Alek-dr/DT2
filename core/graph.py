import numpy as np

class Node():

    def __init__(self, id, nodeType='leaf'):
        self.id = id
        self.type = nodeType
        self.stat = None
        self.attr = None

    def nodeStat(self, stat):
        """param stat: this field can be anything that is statistic"""
        self.stat = stat

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

    def getChilds(self, id):
        childs = [edge[1] for edge in self.edges if edge[0] == id]
        return childs

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

    def getParentId(self,id):
        """
        :param id: int id of leaf node
        :return: parent node id
        """
        for edge in self.edges:
            if edge[1]==id:
                return edge[0]

    def groupLeafByParent(self):
        leafInd = [node.id for node in self.nodes if node.type == 'leaf']
        parents = [self.getParentId(id) for id in leafInd]
        group = {}
        for i, id in enumerate(leafInd):
            if parents[i] not in group:
                group[parents[i]] = [id]
            else:
                group[parents[i]].append(id)
        del leafInd, parents
        return group

    def _makeOneNode_(self,parentId, nodesId):
        # merge to node 0
        nodes = [self.getNode(id) for id in nodesId]
        WPC = nodes[0].stat['WeightsPerClass']
        for node in nodes[1:]:
            WPC['weight'] += node.stat['WeightsPerClass']['weight']
        W = WPC['weight'].sum()
        new_stat = {'W': W, 'WeightsPerClass': WPC}

        for node in nodes[1:]:
            edge = (parentId, node.id)
            self.edges.remove(edge)
            self.nodes.remove(node)
        nodes[0].stat = new_stat

    def prune(self, parentId, childs):
        for id in childs:
            edge = (parentId, id)
            self.edges.remove(edge)
            node = self.getNode(id)
            self.nodes.remove(node)
        node = self.getNode(parentId)
        wpc = node.stat['WeightsPerClass']
        ind = np.argmax(wpc['weight'])
        lbl = wpc['label'][ind]
        node.attr = lbl
        node.type = 'leaf'
