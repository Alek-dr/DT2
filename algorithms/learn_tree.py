from core.graph import Graph, Node
from algorithms.criterions import *


class Tree(Graph):

    def __init__(self, target, attrProp):
        Graph.__init__(self)
        self.target = target
        self.attrributeProperties = attrProp
        self.connectionProp = []

    def _id3_(self, data, currId, parentId):

        print(data)
        print(20*'-')

        node = Node(id=currId)

        # Check if all target values are equal
        if len(data[self.target].unique()) == 1:
            node.type = 'leaf'
            node.attr = data.iloc[0][self.target]
            self.addNode(node,parentId)
        # If only target
        elif len(data.columns) == 1:
            node.type = 'leaf'
            most_freq = data[self.target].value_counts().idxmax()
            node.attr = most_freq
            self.addNode(node, parentId)
        else:
            initial_entropy = info(data, self.target)
            gain = {}
            for attr in data.columns:
                if attr != self.target:
                    I = info_x(data, attr, self.target)
                    gain[attr] = round(initial_entropy - I, 3)

            best_attr = max(gain, key=gain.get)

            node.attr = best_attr
            node.type = 'inner'

            if node.id == 0:
                self.setRootNode(node)
            else:
                self.addNode(node, parentId=parentId)

            for prop in self.attrributeProperties[best_attr]:
                sub = data.loc[data[best_attr] == prop, data.columns != best_attr]
                self.connectionProp.append({(node.id,self._next_id()) : prop})
                self._id3_(sub, currId=self._next_id(), parentId=node.id)

        return self