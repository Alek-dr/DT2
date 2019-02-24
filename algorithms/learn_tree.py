from core.graph import Graph, Node
from algorithms.criterions import *
from numpy import array, round, unique

class Tree(Graph):

    def __init__(self, data, target, attrProp, attrTypes):
        Graph.__init__(self)
        self.target = target
        self.data = data
        self.attrributeProperties = attrProp
        self.attrributeTypes = attrTypes
        self.connectionProp = []

    def _id3_(self, data, currId, parentId):

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

    def _c45_(self, data, currId, parentId):

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
            gainRatio, attrThrsh, thrsh = {}, {}, 0
            for attr in data.columns:
                if attr != self.target:
                    if self.attrributeTypes[attr] == 1:
                        # categorical attribute
                        I = info_x(data, attr, self.target)
                        gain = round(initial_entropy - I, 3)
                        splitInfo = info(data, attr)
                        gainRatio[attr] = round(gain/splitInfo,3)
                    else:
                        # continous attribute
                        vals = unique(data[attr].sort_values().values[:-1])
                        thresholds = (vals[1:] - vals[0:-1])/2
                        thrsh_gain = array([(val+dv, initial_entropy-info_cont_x(data, attr, val+dv, self.target)) for dv, val in zip(thresholds,vals)])
                        round(thrsh_gain, decimals=4, out=thrsh_gain)
                        i = thrsh_gain[:,1].argmax()
                        max_gain, thrsh = thrsh_gain[i,1], thrsh_gain[i,0]
                        splitInfo = info_cont(data, attr, thrsh)
                        gainRatio[attr] = round(max_gain/splitInfo,3)

            best_attr = max(gainRatio, key=gainRatio.get)

            node.attr = best_attr
            node.type = 'inner'

            if node.id == 0:
                self.setRootNode(node)
            else:
                self.addNode(node, parentId=parentId)

            if self.attrributeTypes[best_attr] == 0:
                thrsh = set_threshold(self.data[best_attr],thrsh)
                lessOrEq = data.loc[data[best_attr] <= thrsh]
                attrShortName = best_attr[0:3] if len(best_attr)>=3 else best_attr
                self.connectionProp.append({(node.id, self._next_id()): '{} <= {}'.format(attrShortName, thrsh)})
                self._c45_(lessOrEq, currId=self._next_id(), parentId=node.id)
                great = data.loc[data[best_attr] > thrsh]
                self.connectionProp.append({(node.id, self._next_id()): '{} > {}'.format(attrShortName, thrsh)})
                self._c45_(great, currId=self._next_id(), parentId=node.id)
            else:
                for prop in self.attrributeProperties[best_attr]:
                    sub = data.loc[data[best_attr] == prop, data.columns != best_attr]
                    self.connectionProp.append({(node.id,self._next_id()) : prop})
                    self._c45_(sub, currId=self._next_id(), parentId=node.id)
        return self