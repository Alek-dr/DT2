from core.graph import Graph, Node
from algorithms.criterions import *
from numpy import round, unique, array, dtype
from utils.help_functions import groupSameElements
from operator import itemgetter
import numpy as np


def _getPartition_(data, attr):
    num = data.loc[data[attr].notnull()]['__W__'].sum()
    den = data['__W__'].sum()
    return round(num / den, 3)


class Tree(Graph):

    def __init__(self, data, target, attrProp, attrTypes):
        Graph.__init__(self)
        self.target = target
        self.data = data
        self.data['__W__'] = [1 for _ in range(len(data))]
        self.attrributeProperties = attrProp
        self.attrributeTypes = attrTypes
        self.connectionProp = []
        self.nClasses = len(attrProp[target])
        self.minObj = 2
        self.fmt = 'U' + str(max([len(lbl) for lbl in attrProp[target]]))

    def _id3_(self, data, currId, parentId):

        node = Node(id=currId)

        # Check if all target values are equal
        if len(data[self.target].unique()) == 1:
            node.type = 'leaf'
            node.attr = data.iloc[0][self.target]
            self.addNode(node, parentId)
        # If only target
        elif len(data.columns) == 1:
            node.type = 'leaf'
            most_freq = data[self.target].value_counts().idxmax()
            node.attr = most_freq
            self.addNode(node, parentId)
        else:
            initial_entropy = info(data, self.target, self.target)
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
                self.connectionProp.append({(node.id, self._next_id()): prop})
                self._id3_(sub, currId=self._next_id(), parentId=node.id)
        return self

    def _handleCategorial_(self, data, attr):
        initial_entropy = info(data, self.target, attr=attr)
        F = _getPartition_(data, attr)
        I = info_x(data, attr, self.target)
        gain = round(F * (initial_entropy - I), 3)
        splInfo = splitInfo(data, attr)
        return round(gain / splInfo, 3)

    def _handleNumerical_(self, data, attr):
        if data.empty:
            return 0, None
        minSplit = 0.1 * (data['__W__'].sum() / self.nClasses)
        if minSplit <= self.minObj:
            minSplit = self.minObj
        elif minSplit >= 25:
            minSplit = 25

        W_attr = data.loc[data[attr].notnull()]['__W__'].sum()
        if W_attr <= 2 * minSplit:
            return 0, None

        startEntr = distrEntropy(data, attr, self.target)
        vals = unique(data[attr].sort_values().values[:-1])
        thresholds = (vals[1:] - vals[0:-1]) / 2
        n, infoGain = 0, 0
        bestSplit = vals[0] + thresholds[0]
        W = data['__W__'].sum()
        for v, dv in zip(vals, thresholds):
            leftSubs = data.loc[data[attr] <= v + dv]
            rightSubs = data.loc[data[attr] > v + dv]
            if (leftSubs.shape[0] >= minSplit) and (rightSubs.shape[0] >= minSplit):
                n += 1
                newEntropy = distrEntropy(leftSubs, attr, self.target) + distrEntropy(rightSubs, attr, self.target)
                known = W - W_attr
                rate = known / W
                num = (startEntr - newEntropy) * (1 - rate)
                if 0 <= num <= SMALL:
                    currInfoGain = 0
                else:
                    currInfoGain = num / W_attr
                if currInfoGain >= infoGain:
                    infoGain = currInfoGain
                    bestSplit = v + dv
        # there no usefull split
        if n == 0:
            return 0, None
        infoGain -= log2(n) / W_attr
        if 0 >= infoGain <= SMALL:
            return 0, None
        else:
            gr = gainRatio(data, attr, infoGain, bestSplit)
            return gr, bestSplit

    def _enoughInstances_(self, data):
        W = data['__W__'].sum()
        mostFreq = data[self.target].value_counts().idxmax()
        N = data.loc[data[self.target] == mostFreq]['__W__'].sum()
        if (W > 2 * self.minObj) & (W > N):
            return True, mostFreq
        else:
            return False, mostFreq

    def _getStat_(self, data):
        names, weights = [], []
        # weights per class
        for T in self.attrributeProperties[self.target]:
            w = data.loc[data[self.target] == T]['__W__'].sum()
            names.append(T)
            weights.append(w)
        weightsPerClass = np.zeros(len(names), dtype={'names': ('label', 'weight'),
                                                      'formats': (self.fmt, 'f8')})
        weightsPerClass['label'] = names
        weightsPerClass['weight'] = weights
        # Total weight
        W = data['__W__'].sum()
        del names, weights
        return {'W': W, 'WeightsPerClass': weightsPerClass}

    def _leafNode_(self, node, data):
        node.type = 'leaf'
        most_freq = data[self.target].value_counts().idxmax()
        node.attr = most_freq
        node.stat = self._getStat_(data)
        return node

    def _pruneConnect_(self, parent, chId):
        for id in chId:
            edge = (parent, id)
            for conn in self.connectionProp:
                if edge in conn:
                    self.connectionProp.remove(conn)

    def _mergeNodes_(self, parent, chId):
        allProp = []
        for id in chId:
            edge = (parent, id)
            for conn in self.connectionProp:
                if edge in conn:
                    allProp.append(conn[edge])
        edge = (parent, chId[0])
        ind = [i for i in range(len(self.connectionProp)) if edge in self.connectionProp[i]][0]
        self.connectionProp[ind] = {edge: ';'.join(allProp)}

    def _c45_(self, data, currId, parentId):

        node = Node(id=currId)

        # Check if all target values are equal
        if len(data[self.target].unique()) == 1:
            node.type = 'leaf'
            node.attr = data.iloc[0][self.target]
            node.stat = self._getStat_(data)
            self.addNode(node, parentId)
            return self
        # If only target
        elif len(data.columns) == 1:
            node = self._leafNode_(node, data)
            self.addNode(node, parentId)
            return self
        enough, mostFreq = self._enoughInstances_(data)
        if not enough:
            node.type = 'leaf'
            node.attr = mostFreq
            node.stat = self._getStat_(data)
            self.addNode(node, parentId)
            return self
        else:
            gainRatio, attrThrsh = {}, {}
            for attr in data.columns:
                if (attr != self.target) and (attr != '__W__'):
                    if self.attrributeTypes[attr] == 1:
                        # categorical attribute
                        gainRatio[attr] = self._handleCategorial_(data, attr)
                    else:
                        # continous attribute
                        gr, thrsh = self._handleNumerical_(data, attr)
                        gainRatio[attr] = gr
                        attrThrsh[attr] = thrsh

            best_attr = max(gainRatio, key=gainRatio.get)

            node.attr = best_attr
            node.type = 'inner'
            node.stat = self._getStat_(data)

            if node.id == 0:
                self.setRootNode(node)
            else:
                self.addNode(node, parentId=parentId)

            N = data.loc[data[best_attr].notnull()]['__W__'].sum()
            if self.attrributeTypes[best_attr] == 0:
                thrsh = attrThrsh[best_attr]
                thrsh = set_threshold(self.data[best_attr], thrsh)
                lessOrEq = data.loc[data[best_attr] <= thrsh]
                self.connectionProp.append({(node.id, self._next_id()): '<= {}'.format(thrsh)})
                self._c45_(lessOrEq, currId=self._next_id(), parentId=node.id)
                great = data.loc[data[best_attr] > thrsh]
                self.connectionProp.append({(node.id, self._next_id()): '> {}'.format(thrsh)})
                self._c45_(great, currId=self._next_id(), parentId=node.id)
            else:
                for prop in self.attrributeProperties[best_attr]:
                    sub = data.loc[data[best_attr] == prop, data.columns != best_attr]
                    w = round(sub['__W__'].sum() / N, 3)
                    unknown = data.loc[data[best_attr].isnull(), data.columns != best_attr].copy()
                    unknown['__W__'] = [w for _ in range(unknown.shape[0])]
                    sub = pd.concat([sub, unknown], sort=False)
                    self.connectionProp.append({(node.id, self._next_id()): prop})
                    self._c45_(sub, currId=self._next_id(), parentId=node.id)
        return self

    def _pruneSameChild_(self):
        groups = self.groupLeafByParent()
        for parent, childs in groups.items():
            if len(childs) > 1:
                labels = [self.getNode(id).attr for id in childs]
                sameLbls = groupSameElements(labels)
                if len(sameLbls) > 1:
                    for group, leafs in zip(groups, sameLbls):
                        if len(leafs) > 1:
                            ind_leafs = groups[group]
                            ind_leafs = itemgetter(*leafs)(ind_leafs)
                            self._makeOneNode_(group, ind_leafs)
                            self._mergeNodes_(group, ind_leafs)
                            self._pruneConnect_(group, ind_leafs[1:])
                else:
                    # prune leafs
                    for eq in sameLbls:
                        chId = [childs[i] for i in eq]
                        self.prune(parent, chId)
                        self._pruneConnect_(parent, chId)

    def _getConnections_(self, id):
        return [conn for conn in self.connectionProp for edge in conn.keys() if edge[0] == id]

    def _predict_(self, example, node):
        if node.type == 'leaf':
            return node.attr
        else:
            test = node.attr
            val = example[test].values[0]
            connections = self._getConnections_(node.id)
            nextNode = None
            if val is not None:
                for connect in connections:
                    for k, v in connect.items():
                        if v == val:
                            nextNode = self.getNode(k[1])
                            break
                if nextNode is not None:
                    return self._predict_(example, nextNode)
                else:
                    raise Exception('There no subtree for {}'.format(test))
            else:
                pass
