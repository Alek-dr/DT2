from algorithms.learn_tree import *
from pandas import DataFrame

class DecisionTree():

    def __init__(self):
        self.tree = None
        self.attrribute_properties = {}

    def _setAttrributeProperties(self, data, numerical=()):
        for attr in data:
            if attr not in numerical:
                self.attrribute_properties[attr] = set(data[attr].unique())

    def ID3(self, data, target, numerical=()):
        if isinstance(data,DataFrame):
            if data.empty:
                raise BaseException('Empty data')
            self._setAttrributeProperties(data,numerical)
            tree = Tree(target=target, attrProp=self.attrribute_properties)
            self.tree = tree._id3_(data,currId=0,parentId=-1)
