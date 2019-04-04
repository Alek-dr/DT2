from algorithms.learn_tree import *
from pandas import DataFrame
from numpy import nan
import pickle

class DecisionTree():

    def __init__(self):
        self.tree = None
        self.data = None
        self.attrribute_properties = {}
        self.attribute_types = {}

    def _setAttrributeProperties(self, data, as_categorial=()):
        for attr in data:
            attrType = data[attr].dtype
            if attrType in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
                self.attribute_types[attr] = 0
            elif isinstance(attrType, object) or (attr in as_categorial):
                self.attrribute_properties[attr] = set(filter(lambda x: x not in [None, nan], data[attr].unique()))
                self.attribute_types[attr] = 1
            else:
                raise Exception("Unknown attribute type")

    def ID3(self, data, target, numerical=()):
        if isinstance(data,DataFrame):
            if data.empty:
                raise BaseException('Empty data')
            self._setAttrributeProperties(data,numerical)
            tree = Tree(data=data, target=target, attrProp=self.attrribute_properties)
            self.tree = tree._id3_(data,currId=0,parentId=-1)

    def C45(self, data, target, as_categorial=()):
        if isinstance(data,DataFrame):
            if data.empty:
                raise BaseException('Empty data')
            self._setAttrributeProperties(data,as_categorial)
            self.data = data
            tree = Tree(data=data, target=target, attrProp=self.attrribute_properties, attrTypes=self.attribute_types)
            self.tree = tree._c45_(self.data,currId=0,parentId=-1)
            self.tree._pruneSameChild_()

    def predict(self, example):
        if self.tree._initialized:
            if isinstance(example, DataFrame):
                self.tree.predict(example)

    def save(self, name):
        with open(name, 'wb') as f:
            pickle.dump(self,f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

    def load(self, name):
        with open(name,'rb') as f:
            dt = pickle.load(f)
            f.close()
        self.tree = dt.tree
        self.data = dt.data
        self.attribute_types = dt.attribute_types
        self.attrribute_properties = dt.attrribute_properties