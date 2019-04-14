from algorithms.learn_tree import *
from core.pruning import errorBasedPruning
from pandas import DataFrame, Series
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
                if attr in as_categorial:
                    self.attrribute_properties[attr] = set(filter(lambda x: x not in [None, nan], data[attr].unique()))
                    self.attribute_types[attr] = 1
                else:
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

    def C45(self, data, target, as_categorial=(), pruneLevel=0):
        if isinstance(data,DataFrame):
            if data.empty:
                raise BaseException('Empty data')
            self._setAttrributeProperties(data,as_categorial)
            self.data = data.copy()
            tree = Tree(data=self.data, target=target, attrProp=self.attrribute_properties, attrTypes=self.attribute_types)
            self.tree = tree._c45_(self.data,currId=0,parentId=-1)
            if pruneLevel == 1:
                errorBasedPruning(self.tree)
            elif pruneLevel == 2:
                errorBasedPruning(self.tree)
                self.tree._pruneSameChild_()
            elif pruneLevel == 3:
                self.tree._pruneSameChild_()
                errorBasedPruning(self.tree)
                wasPruned = self.tree._pruneSameChild_()
                while wasPruned:
                    wasPruned = self.tree._pruneSameChild_()
            del self.tree.data
            del self.data

    def learn(self, data, target, as_categorial=(), criterion='Gini', pruneLevel=0 ,q=2):
        if isinstance(data,DataFrame):
            if data.empty:
                raise BaseException('Empty data')
            self._setAttrributeProperties(data,as_categorial)
            self.data = data.copy()
            tree = Tree(data=self.data, target=target, attrProp=self.attrribute_properties, attrTypes=self.attribute_types)
            if criterion == 'Tsallis':
                q = 2 if q==1 else q
            self.tree = tree._c45_(self.data, currId=0, parentId=-1, q=q, criterion=criterion)
            if pruneLevel == 1:
                errorBasedPruning(self.tree)
            elif pruneLevel == 2:
                errorBasedPruning(self.tree)
                self.tree._pruneSameChild_()
            elif pruneLevel == 3:
                self.tree._pruneSameChild_()
                errorBasedPruning(self.tree)
                wasPruned = self.tree._pruneSameChild_()
                while wasPruned:
                    wasPruned = self.tree._pruneSameChild_()
            del self.tree.data
            del self.data

    def predict(self, example, vector=True):
        if self.tree._initialized:
            res = None
            if isinstance(example, DataFrame):
                rootNode = self.tree.getNode(0)
                res = pd.DataFrame(columns=self.tree.targetLbls)
                for _, ex in example.iterrows():
                    y = self.tree._predict_(ex,rootNode)
                    # y = pd.DataFrame(y).T
                    res = res.append(y, ignore_index=True)
            elif isinstance(example, Series):
                rootNode = self.tree.getNode(0)
                res = pd.DataFrame(self.tree._predict_(example, rootNode), index=[0])
            if vector:
                res = res.idxmax(axis=1)
                return res
            else:
                return res

    def save(self, name):
        with open(name, 'wb') as f:
            pickle.dump(self,f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

    def load(self, name):
        with open(name,'rb') as f:
            dt = pickle.load(f)
            f.close()
        self.tree = dt.tree
        self.attribute_types = dt.attribute_types
        self.attrribute_properties = dt.attrribute_properties