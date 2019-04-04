import pandas as pd
from core.decision_tree import DecisionTree
from graph_visualize.dot_convertor import export2dot

tennis = pd.read_csv('../datasets/Iris.csv')
tennis.drop("Id",axis=1,inplace=True)

def train():
    dt = DecisionTree()
    dt.C45(data=tennis, target='Species')
    dt.save('iris.pkl')
    out_name = 'iris'
    export2dot(out_name, dt.tree, writeId=True)

def test():
    dt = DecisionTree()
    dt.load('iris.pkl')
    dt.tree._pruneSameChild_()
    dt.save('iris2.pkl')
    out_name = 'iris2'
    export2dot(out_name, dt.tree, writeId=True)

if __name__=='__main__':
    test()