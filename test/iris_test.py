import pandas as pd
from core.decision_tree import DecisionTree
from graph_visualize.dot_convertor import export2dot
from utils.help_functions import train_test_split

iris = pd.read_csv('../datasets/Iris.csv')
iris.drop("Id",axis=1,inplace=True)

def train():
    dt = DecisionTree()
    # train, test = train_test_split(iris, train=0.8)

    dt.C45(data=iris, target='Species')
    # print(dt.tree.branchStat)
    dt.save('iris.pkl')
    out_name = 'iris'
    export2dot(out_name, dt.tree, writeId=True, write=True)

def test():
    dt = DecisionTree()
    dt.load('iris.pkl')
    dt.tree._pruneSameChild_()
    # dt.save('iris2.pkl')
    # out_name = 'iris2'
    # export2dot(out_name, dt.tree, writeId=True)

if __name__=='__main__':
    train()
    # test()