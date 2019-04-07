import pandas as pd
from core.decision_tree import DecisionTree
from graph_visualize.dot_convertor import export2dot
from utils.help_functions import train_test_split
from core.pruning import errorBasedPruning

iris = pd.read_csv('../../datasets/Iris.csv')
iris.drop("Id",axis=1,inplace=True)

trainData, testData = train_test_split(iris, train=0.8)

out_name = 'pruned'

def train():
    dt = DecisionTree()
    dt.C45(data=iris, target='Species')
    dt.save(out_name + '.pkl')
    export2dot(out_name, dt.tree, writeId=True, write=True)

def test():
    dt = DecisionTree()
    dt.load(out_name + '.pkl')

    Y = testData['Species'].values
    res = dt.predict(testData, vector=True)
    acc = sum(res == Y) / testData.shape[0]
    print(acc)

    # dt.tree._pruneSameChild_()
    # errorBasedPruning(dt.tree)
    # dt.tree._pruneSameChild_()
    # dt.save('full_pruned.pkl')
    # export2dot('full_pruned', dt.tree, writeId=True, write=True)

if __name__=='__main__':
    # train()
    test()