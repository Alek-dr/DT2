import pandas as pd
import graphviz
from core.decision_tree import DecisionTree
from graph_visualize.dot_convertor import export2dot
from utils.help_functions import train_test_split

titanic = pd.read_csv('../../datasets/titanic/titanic.csv')
titanic.drop(['Name'], axis=1, inplace=True)
trainData, testData = train_test_split(titanic)

categorical = ('Survived','Pclass','Ticket')

out_name = 'titanic_not_pruned'

def train():
    c45 = DecisionTree()
    c45.C45(data=trainData, target='Survived', as_categorial=('Survived', 'Pclass', 'Ticket'))
    c45.save(out_name + '.pkl')
    export2dot(out_name, c45.tree, writeId=True, write=True)

def test():
    dt = DecisionTree()
    dt.load(out_name + '.pkl')

    Y = testData['Survived'].values
    res = dt.predict(testData, vector=False)
    # acc = sum(res == Y) / testData.shape[0]
    print(res)
    # print(acc)
    # dt.save(out_name + '.pkl')
    # export2dot(out_name, dt.tree, writeId=True, write=True)

if __name__=='__main__':
    # train()
    test()