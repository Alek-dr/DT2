import pandas as pd
from core.decision_tree import DecisionTree
from graph_visualize.dot_convertor import export2dot

from core.graph import Node
from numpy import zeros

def train():
    tennis = pd.read_csv('../datasets/PlayTennis_2.csv')
    tennis.drop("Day", axis=1, inplace=True)

    dt = DecisionTree()
    dt.C45(data=tennis, target='Play')
    out_name = 'tennis_2'
    # export2dot(out_name, dt.tree, writeId=True, write=True)
    #
    # dt.save(out_name+'.pkl')

def test_1():
    dt = DecisionTree()
    dt.load('tennis_2.pkl')
    example = pd.DataFrame({'Outlook': 'Sunny','Temperature': 70, 'Humidity': None, 'Wind':'No'}, index=[0])
    # example = pd.DataFrame({'Outlook': 'Rain', 'Temperature': 70, 'Humidity': 73, 'Wind': None}, index=[0])

    tennis = pd.read_csv('../datasets/PlayTennis_2.csv')
    tennis.drop("Day", axis=1, inplace=True)
    Y = tennis['Play'].values
    res = dt.predict(example, vector=False)
    print(res)
    # acc = sum(res==Y) / tennis.shape[0]
    # print(acc)

def test_2():
    dt = DecisionTree()
    dt.load('tennis_2.pkl')
    node = Node(8)
    node.attr = 'No'

    row = pd.DataFrame(columns=dt.tree.branchStat.columns, index=[(1,8)])
    row['Yes'] = [1]
    row['No'] = [2]
    dt.tree.branchStat = dt.tree.branchStat.append(row, ignore_index = False)
    node.stat = pd.Series(data=[1,2], index=['Yes','No'])
    dt.tree.addNode(node,1)
    dt.tree._pruneSameChild_()

if __name__=='__main__':
    # train()
    test_1()
    # test_2()
