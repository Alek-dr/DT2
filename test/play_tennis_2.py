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
    export2dot(out_name, dt.tree)

    dt.save(out_name+'.pkl')

def test_1():
    dt = DecisionTree()
    dt.load('tennis_2.pkl')
    example = pd.DataFrame({'Outlook': 'Sunny','Temperature': 70, 'Humidity': None, 'Wind':'No'}, index=[0])
    dt.tree._predict_(example, dt.tree.getNode(0))

def test_2():
    dt = DecisionTree()
    dt.load('tennis_2.pkl')
    node = Node(8)
    node.attr = 'No'

    weightsPerClass = zeros(2, dtype={'names': ('label', 'weight'),
                                                  'formats': ('U3', 'f8')})
    weightsPerClass['label'] = ['Yes','No']
    weightsPerClass['weight'] = [1,2]
    W = 3
    stat = {'W': W, 'WeightsPerClass': weightsPerClass}
    node.stat = stat
    dt.tree.addNode(node,4)
    dt.tree._pruneSameChild_()

if __name__=='__main__':
    #train()
    test_2()
