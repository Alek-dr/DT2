import pandas as pd
from core.decision_tree import DecisionTree
from graph_visualize.dot_convertor import export2dot

from core.graph import Node
from numpy import zeros


def train():
    tennis = pd.read_csv('../datasets/Zenit.csv')
    tennis.drop("Day", axis=1, inplace=True)

    dt = DecisionTree()
    dt.C45(data=tennis, target='Win')

    out_name = 'zenit'
    export2dot(out_name, dt.tree)
    dt.save(out_name + '.pkl')


def test():
    dt = DecisionTree()
    dt.load('zenit.pkl')

    # add one more subtree
    node = Node(5)
    node.attr = 'Yes'
    weightsPerClass = zeros(2, dtype={'names': ('label', 'weight'),
                                      'formats': ('U3', 'f8')})
    weightsPerClass['label'] = ['Yes', 'No']
    weightsPerClass['weight'] = [2, 1]
    W = 3
    stat = {'W': W, 'WeightsPerClass': weightsPerClass}
    node.stat = stat
    dt.tree.addNode(node, 2)
    dt.tree.connectionProp.append({(2, 5): 'Unknown'})
    dt.attrribute_properties['Leaders'] = set(['Yes', 'No', 'Unknown'])
    dt.tree._pruneSameChild_()
    print(dt.tree.edges)


if __name__ == '__main__':
    # train()
    test()
