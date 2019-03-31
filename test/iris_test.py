import pandas as pd
from core.decision_tree import DecisionTree
from graph_visualize.dot_convertor import export2dot

tennis = pd.read_csv('../datasets/Iris.csv')
tennis.drop("Id",axis=1,inplace=True)

dt = DecisionTree()
dt.C45(data=tennis, target='Species')


# dt.load('iris.pkl')

dt.save('iris.pkl')

out_name = 'iris'
export2dot(out_name, dt.tree)