import pandas as pd
from core.decision_tree import DecisionTree
from graph_visualize.dot_convertor import export2dot
from scipy import stats
from core.graph import Node

out_name = 'tennis'

def train():
    tennis = pd.read_csv('../../datasets/PlayTennis_2.csv')
    tennis.drop("Day", axis=1, inplace=True)

    dt = DecisionTree()
    dt.C45(data=tennis, target='Play')
    export2dot(out_name, dt.tree, writeId=True, write=True)
    dt.save(out_name+'.pkl')

def test_1():
    dt = DecisionTree()
    dt.load(out_name + '.pkl')
    example = pd.DataFrame({'Outlook': 'Sunny','Temperature': 70, 'Humidity': None, 'Wind':'No'}, index=[0])
    # example = pd.DataFrame({'Outlook': 'Rain', 'Temperature': 70, 'Humidity': 73, 'Wind': None}, index=[0])

    tennis = pd.read_csv('../../datasets/PlayTennis_2.csv')
    tennis.drop("Day", axis=1, inplace=True)
    Y = tennis['Play'].values
    res = dt.predict(tennis, vector=False)
    print(res)
    # acc = sum(res==Y) / tennis.shape[0]
    # print("Accuracy = {:.3f}".format(acc))

# def binom_interval(success, total, confint=0.95):
#     quantile = (1 - confint) / 2.
#     lower = beta.ppf(quantile, success, total - success + 1)
#     upper = beta.ppf(1 - quantile, success + 1, total - success)
#     return (lower, upper)

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

def test_3():
    tennis = pd.read_csv('../../datasets/PlayTennis_2.csv')
    tennis.drop("Day", axis=1, inplace=True)

    dt = DecisionTree()
    dt.learn(tennis,'Play',criterion='D')
    out_name = 'tennisD'
    # export2dot(out_name, dt.tree, writeId=True, write=False)
    # dt.save(out_name + '.pkl')


if __name__=='__main__':
    # train()
    test_3()
    # test_2()
    # alpha = 0.25
    # z = stats.norm.ppf(1 - alpha/2)
    # print(z)
    #
    # f = stats.binom.pmf(k=6,n=6,p=alpha)
    # print(f)