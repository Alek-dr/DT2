import pandas as pd
from dtrees.core.decision_tree import DecisionTree
from dtrees.graph_visualize.dot_convertor import export2dot

out_name = 'tennis'


def test_1():
    """
    Train decision tree with C4.5 and save model and tree as dot file
    """
    tennis = pd.read_csv('datasets/PlayTennis_2.csv')
    tennis.drop("Day", axis=1, inplace=True)

    dt = DecisionTree()
    dt.C45(data=tennis, target='Play')
    export2dot(out_name, dt.tree, writeId=True, write=True)
    dt.save(out_name + '.pkl')


def test_2():
    """
    Load file from test_1 and predict values
    """
    dt = DecisionTree()
    dt.load(out_name + '.pkl')
    tennis = pd.read_csv('datasets/PlayTennis_2.csv')
    tennis.drop("Day", axis=1, inplace=True)
    res = dt.predict(tennis, vector=False)
    print(res)


def test_3():
    """
    Load model and predict new example
    :return:
    """
    dt = DecisionTree()
    dt.load(out_name + '.pkl')
    example = pd.DataFrame({'Outlook': 'Sunny', 'Temperature': 70, 'Humidity': None, 'Wind': 'No'}, index=[0])

    tennis = pd.read_csv('datasets/PlayTennis_2.csv')
    tennis.drop("Day", axis=1, inplace=True)
    res = dt.predict(example, vector=False)
    print(res)


def test_4():
    """
    Learn model with params
    """
    tennis = pd.read_csv('datasets/PlayTennis_2.csv')
    tennis.drop("Day", axis=1, inplace=True)

    dt = DecisionTree()
    params = {
        'criterion': 'entropy',
        'alpha': 4,
        'pruneLevel': 2,
        'minSamples': 0.1
    }
    dt.learn(tennis, 'Play', params=params)


if __name__ == '__main__':
    test_1()
