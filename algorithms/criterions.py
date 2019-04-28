from numpy import log2, log, sum, round, nonzero, append, square, unique
import pandas as pd

SMALL = 1e-3


# region Entropy

def info(data, target, attr, q=2):
    """
    :param data: pandas DataFrame
    :param target: target column name
    :param attr: current attribute
    :param q: optional parametr. If q = 1 entropy is ordinary Shannon entropy.
    If q is real and n not equal 1, it becomes Tsallis entropy
    :return: entropy
    """
    data = data.loc[data[attr].notnull()]
    freq = data[target].value_counts().values
    v = freq / data.shape[0]
    return round((1 / (q - 1)) * -sum(v * log2(v)), 3)


def splitInfo(data, target):
    freq = data[target].value_counts().values
    miss = data.loc[data[target].isnull()].shape[0]
    if miss > 0:
        freq = append(freq, [miss], axis=0)
    v = freq / data.shape[0]
    return round(-sum(v * log2(v)), 3)


def splitInfoCont(data, target, thrsh):
    data = data.loc[data[target].notnull()]
    thrsh_sort = data[target].apply(lambda x: x >= thrsh)
    freq = thrsh_sort.value_counts().values
    miss = data.loc[data[target].isnull()].shape[0]
    if miss > 0:
        freq = append(freq, [miss], axis=0)
    v = freq / data.shape[0]
    return round(-sum(v * log2(v)), 3)


def req_bits(row, N):
    n = row.sum()
    vals = row.values / n
    vals = vals[nonzero(vals)]
    return -sum(vals * log2(vals)) * (n / N)


def info_x(data, attr, target, q=2):
    data = data.loc[data[attr].notnull()]
    freq = pd.crosstab(data[attr], data[target], normalize=False)
    N = data.shape[0]
    info = freq.apply(lambda row: req_bits(row, N), axis=1)
    return round((1 / (q - 1)) * info.sum(), 3)


# endregion

# region C4.5 numerical attribute handle

def set_threshold(values, curr_thrsh):
    """
    It's a strange point in every C4.5 implementation. We will continue the tradition
    """
    newThrsh = -max(values)
    for v in values:
        tempValue = v
        if (tempValue > newThrsh) & (tempValue <= curr_thrsh):
            newThrsh = tempValue
    return newThrsh


def distrEntropy(data, attr, target, q=2):
    data = data.loc[data[attr].notnull()]
    freq = data.groupby(target)['__W__'].sum().values
    v = sum(freq * log(freq))
    w = data['__W__'].sum()
    return (w * log(w) - v) / (log(2) * (q - 1))


def _splitEntropy_(data, attr, thrsh, W):
    unknown = W - data.loc[data[attr].notnull()]['__W__'].sum()
    leftSubs = data.loc[data[attr] <= thrsh]
    rightSubs = data.loc[data[attr] > thrsh]
    w1 = leftSubs['__W__'].sum()
    w2 = rightSubs['__W__'].sum()
    h = - w1 * log(w1) - w2 * log(w2)
    if unknown > 0:
        h -= unknown * log(unknown)
    h += W * log(W)
    return round(h / log(2), 3)


def gainRatio(data, attr, infoGain, thrsh):
    W = data['__W__'].sum()
    h = _splitEntropy_(data, attr, thrsh, W)
    if 0 <= h <= SMALL:
        return 0
    else:
        h /= W
    return round(infoGain / h, 3)


# endregion

# region Gini

def giniAttr(data, attr):
    data = data.loc[data[attr].notnull()]
    freq = pd.DataFrame(data.groupby([attr])['__W__'].sum()).unstack().fillna(0)
    W = data['__W__'].sum()
    g = 1 - square(freq.values / W).sum()
    return round(g,3)


def gini(data, target, attr):
    """
    :param data: pandas DataFrame
    :param target: target column name
    :param attr: current attribute
    :return: Gini index
    """
    data = data.loc[data[target].notnull()]
    freq = pd.DataFrame(data.groupby([attr, target])['__W__'].sum()).unstack().fillna(0)
    W = data['__W__'].sum()
    g = 0
    for _, row in freq.iterrows():
        w = row.sum()
        val = 1 - square(row.values / w).sum()
        g += (w / W) * val
    return round(g, 3)


def giniCont(data, target, attr):
    """
    :param data: pandas DataFrame
    :param target: target column name
    :param attr: current attribute
    :return: Gini index
    """
    data = data.loc[data[target].notnull()]
    vals = unique(data[attr].sort_values().values[:-1])
    if len(vals) <= 1:
        return 0, None
    thresholds = (vals[1:] - vals[0:-1]) / 2
    thrshGini = {}
    for v, dv in zip(vals, thresholds):
        data['__thrsh__'] = data[attr].apply(lambda x: x >= v + dv)
        g = gini(data, target, '__thrsh__')
        thrshGini[v + dv] = g
    bestSplit = min(thrshGini, key=thrshGini.get)
    g = thrshGini[bestSplit]
    data.drop(['__thrsh__'], axis=1, inplace=True)
    return g, bestSplit


# endregion

# region Donskoy

def D(data, target, attr):
    T = data[target].unique()
    freq = pd.DataFrame(data.groupby([target, attr])['__W__'].sum()).unstack().fillna(0)
    columns = freq.columns.values
    D_value = 0
    for i, col in enumerate(columns[:-1], 1):
        f = freq[col]
        for t in T:
            k = f[t]
            ind = freq.index.isin([t])
            vals = freq[~ind][columns[i:]].values.ravel()
            D_value += sum(k * vals)
    return D_value

def D_cont(data, target, attr):
    data = data.loc[data[target].notnull()]
    vals = unique(data[attr].sort_values().values[:-1])
    if len(vals) <= 1:
        return 0, None
    thresholds = (vals[1:] - vals[0:-1]) / 2
    thrshDonsky = {}
    for v, dv in zip(vals, thresholds):
        data['__thrsh__'] = data[attr].apply(lambda x: x >= v + dv)
        d = D(data, target, '__thrsh__')
        thrshDonsky[v + dv] = d
    bestSplit = min(thrshDonsky, key=thrshDonsky.get)
    d = thrshDonsky[bestSplit]
    data.drop(['__thrsh__'], axis=1, inplace=True)
    return d, bestSplit


# endregion
