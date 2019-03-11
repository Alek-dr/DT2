from numpy import log2, log, sum, round, nonzero, append
import pandas as pd

SMALL = 1e-3

def info(data, target, attr):
    data = data.loc[data[attr].notnull()]
    freq = data[target].value_counts().values
    v = freq / data.shape[0]
    return round(-sum(v * log2(v)), 3)

def splitInfo(data, target):
    freq = data[target].value_counts().values
    miss = data.loc[data[target].isnull()].shape[0]
    if miss > 0:
        freq = append(freq,[miss],axis=0)
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

def info_x(data, attr, target):
    data = data.loc[data[attr].notnull()]
    freq = pd.crosstab(data[attr], data[target], normalize=False)
    N = data.shape[0]
    info = freq.apply(lambda row: req_bits(row, N), axis=1)
    return round(info.sum(), 3)

def info_cont_x(data, attr, thrsh, target):
    data = data.loc[data[attr].notnull()]
    thrsh_sort = data[attr].apply(lambda x: x <= thrsh)
    freq = pd.crosstab(thrsh_sort, data[target], normalize=False)
    N = data.shape[0]
    info = freq.apply(lambda row: req_bits(row, N), axis=1)
    return round(info.sum(), 3)

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

def distrEntropy(data,attr,target):
    data = data.loc[data[attr].notnull()]
    freq = data.groupby(target)['__W__'].sum().values
    v = sum(freq*log(freq))
    w = data['__W__'].sum()
    return (w*log(w) - v)/log(2)

def _splitEntropy_(data, attr, thrsh, W):
    #W = data['__W__'].sum()
    unknown = W - data.loc[data[attr].notnull()]['__W__'].sum()
    leftSubs = data.loc[data[attr] <= thrsh]
    rightSubs = data.loc[data[attr] > thrsh]
    w1 = leftSubs['__W__'].sum()
    w2 = rightSubs['__W__'].sum()
    h = - w1*log(w1) - w2*log(w2)
    if unknown > 0:
        h -= unknown*log(unknown)
    h += W*log(W)
    return round(h/log(2),3)

def gainRatio(data,attr,infoGain,thrsh):
    W = data['__W__'].sum()
    h = _splitEntropy_(data, attr, thrsh, W)
    if 0 <= h <= SMALL:
        return 0
    else:
        h /= W
    return round(infoGain/h,3)

# endregion