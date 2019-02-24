from numpy import log2, sum, round, nonzero
import pandas as pd

def info(data, target):
    freq = data[target].value_counts().values
    v = freq / data.shape[0]
    return round(-sum(v * log2(v)), 3)

def info_cont(data,target, thrsh):
    thrsh_sort = data[target].apply(lambda x: x >= thrsh)
    freq = thrsh_sort.value_counts().values
    v = freq / data.shape[0]
    return round(-sum(v * log2(v)), 3)

def req_bits(row, N):
    n = row.sum()
    vals = row.values / n
    vals = vals[nonzero(vals)]
    return -sum(vals * log2(vals)) * (n / N)

def info_x(data, attr, target):
    freq = pd.crosstab(data[attr], data[target], normalize=False)
    N = data.shape[0]
    info = freq.apply(lambda row : req_bits(row,N), axis=1)
    return round(info.sum(), 3)

def info_cont_x(data, attr, thrsh, target):
    thrsh_sort = data[attr].apply(lambda x: x >= thrsh)
    freq = pd.crosstab(thrsh_sort, data[target], normalize=False)
    N = data.shape[0]
    info = freq.apply(lambda row: req_bits(row, N), axis=1)
    return round(info.sum(), 3)

def set_threshold(values,curr_thrsh):
    """
    It's a strange point in every C4.5 implementation. We will continue the tradition
    """
    newThrsh = -max(values)
    for v in values:
        tempValue = v
        if (tempValue > newThrsh) & (tempValue <= curr_thrsh):
            newThrsh = tempValue
    return newThrsh
