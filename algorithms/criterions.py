from numpy import log2, sum, round, nonzero
import pandas as pd

def freq(data):
    pass

def info(data, target):
    freq = data[target].value_counts().values
    v = freq/data.shape[0]
    return round(-sum(v*log2(v)),3)

def info_x(data, attr, target):
    freq = pd.crosstab(data[attr], data[target], normalize=False)
    N = data.shape[0]

    def req_bits(row):
        n = row.sum()
        vals = row.values/n
        vals = vals[nonzero(vals)]
        return -sum(vals*log2(vals))*(n/N)

    info = freq.apply(req_bits, axis=1)
    return round(info.sum(),3)
