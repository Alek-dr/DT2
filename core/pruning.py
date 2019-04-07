from numpy import sqrt, square
from operator import itemgetter

def error(N,f,z):
    z2 = square(z)
    num = f + (z2/(2*N)) + z*sqrt((f/N) - (square(f)/N) + (z2/(4*square(N))))
    den = 1 + (z2/N)
    return num/den

def errorBasedPruning(tree, z=0.69):
    groups = tree.groupLeafByParent()
    for parent, childs in groups.items():
        if len(childs) > 1:
            errorRates = []
            for chId in childs:
                node = tree.getNode(chId)
                maj = node.attr
                N = node.stat.sum(axis=0)
                f = node.stat.drop(maj, axis=0).sum(axis=0)
                errorRates.append(error(N,f,z)/N)
            errorRates = list(filter(lambda x : x == x, errorRates))
            err = sum(errorRates)
            del errorRates
            if err < 0.51:
                # prune
                tree.prune(parent, childs)
                tree._pruneConnect_(parent, childs)
                tree._pruneBranchStat_(parent, childs)
