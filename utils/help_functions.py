def get_unique(labels):
    uniq = []
    for lbl in labels:
        if lbl not in uniq:
            uniq.append(lbl)
    return uniq


def groupSameElements(labels):
    uniqElements = get_unique(labels)
    if len(uniqElements)==1:
        return [[i for i in range(len(labels))]]
    else:
        grouped = []
        for lbl in uniqElements:
            grouped.append([i for i in range(len(labels)) if labels[i] == lbl])
        return grouped
