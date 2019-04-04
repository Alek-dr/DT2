from algorithms.learn_tree import Tree


# def make_node(node):
#     if node.type == 'inner':
#         label = '{}[label="Id = {}\nAttribute = {}", fillcolor="#ffffff"];\n'.format(node.id, node.id, node.attr)
#     else:
#         label = '{}[label="Id = {}\n{}", fillcolor="#ffffff"];\n'.format(node.id, node.id, node.attr)
#     return label

def make_node(node, writeId):
    if node.type == 'inner':
        if writeId:
            label = '{}[label="{}\n{}", fillcolor="#ffffff"];\n'.format(node.id, node.id, node.attr)
        else:
            label = '{}[label="{}", fillcolor="#ffffff"];\n'.format(node.id, node.attr)
    else:
        if writeId:
            label = '{}[label="{}\n{}", fillcolor="#ffffff"];\n'.format(node.id, node.id, node.attr)
        else:
            label = '{}[label="{}", fillcolor="#ffffff"];\n'.format(node.id, node.attr)
    return label


def make_connection(connect):
    nodes = tuple(connect.keys())[0]
    prop = list(connect.values())[0]
    conn = '{} -> {} [label="{}", fontsize=10]'.format(nodes[0], nodes[1], prop)
    return conn


def export2dot(name, tree, writeId=False):
    if isinstance(tree, Tree):
        dot_graph = 'digraph Tree { \n\tnode [shape=box, style="filled, rounded", color="black", fontname=helvetica] ; edge [fontname=helvetica];\n'
        graph_nodes, connections, ids = [], [], []
        # def get_connect(nodes) : return str(nodes[0]) + ' -> ' + str(nodes[1]) + [style=bold,label="100 times"]; ';\n'
        for connection in tree.connectionProp:
            nodes = tuple(connection.keys())[0]
            node = tree.getNode(nodes[0])
            if node.id not in ids:
                dot_graph += make_node(node, writeId)
                ids.append(node.id)
            node = tree.getNode(nodes[1])
            if node.id not in ids:
                dot_graph += make_node(node, writeId)
                ids.append(node.id)
            dot_graph += make_connection(connection)
        dot_graph += '}'

        with open(name + ".dot", "w") as output:
            output.write(dot_graph)
            output.close()
