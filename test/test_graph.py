from core.graph import *

g = Graph()

root = Node(id=0)
n1 = Node(id=1)
n2 = Node(id=2)
n3 = Node(id=3)
n4 = Node(id=4)
n5 = Node(id=5)
n6 = Node(id=6)
n7 = Node(id=7)

g.setRootNode(root)
g.addNode(n1,0)
g.addNode(n2,1)
g.addNode(n3,1)

g.addNode(n4,0)
g.addNode(n5,0)

g.addNode(n6,5)
g.addNode(n7,5)

print(g.vertices)