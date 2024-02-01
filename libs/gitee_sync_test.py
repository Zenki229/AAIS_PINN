import numpy as np
size = 10
node = np.random.rand(900).reshape((100,9))*2-1


def is_node_in(node):
    aux = np.full(node[:,0].shape, True)
    for i in range(9):
        aux = aux & (-0.9<node[:, i])&(node[:, i]<0.9)
    return aux
loc = is_node_in(node)
node_in = node[is_node_in(node)]
print(node_in.shape[0])
