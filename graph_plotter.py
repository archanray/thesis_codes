import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

def define_fixed_graph():
    adj = np.zeros((10,10))
    # 5x5 block of ones
    L1 = np.ones((5,5))
    # 2x2 block of ones
    L2 = np.ones((2,2))
    adj[7,7] = 1
    adj[0:5, 0:5] = L1
    adj[5:7, 5:7] = L2
    return adj

def draw_net(G):
    from pylab import rcParams
    rcParams['figure.figsize'] = 14, 10
    pos = nx.spring_layout(G, scale=20, k=3/np.sqrt(G.order()))
    d = dict(G.degree)
    nx.draw(G, pos, node_color='lightblue', node_size=[d[k]*300 for k in d])
    dest = "figures/10_3/"
    plt.savefig(dest+"graph.pdf")
    plt.savefig(dest+"graph.png")


A = define_fixed_graph()

G = nx.from_numpy_matrix(A, create_using=nx.MultiGraph)

draw_net(G)