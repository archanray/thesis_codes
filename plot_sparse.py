import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def data(name="identity", size = 100):
    if name == "identity":
        return np.eye(size)
    if name == "facebook":
        data_file = "./data/facebook_combined.txt"
        import networkx as nx
        g = nx.read_edgelist(data_file,create_using=nx.DiGraph(), nodetype = int)
        A = nx.adjacency_matrix(g)
        A = A.todense()
        A = np.asarray(A)
        A = (A+A.T)/2
        ids = range(len(A))
        ids = np.random.permutation(ids)
        ids = ids[0:size]
        A = A[ids][:, ids]
        A[A>0] = 1
        A[A<=0] = 0
        return A

def display_mat(A, name="identity"):
    save_filename = "figures/sparse_mats/"
    if not os.path.isdir(save_filename):
        os.makedirs(save_filename)
    save_filename += name+".pdf"

    fig, ax = plt.subplots(figsize=(10,10))
    ax.axis(False)
    ax.imshow(A, cmap=plt.cm.gray)
    plt.savefig(save_filename)
    return None


data_file = sys.argv[1]
display_mat(data(data_file), data_file)

