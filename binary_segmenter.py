import numpy as np
from random import sample
import matplotlib.pyplot as plt
import sys
import networkx as nx
import math
import os

def display_mat(A, size, ue, mode="base"):
	os.makedirs("figures/"+str(size)+"_"+str(ue), exist_ok = True)
	dest = "figures/"+str(size)+"_"+str(ue)
	fig, ax = plt.subplots()
	im = ax.imshow(A, cmap="gray")
	fig.colorbar(im, orientation='vertical')
	plt.savefig(dest+"/"+mode+"_bin_PSD.pdf")
	plt.savefig(dest+"/"+mode+"_bin_PSD.png")
	return None

def make_Bs(size, start, dims):
	B = np.zeros((size, dims))
	split_point = np.random.randint(0, size)
	chosen_index = np.random.randint(start, start+size)
	B[0:split_point, chosen_index] = 1
	B[split_point:, chosen_index] = -1
	#print(B.shape)
	return B@B.T, B

#*************************** Hyperparameters **********************************#
size = int(sys.argv[1])
unique_eigvals = int(sys.argv[2])
#******************************************************************************#

#=========================== Get the eigenvalues ==============================#
sum_ev = size
while sum_ev >= size:
	eigvals = sample(range(0,int(1.8*size/unique_eigvals)), unique_eigvals)
	sum_ev = np.sum(eigvals)

eigvals = -np.sort(-np.array(eigvals))
#print(eigvals)
#==============================================================================#

#**************************** Fill in the matrix ******************************#
A = np.zeros((size, size))
start = 0
end = 0
B = np.zeros((size, size))
for index in range(len(eigvals)):
	#print(index, eigvals[index])
	end = end+eigvals[index]
	block, Bs = make_Bs(eigvals[index], start, size)
	#print("target block:", start, end)
	A[start:end, start:end] = block
	B[start:end, :] = Bs
	start=end
#******************************************************************************#

# checking
#print(np.sort((np.real(np.linalg.eigvals(A))).astype(int)))
display_mat(A, size, unique_eigvals)
display_mat(B, size, unique_eigvals, mode="B")

#======================== PERMUTE COLUMNS AND ROWS ============================#
index_perms = np.random.permutation(list(range(size)))

A1 = A[:, index_perms]
A1 = A1[index_perms, :]

B1 = B[index_perms, :]

#==============================================================================#
#print(A1.shape)
#print(np.sort((np.real(np.linalg.eigvals(A1))).astype(int)))
display_mat(A1, size, unique_eigvals, mode="perm")
display_mat(B1, size, unique_eigvals, mode="B1")

#==========================replace with 1s and see=============================#
A_with_ones = A
A_with_ones[A_with_ones<0] = 1
A1_with_ones = A1
A1_with_ones[A1_with_ones<0] = 1
B_with_ones = B
B_with_ones[B_with_ones<0] = 1
B1_with_ones = B1
B1_with_ones[B1_with_ones<0] = 1


display_mat(A_with_ones, size, unique_eigvals, mode="A_with_ones")
display_mat(A1_with_ones, size, unique_eigvals, mode="A1_with_ones")
display_mat(B_with_ones, size, unique_eigvals, mode="B_with_ones")
display_mat(B1_with_ones, size, unique_eigvals, mode="B1_with_ones")


#************************ Generate adjacency matrix ***************************#
# d = int(math.log(size, 2.0) / 0.5)
# print(d)
# G = nx.random_regular_graph(d, size)
# Ag = nx.adjacency_matrix(G)
# #display_mat(Ag.todense(), size, unique_eigvals, mode="adj")
# #******************************************************************************#

# #=========================== Conjugate Matrices ===============================#
# Aconj = np.multiply(A, Ag.todense())
# Aconj[Aconj < 0] = 1
# #display_mat(Aconj, size, unique_eigvals, mode="conj")
# #==============================================================================#

# # identity matrix #
# I = np.eye(size)
# display_mat(I, size, unique_eigvals, mode="eye")
