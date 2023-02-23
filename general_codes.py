import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from scipy.stats import ortho_group
from matplotlib import cm as cm
from PIL import Image
import imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import copy
from tqdm import tqdm

# n = 10
# s = 5
# p = 0.2
# G = nx.erdos_renyi_graph(n, p, seed=s)
# LG = nx.line_graph(G)

# plt.figure(figsize=(10,6))

# nx.draw(G, node_color='blue', 
#         with_labels=False, 
#         node_size=100)
# plt.savefig("figures/general_plots/random_graph_50.pdf")

# plt.clf()
# plt.figure(figsize=(10,6))
# nx.draw(LG, node_color='red', 
#         with_labels=False, 
#         node_size=100)
# plt.savefig("figures/general_plots/LG_corr_to_random_graph_50.pdf")

######################################################################################

# A = np.random.random((20,20))
# A = A+A.T
# A = A/2
# A = A / np.max(A)

# plt.figure(figsize=(15,15))
# plt.imshow(A, cmap="Purples_r")
# plt.axis('off')
# plt.savefig("figures/general_plots/matrix.pdf")

# v1 = np.random.random((20,1))
# plt.figure(figsize=(3,15))
# plt.imshow(v1, cmap="magma")
# plt.axis('off')
# plt.savefig("figures/general_plots/v1.pdf")

# v2 = np.random.random((20,1))
# plt.figure(figsize=(3,15))
# plt.imshow(v2, cmap="magma")
# plt.axis('off')
# plt.savefig("figures/general_plots/v2.pdf")

# Av1 = A@v1
# Av2 = A@v2

# plt.figure(figsize=(3,15))
# plt.imshow(Av1, cmap="prism")
# plt.axis('off')
# plt.savefig("figures/general_plots/Av1.pdf")
# plt.figure(figsize=(3,15))
# plt.imshow(Av2, cmap="seismic")
# plt.axis('off')
# plt.savefig("figures/general_plots/Av2.pdf")

######################################################################################

# # random.seed(11)
# # np.random.seed(11)
# mu, sigma = 0, 0.05
# eigs = np.sort(np.random.normal(mu, sigma, 20))
# x = ortho_group.rvs(20)
# eigs = np.diag(eigs)
# A = x @ eigs @ x.T
# A = (A+A.T) / 2
# A = A / np.max(A)

# s = np.sort(random.sample(list(np.arange(0,20)), 15))
# sAs = A[s][:,s]

# L = -np.sort(-np.linalg.eigvals(A))
# Ls = np.linalg.eigvals(sAs)
# Ls = -np.sort(-np.array(list(Ls) + list(np.zeros(5))))
# xaxis = np.arange(1,21)

# fig, ax = plt.subplots()
# plt.scatter(xaxis, L, label="original eigenvalues", color=['#ff7f0e'])
# plt.scatter(xaxis, Ls, label="approx eigenvalues", color=['#1f77b4'])
# xtick_loc = np.arange(1,21,2)
# ax.set_xticks(xtick_loc)
# plt.xlabel("Eigenvalue indices")
# plt.ylabel("Eigenvalues")
# plt.legend()
# plt.savefig("figures/general_plots/eigenvalue_estimates.pdf")

######################################################################################

# # sublinear queries plot
# A = np.random.random((20,20))
# cmap = cm.get_cmap('YlGnBu')
# fig, ax = plt.subplots(figsize=(20,20))
# cax = ax.matshow(A, interpolation='nearest', cmap=cmap)
# ax.axis(False)
# plt.savefig("figures/general_plots/random_matrix.png", bbox_inches='tight')
# plt.clf()

# A = np.zeros((20,20))
# indices = np.random.choice(np.arange(A.size), replace=False, size=int(A.size * 0.1))
# A[np.unravel_index(indices, A.shape)] = 255
# cmap = cm.get_cmap('YlOrRd')
# fig, ax = plt.subplots(figsize=(20,20))
# cax = ax.matshow(A, cmap=cmap)
# ax.axis(False)
# plt.savefig("figures/general_plots/sample_random_matrix.png",bbox_inches='tight')
# plt.clf()

# # convert to background transparent
# img = Image.open("figures/general_plots/sample_random_matrix.png")
# img = img.convert("RGBA")
# datas = img.getdata()
# newData = []
# for item in datas:
#     if item[0] == 255 and item[1] == 255 and item[2] == 255:
#         newData.append((255, 255, 255, 0))
#     elif item[0] == 255 and item[1] == 255 and item[2] == 204:
#         newData.append((255, 255, 255, 0))
#     else: 
#         newData.append(item)
# img.putdata(newData)
# img.save("./figures/general_plots/sample_random_matrix.png", "PNG")

# # also add corresponding pixel colors in the initial image
# img1 = Image.open("figures/general_plots/random_matrix.png")
# img1 = img1.convert("RGBA")
# datas1 = img1.getdata()
# newData1 = []
# for i in range(len(datas)):
#     item = datas[i]
#     item2 = datas1[i]
#     if item[0] == 255 and item[1] == 255 and item[2] == 255:
#         newData1.append(item2)
#     else: 
#         newData1.append(item)
# img.putdata(newData1)
# img.save("./figures/general_plots/base_matrix.png", "PNG")

######################################################################################

# make GIF
def fadeout(fl1, fl2, factor):
        white = np.array([255,255,255,255], np.uint8)
        img1 = Image.open(fl2)
        img1 = img1.convert("RGBA")
        datas1 = img1.getdata()

        img2 = Image.open(fl1)
        img2 = img2.convert("RGBA")
        datas2 = img2.getdata()
        newData = []
        for i in range(len(datas1)):
            item1 = datas1[i]
            item2 = datas2[i]
            if item1[0] == 255 and item1[1] == 255 and item1[2] == 255:
                item2 = np.array(list(item2))
                vector = white - item2
                value = item2 + vector*factor
                value = value.astype(np.uint8)
                item2 = tuple(list(value))
                newData.append(item2)
            else: 
                newData.append(item1)
        img1.putdata(newData)
        return img1
    

images = []
file1 = "figures/general_plots/random_matrix.png"
file2 = "figures/general_plots/sample_random_matrix.png"
file3 = "figures/general_plots/base_matrix.png"
images.append(imageio.imread(file3))
# img = fadeout(file1, file2, 1/5)
# plt.imshow(img)
# plt.show()
le = 15
for i in range(le):
        images.append(fadeout(file1, file2, 1-1/(i+1)))

images.append(imageio.imread(file2))
for _ in range(4):
    images.append(imageio.imread("figures/general_plots/sample_random_matrix.png"))
imageio.mimsave('figures/general_plots/sublinear_sampler.gif', images)

######################################################################################

# # large entry in a matrix
# A = np.random.random((20,20))
# A = (A +A.T)/2
# A[7,13] = 3
# A[13,7] = 3
# cmap = cm.get_cmap('YlOrRd')
# fig, ax = plt.subplots(figsize=(20,20))
# im = ax.matshow(A, cmap=cmap)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# ax.axis(False)
# plt.colorbar(im, cax=cax)
# plt.savefig("figures/general_plots/large_entry_matrix.png",bbox_inches='tight')
# plt.clf()

######################################################################################

# cmap = cm.get_cmap('plasma')
# mu, sigma = 0, 0.05
# eigs = np.sort(np.random.normal(mu, sigma, 20))
# x = ortho_group.rvs(20)
# eigs = np.diag(eigs)
# A = x @ eigs @ x.T
# A = (A+A.T) / 2
# A = A / np.max(A)

# s = np.sort(random.sample(list(np.arange(0,20)), 15))
# sAs = A[s][:,s]

# L, V = np.linalg.eig(A)

# fig, ax = plt.subplots(figsize=(10,15))
# im = ax.matshow(A, cmap=cmap, vmin=-2, vmax=2)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# ax.axis(False)
# plt.colorbar(im, cax=cax)
# plt.savefig("figures/general_plots/matrix.pdf", bbox_inches='tight')
# plt.clf()

# fig, ax = plt.subplots(figsize=(10,15))
# im = ax.matshow(np.diag(L), cmap=cmap, vmin=-2, vmax=2)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# ax.axis(False)
# plt.colorbar(im, cax=cax)
# plt.savefig("figures/general_plots/lambda.pdf", bbox_inches='tight')
# plt.clf()

# fig, ax = plt.subplots(figsize=(10,15))
# im = ax.matshow(V, cmap=cmap, vmin=-2, vmax=2)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# ax.axis(False)
# plt.colorbar(im, cax=cax)
# plt.savefig("figures/general_plots/V.pdf", bbox_inches='tight')
# plt.clf()

# fig, ax = plt.subplots(figsize=(10,15))
# im = ax.matshow(V.T, cmap=cmap, vmin=-2, vmax=2)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# ax.axis(False)
# plt.colorbar(im, cax=cax)
# plt.savefig("figures/general_plots/VT.pdf", bbox_inches='tight')
# plt.clf()

# VoT = copy(V.T)
# Lo = copy(L)
# for i in range(len(L)):
#         if np.abs(L[i]) < 1.28940:
#                 Lo[i] = 0
#                 VoT[i,:] = np.zeros(20)
#         else:
#                 pass
#         pass
# Lm = L - Lo
# VmT = V.T - VoT

# fig, ax = plt.subplots(figsize=(10,15))
# im = ax.matshow(np.diag(Lo), cmap=cmap, vmin=-2, vmax=2)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# ax.axis(False)
# plt.colorbar(im, cax=cax)
# plt.savefig("figures/general_plots/Lo.pdf", bbox_inches='tight')
# plt.clf()

# fig, ax = plt.subplots(figsize=(10,15))
# im = ax.matshow(np.diag(Lm), cmap=cmap, vmin=-2, vmax=2)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# ax.axis(False)
# plt.colorbar(im, cax=cax)
# plt.savefig("figures/general_plots/Lm.pdf", bbox_inches='tight')
# plt.clf()

# fig, ax = plt.subplots(figsize=(10,15))
# im = ax.matshow(VoT.T, cmap=cmap, vmin=-2, vmax=2)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# ax.axis(False)
# plt.colorbar(im, cax=cax)
# plt.savefig("figures/general_plots/Vo.pdf", bbox_inches='tight')
# plt.clf()

# fig, ax = plt.subplots(figsize=(10,15))
# im = ax.matshow(VmT.T, cmap=cmap, vmin=-2, vmax=2)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# ax.axis(False)
# plt.colorbar(im, cax=cax)
# plt.savefig("figures/general_plots/Vm.pdf", bbox_inches='tight')
# plt.clf()


# sumVo = np.linalg.norm(VoT.T, axis=1)
# sumVo = sumVo**2
# xaxis = np.array(list(range(1,21)))
# fig, ax = plt.subplots(figsize=(10,10))
# im = ax.scatter(xaxis, sumVo, cmap=cmap)
# plt.ylim([0,1])
# ax.axis(True)
# plt.savefig("figures/general_plots/rowNormVo.pdf", bbox_inches='tight')
# plt.clf()