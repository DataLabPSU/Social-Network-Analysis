from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import math
from numpy.random import seed
from numpy.random import randn
import pandas as pd
import numpy as np
# plt.rcParams.update({'font.size': 10})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
import scipy as sp
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev
from scipy.optimize import minimize, curve_fit
from scipy.special import factorial
from scipy.stats import anderson, norm, pearsonr, wasserstein_distance
from scipy import stats
import scipy.stats as st
import scipy.stats
from sklearn.neighbors import KernelDensity
from statsmodels.graphics.gofplots import qqplot
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
# matplotlib.rcParams.update({'font.weight': 'bold'})
matplotlib.rcParams.update({'figure.autolayout': True})
zzz = []
zzz_pdf = []
zzzz = []
ZZZ = []
ZZZ_pdf = []
ZZZZ = []


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance, divergence

for q in range(0,100):
    q = q/100
    likemindedlistoflists = []
    threshold = q
    # foldername = "Node_altered_likemindedness_in_degree{}".format(threshold)
    # foldername = "Node_altered_likemindedness_out_degree{}".format(threshold)
    # foldername = "likemindedness_altered_in_degree{}".format(threshold)
    # foldername = "likemindedness_altered_in_degree_with_only_T{}".format(threshold)
    # foldername = "likemindedness_altered_in_degree_with_FA_and_F{}".format(threshold)
    foldername = "likemindedness_altered_out_degree{}".format(threshold)
    # foldername = "likemindedness_altered_out_degree_with_only_T{}".format(threshold)
    # foldername = "likemindedness_altered_out_degree_with_FA_and_F{}".format(threshold)

    with open(foldername + "/likemindedlistofliststde", "rb") as lmlol1:
        likemindedlistofliststde = pickle.load(lmlol1)
    # with open(foldername + "/likemindedlistofliststie", "rb") as lmlol2:
    #     likemindedlistofliststie = pickle.load(lmlol2)

    zzz.append([xx for xx in likemindedlistofliststde])
    # meanz = np.mean(zzz[-1])
    # zzzz.append([meanz for i in range(len(zzz[-1]))])
    # ZZZ.append([xxx[0] for xxx in likemindedlistofliststie])
    # meanZ = np.mean(ZZZ[-1])
    # ZZZZ.append([meanZ for i in range(len(ZZZ[-1]))])

zzz = np.asarray(zzz)
for elem in range(zzz.shape[1]):
    minn = min(zzz[1:,elem])
    maxx = max(zzz[1:,elem])
    for value in range(1,zzz.shape[0]):
        zzz[value][elem] = (zzz[value][elem] - minn) / (maxx - minn)

# ZZZ = np.asarray(ZZZ)
# for elem in range(ZZZ.shape[1]):
#     minn = min(ZZZ[:,elem])
#     maxx = max(ZZZ[:,elem])
#     for value in range(ZZZ.shape[0]):
#         ZZZ[value][elem] = (ZZZ[value][elem] - minn) / (maxx - minn)
# zzzz = np.asarray(zzzz)
# ZZZZ = np.asarray(ZZZZ)
x = np.arange(1,100)
plt.plot(x, zzz[1:,0], 'b-')
# plt.plot(x, 1-zzz[:,0], 'r-')
plt.show()
# avg_ie = []
# avg_de = []
# x = np.linspace(0, 101, 100)
# for j in range(zzzz.shape[0]):
#     avg_de.append(zzzz[j][0])
# bspl_de = splrep(x, avg_de, s=50)
# bspl_avg_de = splev(x, bspl_de)
# data_1 = avg_de
# bspl_avg_de = np.repeat(bspl_avg_de.reshape(bspl_avg_de.shape[0], 1), 143, axis=1)

# for jj in range(ZZZZ.shape[0]):
#     avg_ie.append(ZZZZ[jj][0])
# bspl_ie = splrep(x, avg_ie, s=700)
# bspl_avg_ie = splev(x, bspl_ie)
# data_2 = avg_ie
# bspl_avg_ie = np.repeat(bspl_avg_ie.reshape(bspl_avg_ie.shape[0], 1), 143, axis=1)

def power(k, lamb):
    """poisson pdf, parameter lamb is the fit parameter"""
    return (k**(-lamb))

def poisson(k, lamb):
    """poisson pdf, parameter lamb is the fit parameter"""
    return (lamb**k/factorial(k)) * np.exp(-lamb)

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / (2*wid**2))

def negLogLikelihood(params, data):
    """ the negative log-Likelohood-Function"""
    lnl = - np.sum(np.log(poisson(data, params[0])))
    # lnl = - np.sum(np.log(power(data, params[0])))
    # lnl = - np.sum(np.log(gaussian(data, params[0], params[1], params[2])))
    return lnl

# minimize the negative log-Likelihood

# result_1 = minimize(negLogLikelihood,  # function to minimize
#                   x0=np.ones(1),     # start value
#                   args=(data_1,),      # additional arguments for functionqqq
#                   method='BFGS',   # minimization method, see docs
#                   )

# result_2 = minimize(negLogLikelihood,  # function to minimize
#                   x0=np.ones(1),     # start value
#                   args=(data_2),      # additional arguments for function
#                   method='BFGS',   # minimization method, see docs
#                   )
# result is a scipy optimize result object, the fit parameters 
# are stored in result.x
# print (result_1.x)
# print (result_2.x)
# model_1 = poisson
# test_1 = np.random.poisson(result_1.x, 100)
# test_2 = np.random.poisson(result_2.x, 100)

def data_splitter(timestampoftimestamps, q):
    temps1 = []
    temps2 = []
    temps3 = []
    temps4 = []
    temps5 = []
    for i in range(len(timestampoftimestamps)):
        numberoftrues = 0
        numberoffakes = 0
        numberofshares = 0
        credibility = 0
        numberofnodes = 0
        J = len(timestampoftimestamps[i])
        for j in range(J):
            # numberofnodes += len(timestampoftimestamps[i][j])
            numberoftrues += timestampoftimestamps[i][j][0]
            numberoffakes += timestampoftimestamps[i][j][1]
            numberofshares += timestampoftimestamps[i][j][2]
            credibility += timestampoftimestamps[i][j][3]
        temps1.append(numberoftrues)
        temps2.append(numberoffakes)
        temps3.append(numberofshares)
        temps4.append(credibility) 
        # temps5.append(numberofnodes) 
        # print("numberoftrues: {}, numberoffakes: {}, numberofshares: {}, credibility: {}".format(numberoftrues, numberoffakes, numberofshares, credibility))
    # print("#############################End of {}th similarity metric timestamps###################################".format(q))
    return temps1, temps2, temps3, temps4


def main():
    number_of_trues = []
    number_of_fakes = []
    number_of_shares = []
    credibilities = []
    number_of_nodes = []
    zzzz = []
    ZZZZ = []
    for q in range(0,100):
        q = q/100
        threshold = q
        # foldername = "Node_altered_likemindedness_in_degree{}".format(threshold)
        # foldername = "Node_altered_likemindedness_out_degree{}".format(threshold)
        # foldername = "likemindedness_altered_in_degree{}".format(threshold)
        # foldername = "likemindedness_altered_in_degree_with_only_T{}".format(threshold)
        # foldername = "likemindedness_altered_in_degree_with_FA_and_F{}".format(threshold)
        foldername = "likemindedness_altered_out_degree{}".format(threshold)
        # foldername = "likemindedness_altered_out_degree_with_only_T{}".format(threshold)
        # foldername = "likemindedness_altered_out_degree_with_FA_and_F{}".format(threshold)

        with open(foldername + "/timestampoftimestamps", "rb") as tsots:
            timestampoftimestamps = pickle.load(tsots)
        
        numoft,numoff,numofs,cred = data_splitter(timestampoftimestamps, q)
        number_of_trues.append(numoft)
        number_of_fakes.append(numoff)
        number_of_shares.append(numofs)
        credibilities.append(cred)
        # number_of_nodes.append(non)
        meanz = np.mean(number_of_trues[-1])
        zzzz.append([meanz for i in range(len(number_of_trues[-1]))])
        meanzz = np.mean(number_of_fakes[-1])
        ZZZZ.append([meanzz for i in range(len(number_of_fakes[-1]))])

    X, Y = np.meshgrid(np.arange(len(number_of_trues[0])), np.arange(len(number_of_trues)))
    number_of_trues = np.asarray(number_of_trues)
    number_of_fakes = np.asarray(number_of_fakes)
    number_of_shares = np.asarray(number_of_shares)
    credibilities = np.asarray(credibilities)
    # number_of_nodes = np.asarray(number_of_nodes)
    zzzz = np.asarray(zzzz)
    ZZZZ = np.asarray(ZZZZ)
    for elem in range(number_of_trues.shape[1]):
        minn = min(number_of_trues[1:,elem])
        maxx = max(number_of_trues[1:,elem])
        # print (minn, maxx)
        for value in range(1,number_of_trues.shape[0]):
            number_of_trues[value][elem] = (number_of_trues[value][elem] - minn) / (maxx - minn)

    for elem in range(number_of_fakes.shape[1]):
        minn = min(number_of_fakes[1:,elem])
        maxx = max(number_of_fakes[1:,elem])
        # print (minn, maxx)
        for value in range(1,number_of_fakes.shape[0]):
            number_of_fakes[value][elem] = (number_of_fakes[value][elem] - minn) / (maxx - minn)

    # for elem in range(number_of_shares.shape[1]):
    #     minn = min(number_of_shares[:,elem])
    #     maxx = max(number_of_shares[:,elem])
    #     # print (minn, maxx)
    #     for value in range(number_of_shares.shape[0]):
    #         number_of_shares[value][elem] = (number_of_shares[value][elem] - minn) / (maxx - minn)

    # for elem in range(number_of_nodes.shape[1]):
    #     minn = min(number_of_nodes[:,elem])
    #     maxx = max(number_of_nodes[:,elem])
    #     # print (minn, maxx)
    #     for value in range(number_of_nodes.shape[0]):
    #         number_of_nodes[value][elem] = (number_of_nodes[value][elem] - minn) / (maxx - minn)

    return number_of_trues, number_of_fakes, number_of_shares, zzzz, ZZZZ

nott, noff, noss, zzzzz, ZZZZZ = main()

x = np.linspace(0, 100, 100)
y = np.linspace(0, 143, 143)
X, Y = np.meshgrid(np.arange(len(zzz[0])-1), np.arange(len(zzz)-1))
nott_pdf = []
noff_pdf = []
noss_pdf = []
# nonn_pdf = []

# zzzzzz = 1-zzz
# ZZZZZZ = 1-ZZZ
# nottt = 1-nott
# nofff = 1-noff
# nosss = 1-noss
# nonnn = 1-nonn

zzz_pdf = zzz
# ZZZ_pdf = 1-ZZZ
nott_pdf = nott
noff_pdf = noff
noss_pdf = noss
# nonn_pdf =nonn
# for akqr in range(ZZZZZZ.shape[0]):
#     plt.plot(ZZZZZZ[:,akqr], color='orange', linewidth=0.6)
# plt.show()

# zzz_pdf.append([zzzzzz[element+1]-zzzzzz[element] for element in range(zzzzzz.shape[0]-1)])
# zzz_pdf = np.asarray(zzz_pdf)

# ZZZ_pdf.append([ZZZZZZ[element+1]-ZZZZZZ[element] for element in range(ZZZZZZ.shape[0]-1)])
# ZZZ_pdf = np.asarray(ZZZ_pdf)

# nott_pdf.append([nottt[element+1]-nottt[element] for element in range(nottt.shape[0]-1)])
# nott_pdf = np.asarray(nott_pdf)

# noff_pdf.append([nofff[element+1]-nofff[element] for element in range(nofff.shape[0]-1)])
# noff_pdf = np.asarray(noff_pdf)

# noss_pdf.append([nosss[element+1]-nosss[element] for element in range(nosss.shape[0]-1)])
# noss_pdf = np.asarray(noss_pdf)

# nonn_pdf.append([nonnn[element+1]-nonnn[element] for element in range(nonnn.shape[0]-1)])
# nonn_pdf = np.asarray(nonn_pdf)

xxxx = np.linspace(1, 100, 99)

plt.figure()
plt.subplot(211)
for a in range(zzz_pdf.shape[1]):
    plt.plot(xxxx, zzz_pdf[1:,a], color='blue', linewidth=0.8)
    # plt.plot(xxxx, ZZZ_pdf[:,a], color='orange', linewidth=0.8)
    # neg_num=[ZZZ_pdf[:,a].tolist()[negnum] for negnum in range(len(ZZZ_pdf[:,a].tolist())-1) if ZZZ_pdf[:,a].tolist()[negnum] < 0.0]  
    # plt.scatter(xxxx, zzz_pdf[0][:,a], color='blue', linewidth=1)
    # plt.scatter(xxxx, ZZZ_pdf[0][:,a], color='orange', linewidth=1)
plt.legend(['Entropy'], loc='upper right')
plt.ylabel("Entropy")
plt.xlabel("Media Node-Edge Homogeneity")
plt.xlim(1,100)
plt.grid(True)

# plt.subplot(412)
# zzz_time = np.cumsum(zzz,axis=0)   
# # ZZZ_time = np.cumsum(ZZZ,axis=0)  
# for b in range(zzz.shape[1]):
#     plt.plot(x, zzz[:,b], linewidth=0.8, color='blue')
#     # plt.plot(x, ZZZ[:,b], linewidth=0.8, color='orange')
#     # plt.scatter(x, zzz[:,b], linewidth=1, color='blue')
#     # plt.scatter(x, ZZZ[:,b], linewidth=1, color='orange')
# plt.legend(['Complementary Cumulative Distribution of Dependent Entropy', 'Complementary Cumulative Distribution of Independent Entropy'], loc='upper right')
# plt.ylabel("Entropy")
# plt.grid(True)

plt.subplot(212)
for c in range(nott_pdf.shape[1]):
    plt.plot(xxxx, nott_pdf[1:,c], linewidth=0.8, color='green')
    plt.plot(xxxx, noff_pdf[1:,c], linewidth=0.8, color='red')
    # plt.plot(xxxx, noss_pdf[0][:,141], 'y-')
    # plt.plot(xxxx, nonn_pdf[:,c], linewidth=0.8, color='black')
# plt.legend(['Distribution of True Video Likes', 'Distribution of Fake Video Likes', 'Distribution of Video Shares', 'Distribution of Nodes'], loc='upper left')
plt.legend(['authentic information', 'misinformation'], loc='upper right')
plt.xlabel("Media Node-Edge Homogeneity")
plt.ylabel("Likes")
plt.xlim(1,100)
plt.grid(True)

# plt.subplot(414)
# for d in range(nott.shape[1]):
#     plt.plot(x, nott[:,d], linewidth=0.8, color='green')
#     plt.plot(x, noff[:,d], linewidth=0.8, color='red')
#     # plt.plot(x, noss[:,141], 'y-')
#     # plt.plot(x, nonn[:,d], linewidth=0.8, color='black')
# # plt.legend(['Complementary Cumulative Distribution of True Video Likes', 'Complementary Cumulative Distribution of Fake Video Likes', 'Complementary Cumulative Distribution of Video Shares', 'Complementary Cumulative Distribution of Nodes'], loc='upper right')
# plt.legend(['Complementary Cumulative Distribution of True Video Likes', 'Complementary Cumulative Distribution of Fake Video Likes', 'Complementary Cumulative Distribution of Nodes'], loc='upper right')
# plt.xlabel("Node-Edge Homogeneity")
# plt.ylabel("Value")
# plt.grid(True)
plt.show()
print (zzz_pdf[1:,:].shape)
XX, YY = np.meshgrid(np.arange(143), np.arange(99))
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.plot_surface(XX, YY, zzz_pdf[1:,:], color = 'blue')
ax.plot_wireframe(XX, YY, zzz_pdf[1:,:], color = 'blue')
ax.set_title("Entropy Distribution")
ax.set_xlabel('Timestamps')
ax.set_ylabel('Media Node-Edge Homogeneity')
ax.set_zlabel('Entropy')
ax.set_xlim(143,0)
ax.set_ylim(100,1)
plt.show()

fig1 = plt.figure(figsize=plt.figaspect(0.5))
ax = fig1.add_subplot(1, 2, 1, projection='3d')
ax.plot_wireframe(X, Y, nott_pdf[1:,:], color = 'green')
# ax.plot_surface(X, Y, nott_pdf[1:,:], color = 'green')
ax.set_title("Likes Distribution \n # of authentic information likes (green)")
ax.set_xlabel('Timestamps')
ax.set_ylabel('Media Node-Edge Homogeneity')
ax.set_zlabel('Likes')
ax.set_xlim(143,0)
ax.set_ylim(100,1)
ax.set_zlim(0, (max([max(sublist) for list_of_list in (nott.tolist(), noff.tolist()) for sublist in list_of_list])))
ax = fig1.add_subplot(1, 2, 2, projection='3d')
ax.plot_wireframe(X, Y, noff_pdf[1:,:], color = 'red')
# ax.plot_surface(X, Y, noff_pdf[1:,:], color = 'red')
ax.set_title("Likes Distribution \n # of misinformation likes (red)")
ax.set_xlabel('Timestamps')
ax.set_ylabel('Media Node-Edge Homogeneity')
ax.set_zlabel('Likes')
ax.set_xlim(143,0)
ax.set_ylim(100,1)
ax.set_zlim(0, (max([max(sublist) for list_of_list in (nott.tolist(), noff.tolist()) for sublist in list_of_list])))
plt.show()

XXXXX, YY = np.meshgrid(np.arange(142), np.arange(99))

# fig2 = plt.figure(figsize=plt.figaspect(0.5))
# ax = fig2.add_subplot(1, 2, 1, projection='3d')
# # ax.set_title("Probability Density Distribution \n# of True Video Likes (green)\n# of Fake Video Likes (red)")
# # ax.set_xlabel('Timestamps')
# # ax.set_ylabel('Node-Edge Homogeneity')
# # ax.set_zlabel('Value')
# # ax.set_xlim(143,0)
# # ax.set_ylim(0,100)
# # ax.set_zlim(0, (max([max(sublist) for list_of_list in (nott_pdf[0].tolist(), noff_pdf[0].tolist()) for sublist in list_of_list])))
# # ax.plot_wireframe(XXXXX, YY, nott_pdf, color = 'green', linewidths=0.5)

# # ax = fig2.add_subplot(1, 3, 2, projection='3d')
# # ax.plot_wireframe(XXXXX, YY, noff_pdf, color = 'red', linewidths=0.5)
# difference = abs(noff_pdf-nott_pdf) ##CDF of difference between the true and false media pdf
# print (difference.shape)
# copy = np.copy(zzz_pdf.T) ##Dependent Entropy CDF
# for abcd in range(zzz_pdf.shape[0]):
#     if abcd == 0:
#         copy[:,abcd] = zzz_pdf.T[:,abcd]
#     else:
#         copy[:,abcd] = abs(copy[:,abcd] - zzz_pdf.T[:,abcd-1])
# print (copy.shape)
# # ax.plot_surface(XXXXX, YY, nott_pdf, color = 'green', linewidths=0.5)
# # ax.plot_surface(XXXXX, YY, noff_pdf, color = 'red', linewidths=0.5)
# XX, YYY = np.meshgrid(np.arange(143), np.arange(100))

# copyy = np.copy(difference.T) ##CDF of difference between the true and false media pdf
# for abcd in range(difference.shape[0]):
#     if abcd == 0:
#         copyy[:,abcd] = difference.T[:,abcd]
#     else:
#         copyy[:,abcd] = abs(copyy[:,abcd] - difference.T[:,abcd-1])
# print (copyy.shape)
# ax.plot_wireframe(XXXXX, YY, copyy.T, color = 'blue', linewidths=0.5) ##Time Vs. Homogeneity ~ PDF of difference between the true and false media pdf
# ax.plot_wireframe(XX, YYY, copy.T, color = 'orange', linewidths=0.5) ##Time Vs. Homogeneity ~ PDF of dependent entropy

# # ax.plot_surface(XXXXX, YY, nonn_pdf, color = 'black', linewidths=0.5)
# XX, YY = np.meshgrid(np.arange(143), np.arange(100))

# ax = fig2.add_subplot(1, 2, 2, projection='3d')
# ax.plot_wireframe(XX, YYY, zzz_pdf, color = 'orange', linewidths=0.5)

# # ax.plot_wireframe(XX, YY, -noss_pdf[0], color = 'purple')
# # ax.plot_wireframe(X, Y, credibilities, color = 'blue')
# # ax.plot_wireframe(XX, YY, -nonn_pdf[0], color = 'black')

# ax.set_title("Probability Density Distribution \n# of True Video Likes (green)\n# of Fake Video Likes (red)")
# ax.set_xlabel('Timestamps')
# ax.set_ylabel('Node-Edge Homogeneity')
# ax.set_zlabel('Value')
# # ax.set_xlim(143,0)
# # ax.set_ylim(0,100)
# # ax.set_zlim(0, (max([max(sublist) for list_of_list in (nott_pdf[0].tolist(), noff_pdf[0].tolist()) for sublist in list_of_list])))

# # ax = fig2.add_subplot(1, 3, 3, projection='3d')
# # ax.set_title("Probability Density Distribution \n# of True Video Likes (green)\n# of Fake Video Likes (red)")
# # ax.set_xlabel('Timestamps')
# # ax.set_ylabel('Node-Edge Homogeneity')
# # ax.set_zlabel('Value')
# # ax.set_xlim(143,0)
# # ax.set_ylim(0,100)
# # ax.set_zlim(0, (max([max(sublist) for list_of_list in (nott_pdf[0].tolist(), noff_pdf[0].tolist()) for sublist in list_of_list])))
# plt.show()
# plt.figure(5)
# from scipy.stats import ks_2samp
# d = []
# dd = []
# ddd =[]
# dddd=[]
# ddddd=[]
# for xyz in range(zzz_pdf.shape[1]-1):
#     # plt.subplot(311)
#     # sns.distplot(nott[1:,xyz], hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':1.00})
#     # plt.subplot(312)
#     # sns.distplot(noff[1:,xyz], hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':1.00})

#     dis, div = jensen_shannon_distance(zzz_pdf[1:,xyz], nott_pdf[1:,xyz])
#     dd.append(div)
#     dis, div = jensen_shannon_distance(zzz_pdf[1:,xyz], noff_pdf[1:,xyz])
#     ddddd.append(div)

# np.save("Node_Out_T_Ent_True.npy", dd)
# np.save("Node_Out_T_Ent_False.npy", ddddd)
# np.save("Node_Out_T_nott.npy", nott_pdf)
# np.save("Node_Out_T_noff.npy", noff_pdf)
# np.save("Node_Out_T_DepEnt.npy", zzz_pdf)


# plt.subplot(313)
# plt.plot(dd, "g-", label="JS Divergence Entropy and True Media")
# plt.plot(ddddd, "r-", label="JS Divergence Entropy and False Media")

# plt.legend(loc="upper right")
plt.show()


# x = np.arange(-10, 10, 0.001)
# p = norm.pdf(x, 0, 2)
# q = norm.pdf(x, 2, 2)
# d = jensen_shannon_distance(p, q)
# print (d)
# for ijk in range(copy.shape[1]):
#     d = jensen_shannon_distance(copy.T[:,ijk+1], copyy.T[:,ijk])
#     print (d)
# difference = noff_pdf-nott_pdf
# print (difference.shape)
# fig3 = plt.figure(figsize=plt.figaspect(0.5))
# ax = fig3.add_subplot(1, 2, 1, projection='3d')
# ax.plot_wireframe(XXXXX, YY, difference, color = 'green', linewidths=0.5)

# ax = fig3.add_subplot(1, 2, 2, projection='3d')
# ax.plot_wireframe(XX, YY, ZZZ_pdf, color = 'red', linewidths=0.5)

# plt.show()
# for iqw in range(nott_pdf[0].shape[0]):
#     sns.distplot(nott_pdf[0][:,iqw])
# plt.show()
# plt.figure()

# plt.subplot(211)

# # for k in range(zzz_pdf[0].shape[0]):
# #     if (k == 0):
# #         kk = np.convolve(zzz_pdf[0][:,k], zzz_pdf[0][:,k+1])
# #     else:
# #         kk = np.convolve(kk, zzz_pdf[0][:,k+1])

# # for kkk in range(ZZZ_pdf[0].shape[0]):
# #     if (kkk == 0):
# #         kkkk = np.convolve(ZZZ_pdf[0][:,k], ZZZ_pdf[0][:,k+1])
# #     else:
# #         kkkk = np.convolve(kkkk, ZZZ_pdf[0][:,k+1])
# for k in range(nott_pdf[0].shape[0]):
#     if (k == 0):
#         kk = np.convolve(nott_pdf[0][:,k], nott_pdf[0][:,k+1])
#     else:
#         kk = np.convolve(kk, nott_pdf[0][:,k+1])

# for kkk in range(noff_pdf[0].shape[0]):
#     if (kkk == 0):
#         kkkk = np.convolve(noff_pdf[0][:,k], noff_pdf[0][:,k+1])
#     else:
#         kkkk = np.convolve(kkkk, noff_pdf[0][:,k+1])

# # sns.distplot(kk, hist = False, kde = True, norm_hist=True, color="g")
# for iqw in range(nott_pdf[0].shape[0]):
#     sns.distplot(nott_pdf[0][:,iqw])
# # sns.distplot(kk, hist = False, kde = True, norm_hist=True, 
# #                  kde_kws = {'shade': True, 'linewidth': 1.5, 'cumulative':False, 'cut':0, 'clip': (0,100), 'bw':0.1}, 
# #                   label = "Probability Distribution of True Video Likes", color = 'green')

# # sns.distplot(noff_pdf[0][:,141], hist = False, kde = True, norm_hist=True,
# #                  kde_kws = {'shade': True, 'linewidth': 1.5, 'cumulative':False, 'cut':0, 'clip': (0,100), 'bw':1}, 
# #                   label = "Probability Distribution of False Video Likes", color = 'red', axlabel = 'Number of Video Likes')

# # sns.distplot(noss_pdf[0][:,141], hist = False, kde = True, norm_hist=True,
# #                  kde_kws = {'shade': True, 'linewidth': 1.5, 'cumulative':False, 'cut':0, 'clip': (0,100), 'bw':1}, 
# #                   label = "Probability Distribution of Shares", color = 'yellow')

# # sns.distplot(nonn_pdf[0][:,141], hist = False, kde = True, norm_hist=True,
# #                  kde_kws = {'shade': True, 'linewidth': 1.5, 'cumulative':False, 'cut':0, 'clip': (0,100), 'bw':1}, 
# #                   label = "Probability Distribution of Nodes", color = 'black')
# # plt.xlim(0.0, 70.0)

# plt.subplot(212)
# for iqww in range(noff_pdf[0].shape[0]):
#     sns.distplot(noff_pdf[0][:,iqww])
# # sns.distplot(kkkk, hist = False, kde = True, norm_hist=True, color="r")

# # sns.distplot(kkkk, hist = False, kde = True, norm_hist=True,
# #                  kde_kws = {'shade': True, 'linewidth': 1.5, 'cumulative':False, 'cut':0, 'clip': (0,100), 'bw':0.1}, 
# #                   label = "Probability Distribution of Dependent Entropy", color = 'red')

# # sns.distplot(ZZZ_pdf[0][:,141], hist = False, kde = True, norm_hist=True,
# #                  kde_kws = {'shade': True, 'linewidth': 1.5, 'cumulative':False, 'cut':0, 'clip': (0,100), 'bw':0.01}, 
# #                   label = "Probability Distribution of Independent Entropy", color = 'orange', axlabel = 'Entropy')
# # plt.xlim(0.0, 4.0)

# dfnott = pd.Series(nott_pdf[0][:,141].T, name='nott')
# dfnoff = pd.Series(noff_pdf[0][:,141].T, name='noff')
# dfnoss = pd.Series(noss_pdf[0][:,141].T, name='noss')
# dfnonn = pd.Series(nonn_pdf[0][:,141].T, name='nonn')
# dfdepent = pd.Series(zzz_pdf[0][:,141].T, name='depent')
# dfindent = pd.Series(ZZZ_pdf[0][:,141].T, name='indent')
# df = [dfnott, dfnoff, dfnoss, dfnonn, dfdepent, dfindent]
# dff = pd.concat(df, axis=1)
# # dff.to_csv('node_out.csv', index=False)
# plt.show()

# X, Y = np.meshgrid(np.arange(len(zzz[0])), np.arange(len(zzz)))
# fig3 = plt.figure(figsize=plt.figaspect(0.5))

# ax = fig3.add_subplot(1, 2, 1, projection='3d')
# ax.plot_wireframe(X, Y, zzz, color = 'blue')
# ax.set_title("Independent Entropy (orange) Vs. Dependent Entropy (blue)")
# ax.set_xlabel('Timestamps')
# ax.set_ylabel('Node-Edge Homogeneity')
# ax.set_zlabel('Entropy')
# ax.set_xlim(143,0)
# ax.set_ylim(0,100)
# ax.set_zlim(0, (max([max(sublist) for list_of_list in (zzz.tolist(), ZZZ.tolist()) for sublist in list_of_list])))
# ax = fig3.add_subplot(1, 2, 2, projection='3d')
# ax.plot_wireframe(X, Y, ZZZ, color = 'orange')
# ax.set_title("Independent Entropy (orange) Vs. Dependent Entropy (blue)")
# ax.set_xlabel('Timestamps')
# ax.set_ylabel('Node-Edge Homogeneity')
# ax.set_zlabel('Entropy')
# ax.set_xlim(143,0)
# ax.set_ylim(0,100)
# ax.set_zlim(0, (max([max(sublist) for list_of_list in (zzz.tolist(), ZZZ.tolist()) for sublist in list_of_list])))
# # XX, YY = np.meshgrid(np.arange(143), np.arange(99))
# # fig4 = plt.figure()
# # ax = plt.axes(projection='3d')
# # ax.plot_wireframe(XX, YY, zzz_pdf[0], color = 'blue')
# # ax.plot_wireframe(XX, YY, ZZZ_pdf[0], color = 'orange')

# # ax.set_title("Independent Entropy (orange) Vs. Dependent Entropy (blue)")
# # ax.set_xlabel('Timestamps')
# # ax.set_ylabel('Node-Edge Homogeneity')
# # ax.set_zlabel('Entropy')

# plt.show()
# # init_vals = [7, 8, 7]
# # popt1, pcov1 = curve_fit(gaussian, x, td1.flatten(), p0=init_vals)
# # popt2, pcov2 = curve_fit(gaussian, x, td2.flatten(), p0=init_vals)
# # plt.plot(x, td1, 'b-')
# # plt.plot(x, gaussian(x, *popt1), 'b-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt1))
# # plt.plot(x, td2, 'b-')
# # plt.plot(x, gaussian(x, *popt2), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt2))
# # plt.legend()
# # plt.show()
