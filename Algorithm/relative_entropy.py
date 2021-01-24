import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import wasserstein_distance, ks_2samp, multivariate_normal
from scipy import linalg, stats
from sklearn import mixture
import itertools
import seaborn as sns
from scipy import signal
import scipy
import math
import pickle
from sklearn.preprocessing import normalize
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
# matplotlib.rcParams.update({'font.weight': 'bold'})
matplotlib.rcParams.update({'figure.autolayout': True})
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

np.random.seed(0)
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])

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

def plot_results(X, Y, means, covariances, index, title, nott_pdf_Media_Out, noff_pdf_Media_Out, kk, indexxx):
    average_original = []
    average_gaussian = []
    # plt.figure(5)
    meanu = []
    covaru = []
    DEP = []
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        # print("{}:{}".format(i,mean))
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        meanu.append(mean[0])
        covaru.append(covar[0][0])

        XXX = np.where(Y == i)
        XXX = list(XXX)

        plt.subplot(212)
        Dependent_Entropy_Distributions = []
        for j in range(X[Y == i].shape[0]):
            Dependent_Entropy_Distributions.append(X[Y == i][j][0])
        
        if i == 0:
            yyyy_original = noff_pdf_Media_Out[1:,indexxx][XXX[0]].tolist()
        else:
            yyyy_original = nott_pdf_Media_Out[1:,indexxx][XXX[0]].tolist()
        yyyy_original = np.asarray(yyyy_original)

        Dependent_Entropy_Distributions = np.asarray(Dependent_Entropy_Distributions)
        DEP.append(Dependent_Entropy_Distributions)
        dis_orig, div_orig = jensen_shannon_distance(yyyy_original, Dependent_Entropy_Distributions)
        average_original.append(div_orig)
        if i == 0:
            sns.distplot(Dependent_Entropy_Distributions, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':1.0}, color = "red")
        else:
            sns.distplot(Dependent_Entropy_Distributions, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':1.0}, color = "green")
            
        plt.title("Predicted by Gaussian Model")
        plt.ylabel("Density")
        plt.xlabel("Likes")
        plt.legend(loc="upper right")

        # plt.xlim(0.0,1.0)
        plt.xlim(0.0,300.0)

        # plt.ylim(0,10)

        # plt.subplot(313)
        # Resampled_Dependent_Entropy_Distributions = np.random.normal(mean, covar[0], 99)
        # X = X.tolist()
        # X_prime = []
        # for ijk in XXX[0]:
        #     X_prime.append(X[ijk])
        # X = np.asarray(X) 
        # print (X.shape)
        # dis_gaus, div_gaus = jensen_shannon_distance(X, Resampled_Dependent_Entropy_Distributions)
        # average_gaussian.append(div_gaus)
        # sns.distplot(Resampled_Dependent_Entropy_Distributions, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':1.0}, color = color)
        # plt.title("Re-sampled using parameters from Gaussian Model")
        # plt.xlim(0.0,300.0)
        # # plt.ylim(0,10)
        # plt.ylabel("Density")
        # plt.xlabel("Likes")
    
    # dis_actual, div_actual = jensen_shannon_distance(np.asarray(X.tolist()),kk)
    # plt.subplot(413)
    # sns.distplot(kk, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, color = color)
    # plt.title("Actual total media distribution")
    # plt.xlim(-0.2,0.6)
    # plt.ylim(0,10)

    # x = np.random.normal(meanu[1]-meanu[0],covaru[1]+covaru[0],100) 
    # dis_gaus_all, div_gaus_all = jensen_shannon_distance(np.asarray(X.tolist()),x)
    # plt.subplot(414)
    # sns.distplot(x, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, color = color)
    # plt.title("Re-sampled total media distribution")
    # plt.xlim(-0.2,0.6)
    # plt.ylim(0,10)
    
    # return average_original, average_gaussian, div_actual, div_gaus_all
    return average_original, average_gaussian, DEP

nott_pdf_Media_Outttt = np.load("Node_In_nott.npy")
noff_pdf_Media_Outttt = np.load("Node_In_noff.npy")
DepEnt_pdf_Media_Outttt = np.load("Node_In_DepEnt.npy")

nott_pdf_Media_Outttt_FA = np.load("Node_In_FA_F_nott.npy")
noff_pdf_Media_Outttt_FA = np.load("Node_In_FA_F_noff.npy")
DepEnt_pdf_Media_Outttt_FA = np.load("Node_In_FA_F_DepEnt.npy")

nott_pdf_Media_Outttt_T = np.load("Node_In_T_nott.npy")
noff_pdf_Media_Outttt_T = np.load("Node_In_T_noff.npy")
DepEnt_pdf_Media_Outttt_T = np.load("Node_In_T_DepEnt.npy")

nott_pdf_Media_Out = np.load("Node_Out_nott.npy")
noff_pdf_Media_Out = np.load("Node_Out_noff.npy")
DepEnt_pdf_Media_Out = np.load("Node_Out_DepEnt.npy")

nott_pdf_Media_Out_FA = np.load("Node_Out_FA_F_nott.npy")
noff_pdf_Media_Out_FA = np.load("Node_Out_FA_F_noff.npy")
DepEnt_pdf_Media_Out_FA = np.load("Node_Out_FA_F_DepEnt.npy")

nott_pdf_Media_Out_T = np.load("Node_Out_T_nott.npy")
noff_pdf_Media_Out_T = np.load("Node_Out_T_noff.npy")
DepEnt_pdf_Media_Out_T = np.load("Node_Out_T_DepEnt.npy")

# copy = np.copy(nott_pdf_Node.T)
# for abcd in range(nott_pdf_Node.shape[0]):
#     if abcd == 0:
#         copy[:,abcd] = nott_pdf_Node.T[:,abcd]
#     else:
#         copy[:,abcd] = copy[:,abcd] - nott_pdf_Node.T[:,abcd-1]
# nott_pdf_Node = copy.T

# copy = np.copy(noff_pdf_Node.T)
# for abcd in range(noff_pdf_Node.shape[0]):
#     if abcd == 0:
#         copy[:,abcd] = noff_pdf_Node.T[:,abcd]
#     else:
#         copy[:,abcd] = copy[:,abcd] - noff_pdf_Node.T[:,abcd-1]
# noff_pdf_Node = copy.T

# copy = np.copy(IndEnt_pdf_Node.T)
# for abcd in range(IndEnt_pdf_Node.shape[0]):
#     if abcd == 0:
#         copy[:,abcd] = IndEnt_pdf_Node.T[:,abcd]
#     else:
#         copy[:,abcd] = copy[:,abcd] - IndEnt_pdf_Node.T[:,abcd-1]
# IndEnt_pdf_Node = copy.T

# copy = np.copy(DepEnt_pdf_Node.T)
# for abcd in range(DepEnt_pdf_Node.shape[0]):
#     if abcd == 0:
#         copy[:,abcd] = DepEnt_pdf_Node.T[:,abcd]
#     else:
#         copy[:,abcd] = copy[:,abcd] - DepEnt_pdf_Node.T[:,abcd-1]
# DepEnt_pdf_Node = copy.T
# ############################################################################################
# copy = np.copy(nott_pdf_Media_Out.T)
# for abcd in range(nott_pdf_Media_Out.shape[0]):
#     if abcd == 0:
#         copy[:,abcd] = nott_pdf_Media_Out.T[:,abcd]
#     else:
#         copy[:,abcd] = copy[:,abcd] - nott_pdf_Media_Out.T[:,abcd-1]
# nott_pdf_Media_Out = copy.T

# copy = np.copy(noff_pdf_Media_Out.T)
# for abcd in range(noff_pdf_Media_Out.shape[0]):
#     if abcd == 0:
#         copy[:,abcd] = noff_pdf_Media_Out.T[:,abcd]
#     else:
#         copy[:,abcd] = copy[:,abcd] - noff_pdf_Media_Out.T[:,abcd-1]
# noff_pdf_Media_Out = copy.T

# copy = np.copy(IndEnt_pdf_Media_Out.T)
# for abcd in range(IndEnt_pdf_Media_Out.shape[0]):
#     if abcd == 0:
#         copy[:,abcd] = IndEnt_pdf_Media_Out.T[:,abcd]
#     else:
#         copy[:,abcd] = copy[:,abcd] - IndEnt_pdf_Media_Out.T[:,abcd-1]
# IndEnt_pdf_Media_Out = copy.T

# copy = np.copy(DepEnt_pdf_Media_Out.T)
# for abcd in range(DepEnt_pdf_Media_Out.shape[0]):
#     if abcd == 0:
#         copy[:,abcd] = DepEnt_pdf_Media_Out.T[:,abcd]
#     else:
#         copy[:,abcd] = copy[:,abcd] - DepEnt_pdf_Media_Out.T[:,abcd-1]
# DepEnt_pdf_Media_Out = copy.T
# ############################################################################################
# copy = np.copy(nott_pdf_Media_Out_FA.T)
# for abcd in range(nott_pdf_Media_Out_FA.shape[0]):
#     if abcd == 0:
#         copy[:,abcd] = nott_pdf_Media_Out_FA.T[:,abcd]
#     else:
#         copy[:,abcd] = copy[:,abcd] - nott_pdf_Media_Out_FA.T[:,abcd-1]
# nott_pdf_Media_Out_FA = copy.T

# copy = np.copy(noff_pdf_Media_Out_FA.T)
# for abcd in range(noff_pdf_Media_Out_FA.shape[0]):
#     if abcd == 0:
#         copy[:,abcd] = noff_pdf_Media_Out_FA.T[:,abcd]
#     else:
#         copy[:,abcd] = copy[:,abcd] - noff_pdf_Media_Out_FA.T[:,abcd-1]
# noff_pdf_Media_Out_FA = copy.T

# copy = np.copy(IndEnt_pdf_Media_Out_FA.T)
# for abcd in range(IndEnt_pdf_Media_Out_FA.shape[0]):
#     if abcd == 0:
#         copy[:,abcd] = IndEnt_pdf_Media_Out_FA.T[:,abcd]
#     else:
#         copy[:,abcd] = copy[:,abcd] - IndEnt_pdf_Media_Out_FA.T[:,abcd-1]
# IndEnt_pdf_Media_Out_FA = copy.T

# copy = np.copy(DepEnt_pdf_Media_Out_FA.T)
# for abcd in range(DepEnt_pdf_Media_Out_FA.shape[0]):
#     if abcd == 0:
#         copy[:,abcd] = DepEnt_pdf_Media_Out_FA.T[:,abcd]
#     else:
#         copy[:,abcd] = copy[:,abcd] - DepEnt_pdf_Media_Out_FA.T[:,abcd-1]
# DepEnt_pdf_Media_Out_FA = copy.T

# plt.figure(1)
# plt.subplot(211)
# plt.plot(dddd_Node, "r-", label="JS Distance_Node_Hom")
# plt.plot(dddd_Media_Out, "g-", label="JS Distance_Media_Hom")
# plt.plot(dddd_Media_Out_FA, "b-", label="JS Distance_Media_Hom_FA")
# plt.legend(loc="upper right")

# plt.subplot(212)
# plt.plot(ddddd_Node, "r-", label="JS Divergence_Node_Hom")
# plt.plot(ddddd_Media_Out, "g-", label="JS Divergence_Media_Hom")
# plt.plot(ddddd_Media_Out_FA, "b-", label="JS Divergence_Media_Hom_FA")
# plt.legend(loc="upper right")

d = []
dd = []
ddd = []
dddd = []
ddddd = []
dddddd = []

for xyz in range(1,nott_pdf_Media_Out.shape[1]-1):
    try:
        # DepEnt_pdf_Media_Outt = normalize(DepEnt_pdf_Media_Out[1:,xyz].reshape(-1,1), norm='max', axis=0)
        # DepEnt_pdf_Media_Outt_T = normalize(DepEnt_pdf_Media_Out_T[1:,xyz].reshape(-1,1), norm='max', axis=0)
        # DepEnt_pdf_Media_Outt_FA = normalize(DepEnt_pdf_Media_Out_FA[1:,xyz].reshape(-1,1), norm='max', axis=0)
        # nott_pdf_Media_Outt = normalize(nott_pdf_Media_Out[1:,xyz].reshape(-1,1), norm='max', axis=0)
        # noff_pdf_Media_Outt = normalize(noff_pdf_Media_Out[1:,xyz].reshape(-1,1), norm='max', axis=0)
        # nott_pdf_Media_Outt_T = normalize(nott_pdf_Media_Out_T[1:,xyz].reshape(-1,1), norm='max', axis=0)
        # noff_pdf_Media_Outt_T = normalize(noff_pdf_Media_Out_T[1:,xyz].reshape(-1,1), norm='max', axis=0)
        # nott_pdf_Media_Outt_FA = normalize(nott_pdf_Media_Out_FA[1:,xyz].reshape(-1,1), norm='max', axis=0)
        # noff_pdf_Media_Outt_FA = normalize(noff_pdf_Media_Out_FA[1:,xyz].reshape(-1,1), norm='max', axis=0)

        # DepEnt_pdf_Media_Outttt = normalize(DepEnt_pdf_Media_Outttt[1:,xyz].reshape(-1,1), norm='max', axis=0)
        # DepEnt_pdf_Media_Outttt_T = normalize(DepEnt_pdf_Media_Outttt_T[1:,xyz].reshape(-1,1), norm='max', axis=0)
        # DepEnt_pdf_Media_Outttt_FA = normalize(DepEnt_pdf_Media_Outttt_FA[1:,xyz].reshape(-1,1), norm='max', axis=0)
        # nott_pdf_Media_Outttt = normalize(nott_pdf_Media_Outttt[1:,xyz].reshape(-1,1), norm='max', axis=0)
        # noff_pdf_Media_Outttt = normalize(noff_pdf_Media_Outttt[1:,xyz].reshape(-1,1), norm='max', axis=0)
        # nott_pdf_Media_Outttt_T = normalize(nott_pdf_Media_Outttt_T[1:,xyz].reshape(-1,1), norm='max', axis=0)
        # noff_pdf_Media_Outttt_T = normalize(noff_pdf_Media_Outttt_T[1:,xyz].reshape(-1,1), norm='max', axis=0)
        # nott_pdf_Media_Outttt_FA = normalize(nott_pdf_Media_Outttt_FA[1:,xyz].reshape(-1,1), norm='max', axis=0)
        # noff_pdf_Media_Outttt_FA = normalize(noff_pdf_Media_Outttt_FA[1:,xyz].reshape(-1,1), norm='max', axis=0)

        dis, div = jensen_shannon_distance(DepEnt_pdf_Media_Outttt_T[1:,xyz], nott_pdf_Media_Outttt_T[1:,xyz])
        d.append(div)

        dis, div = jensen_shannon_distance(DepEnt_pdf_Media_Outttt[1:,xyz], nott_pdf_Media_Outttt[1:,xyz])
        dd.append(div)
        # print ("{}".format(ks_2samp(DepEnt_pdf_Media_Outttt.ravel(), nott_pdf_Media_Outttt.ravel())))
        dis, div = jensen_shannon_distance(DepEnt_pdf_Media_Outttt[1:,xyz], noff_pdf_Media_Outttt[1:,xyz])
        ddd.append(div)
        # print ("{}".format(ks_2samp(DepEnt_pdf_Media_Outttt.ravel(), noff_pdf_Media_Outttt.ravel())))

        dis, div = jensen_shannon_distance(DepEnt_pdf_Media_Outttt_FA[1:,xyz], noff_pdf_Media_Outttt_FA[1:,xyz])
        dddddd.append(div)
    except IndexError:
        d.append(0.0)
        dd.append(0.0)
        ddd.append(0.0)
        dddddd.append(0.0)
        print ("i was here")

plt.figure(2)
plt.subplot(311)
plt.plot(d, "g-", label="Jensen-Shannon divergence between entropy and authentic information \n (Users who liked only authentic information)")
# plt.plot(dddd, "r-", label="Entropy Media True VS. Noff_Media_Out_True")
plt.xlabel("Timestamps")
plt.ylabel("JS Divergence")
plt.legend(loc="upper right")

plt.subplot(312)
# plt.plot(ddddd, "g-", label="Entropy Media FA VS. Nott_Media_Out_FA")
plt.plot(dddddd, "r-", label="Jensen-Shannon divergence between entropy and misinformation \n (Users who liked only misinformation)")
plt.xlabel("Timestamps")
plt.ylabel("JS Divergence")
plt.legend(loc="upper right")

plt.subplot(313)
plt.plot(dd, "g-", label="Jensen-Shannon divergence between entropy and authentic information")
plt.plot(ddd, "r-", label="Jensen-Shannon divergence between entropy and misinformation")
plt.xlabel("Timestamps")
plt.ylabel("JS Divergence")
plt.legend(loc="upper right")

plt.show()
plt.figure(3)
plt.subplot(311)
sns.distplot(d, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "I(users who liked authentic information only)", color = 'green')
# sns.distplot(dddd, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "Information gain distribution from entropy of users who have only liked False Media", color = 'red')
plt.xlim(0.0,1.0)
# plt.ylim(0,10)
plt.xlabel("Information Gain")
plt.ylabel("Density")
plt.legend(loc="upper right")


plt.subplot(312)
# sns.distplot(ddddd, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "Information gain from True Video Distribution between metrics", color = 'green')
sns.distplot(dddddd, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "I(users who liked misinformation only)", color = 'red')
plt.xlim(0.0,1.0)
# plt.ylim(0,10)
plt.xlabel("Information Gain")
plt.ylabel("Density")
plt.legend(loc="upper right")


plt.subplot(313)
sns.distplot(dd, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "I(all users)", color = 'green')
sns.distplot(ddd, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "I(all users)", color = 'red')
plt.xlim(0.0,1.0)
# plt.ylim(0,10)
plt.xlabel("Information Gain")
plt.ylabel("Density")
plt.legend(loc="upper right")


kk = np.zeros((nott_pdf_Media_Outttt.shape[0],nott_pdf_Media_Outttt.shape[1]))
for k in range(nott_pdf_Media_Outttt.shape[1]):
    conv_pmf = signal.fftconvolve(nott_pdf_Media_Outttt[:,k], noff_pdf_Media_Outttt[:,k],'same')
    kk[:,k] = conv_pmf/sum(conv_pmf)

plt.figure(4)
plt.subplot(211)
for qwerty in range(1,142):
    sns.distplot(nott_pdf_Media_Outttt[1:,qwerty], hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, color = 'green')
    sns.distplot(noff_pdf_Media_Outttt[1:,qwerty], hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, color = 'red')
plt.xlim(0.0,300.0)
# plt.ylim(0,10)
plt.title("From Actual Data")
plt.xlabel("Likes")
plt.ylabel("Density")
plt.legend(loc="upper right")

ABCD_orig = []
ABCD_gaus = []
ABCD_actual = []
ABCD_gaus_all = []
DEPP = []
for abcdd in range(1,142):
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full', max_iter=100).fit(DepEnt_pdf_Media_Outttt[1:,abcdd].reshape(-1,1))
    # orig, gaus, actual, gaus_all = plot_results(DepEnt_pdf_Media_Out[:,abcdd].reshape(-1,1), gmm.predict(DepEnt_pdf_Media_Out[:,abcdd].reshape(-1,1)), gmm.means_, gmm.covariances_, 0, 'Expectation-maximization', nott_pdf_Media_Out, noff_pdf_Media_Out, kk[:,abcdd], abcdd)
    orig, gaus, DEP = plot_results(DepEnt_pdf_Media_Outttt[1:,abcdd].reshape(-1,1), gmm.predict(DepEnt_pdf_Media_Outttt[1:,abcdd].reshape(-1,1)), gmm.means_, gmm.covariances_, 0, 'Expectation-maximization', nott_pdf_Media_Outttt, noff_pdf_Media_Outttt, kk[1:,abcdd], abcdd)

    ABCD_orig.append(orig)
    ABCD_gaus.append(gaus)
    DEPP.append(DEP)
    # ABCD_actual.append(actual)
    # ABCD_gaus_all.append(gaus_all)

ABCDD_orig = [ABCD_orig[jj][0] for jj in range(len(ABCD_orig))]
ABCDDD_orig = [ABCD_orig[jj][1] for jj in range(len(ABCD_orig))]
ABCDD_orig = [x for x in ABCDD_orig if ~np.isnan(x)]
ABCDDD_orig = [y for y in ABCDDD_orig if ~np.isnan(y)]
ABCDD_orig = [xx for xx in ABCDD_orig if ~np.isinf(xx)]
ABCDDD_orig = [yy for yy in ABCDDD_orig if ~np.isinf(yy)]


# ABCDD_gaus = [ABCD_gaus[jj][0] for jj in range(len(ABCD_gaus))]
# ABCDDD_gaus = [ABCD_gaus[jj][1] for jj in range(len(ABCD_gaus))]
# for b in range(len(ABCDD_gaus)):
#     ABCDD_gaus[b] = [x for x in ABCDD_gaus[b] if ~np.isnan(x)]
#     ABCDDD_gaus[b] = [y for y in ABCDDD_gaus[b] if ~np.isnan(y)]
#     ABCDD_gaus[b] = [xx for xx in ABCDD_gaus[b] if ~np.isinf(xx)]
#     ABCDDD_gaus[b] = [yy for yy in ABCDDD_gaus[b] if ~np.isinf(yy)]

# ABCDD_actual = [ABCD_actual[jj] for jj in range(len(ABCD_actual))]
# for b in range(len(ABCDD_actual)):
#     ABCDD_actual[b] = [x for x in ABCDD_actual[b] if ~np.isnan(x)]
#     ABCDD_actual[b] = [xx for xx in ABCDD_actual[b] if ~np.isinf(xx)]

# ABCDD_gaus_all = [ABCD_gaus_all[jj] for jj in range(len(ABCD_gaus_all))]
# for b in range(len(ABCDD_gaus_all)):
#     ABCDD_gaus_all[b] = [x for x in ABCDD_gaus_all[b] if ~np.isnan(x)]
#     ABCDD_gaus_all[b] = [xx for xx in ABCDD_gaus_all[b] if ~np.isinf(xx)]

plt.figure(6)
plt.subplot(411)
sns.distplot(d, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "I(users who liked authentic information only)", color = 'green')
# sns.distplot(dddd, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "Information gain distribution from entropy of users who have only liked False Media", color = 'red')
plt.xlim(0.0,1.0)
plt.xlabel("Information Gain")
plt.ylabel("Density")
plt.legend(loc="upper right")


plt.subplot(412)
# sns.distplot(ddddd, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "Information gain from True Video Distribution between metrics", color = 'green')
sns.distplot(dddddd, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "I(users who liked misinformation only)", color = 'red')
plt.xlim(0.0,1.0)
plt.xlabel("Information Gain")
plt.ylabel("Density")
plt.legend(loc="upper right")


plt.subplot(413)
sns.distplot(dd, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "I(all users)", color = 'green')
sns.distplot(ddd, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "I(all users)", color = 'red')
plt.xlim(0.0,1.0)
# plt.ylim(0,10)
plt.xlabel("Information Gain")
plt.ylabel("Density")
plt.legend(loc="upper right")

plt.subplot(414)
sns.distplot(ABCDD_orig, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label="I'(all users) [I(A)] from GMM", color = 'green')
sns.distplot(ABCDDD_orig, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label="I'(all users) [I(B)] from GMM", color = 'red')
plt.xlim(0.0,1.0)
# plt.ylim(0,10)
plt.xlabel("Information Gain")
plt.ylabel("Density")
plt.legend(loc="upper right")
# plt.ylim(0,10)
# plt.subplot(313)
# # for b in range(len(ABCDD_gaus)):
# sns.distplot(ABCDD_gaus, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, color = 'red')
# sns.distplot(ABCDDD_gaus, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, color = 'green')
# plt.ylim(0,10)
# plt.xlim(-0.2,2.0)
# plt.subplot(514)
# # for b in range(len(ABCDD_actual)):
# #     sns.distplot(ABCDD_actual[b], hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, color = 'brown')
# # plt.ylim(0,10)
# # plt.xlim(-0.2,2.0)
# plt.subplot(515)
# for b in range(len(ABCDD_gaus_all)):
#     sns.distplot(ABCDD_gaus_all[b], hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, color = 'brown')
# plt.ylim(0,10)
# plt.xlim(-0.2,2.0)

# for lmn in range(len(ABCDD_gaus)):
#     for opq in range(len(ABCDD_gaus[lmn])):
#         print ([opq])

plt.figure(7)
plt.subplot(211)
plt.plot(dd, "g-", label="Information Gain from True Media Distribution")
plt.plot(ddd, "r-", label="Information Gain from False Media Distribution")
plt.ylim(0,0.5)
plt.xlim(0,142)
plt.xlabel("Timestamps")
plt.ylabel("Information Gain")
plt.subplot(212)
plt.plot(ABCDD_orig, "g-", label="Information Gain from GMM assigned True Media Distribution")
plt.plot(ABCDDD_orig, "r-", label="Information Gain from GMM assigned False Media Distribution")
plt.ylim(0,0.5)
plt.xlim(0,142)
plt.xlabel("Timestamps")
plt.ylabel("Information Gain")
print ("{}".format(ks_2samp(ABCDD_orig[1:],ABCDDD_orig[1:])))
print ("{}".format(ks_2samp(dd[1:], ddd[1:])))



# plt.subplot(313) 
# for b in range(98):
#     bb = []
#     for bbb in range(141):
#         bb.append(ABCDD_gaus[bbb][b])
#     plt.plot(bb, "r-", label="Re-sampled Media Information Gain Distribution")
# for b in range(98):
#     bb = []
#     for bbb in range(141):
#         bb.append(ABCDDD_gaus[bbb][b])
#     plt.plot(bb, "g-", label="Re-sampled Media Information Gain Distribution")
# plt.ylim(0,2)
# plt.xlim(0,142)
# # print ("{}".format(ks_2samp(ABCDD_gaus, ABCDDD_gaus)))

# plt.subplot(514)    
# # for b in range(98):
# #     bb = []
# #     for bbb in range(141):
# #         bb.append(ABCDD_actual[bbb][b])
# #     plt.plot(bb, "k-", label="Actual Media Information Gain Distribution")
# # plt.ylim(0,2)
# # plt.xlim(0,142)
# plt.subplot(515)
# for b in range(100):
#     bb = []
#     for bbb in range(142):
#         bb.append(ABCDD_gaus_all[bbb][b])
#     plt.plot(bb, "k-", label="Re-sampled Media Information Gain Distribution")
# plt.ylim(0,2)
# plt.xlim(0,142)
# print ("{}".format(ks_2samp(ABCDD_actual, ABCDD_gaus_all)))



plt.show()

import json
def intersection(lst1, lst2): 
    lst3 = [list(filter(lambda x: x in lst1, sublist)) for sublist in lst2] 
    return lst3
rangee = [0.0,1.0,2.0,3.0,4.0]
F11 = []
prec = []
rec = []
acc = []
for abcd in rangee:
    listtt = []
    medias = []
    for i in range(1,101):
        foldername = ("likemindedness_altered_in_degree{}".format(i/100))
        with open(foldername + "/likemindedlistoflistsEnt_Storages", "rb") as lmlol:
            likemindedlistoflists = pickle.load(lmlol)
        with open(foldername+"/likemindedlistoflistsnodes", "rb") as lmlol1:
            likemindedlistoflistsnodes = pickle.load(lmlol1)
        for a in range(143):
            with open('Media_Dictionary_{}.json'.format(a), 'r') as fp:
                data = json.load(fp)
            if likemindedlistoflists[0][a] > abcd:
                listtt.append(a)
    m = set(listtt)
    for iiii in m:
        counteer = iiii+1
        medias.append(data[str(counteer)])
    flat_list = [item for sublist in medias for item in sublist]
    hist, bin_edges = np.histogram(flat_list, bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    summ = np.sum(hist)
    finale = []
    for lmi in range(hist.shape[0]):
        if hist[lmi] > summ/12:
            hist[lmi] = 1
        else:
            hist[lmi] = 0
    print (hist)
    y_test = [1,1,1,1,1,1,0,0,1,1,1,1]
    F11.append(f1_score(y_test, hist.tolist()))
    prec.append(precision_score(y_test, hist.tolist()))
    rec.append(recall_score(y_test, hist.tolist()))
    acc.append(accuracy_score(y_test, hist.tolist()))

    # print (set(medias[0]).intersection(*medias[:1]))
    # print (len(likemindedlistoflists[0]))
    # print (min(likemindedlistoflists[0]))
    # print (max(likemindedlistoflists[0]))
print (F11)  
print (prec)  
print (rec)  
print (acc)  

# Driver Code 
# lst1 = [1, 6, 7, 10, 13, 28, 32, 41, 58, 63] 
# lst2 = [[13, 17, 18, 21, 32], [7, 11, 13, 14, 28], [1, 5, 6, 8, 15, 16]] 
# print(intersection(lst1, lst2))


# print (len(DEPP))
# ABCDDD_orig.sort()
# ABCDD_orig.sort()
# lenn = []
# indexu = []
# for qw in ABCDD_orig:
#     if qw > 0.00:
#         print ("{}:{}".format(qw,sum(i > qw for i in ABCDD_orig)))
#         lenn.append(sum(i > qw for i in ABCDD_orig))
#         indexu.append(qw)

# mins = []
# maxs = []
# ground_truth = [1,2,3,4,5,6,7,8,9,10,11,12]
# thold = 0.0
# F1 = []
# X = []
# for thresholdd in range(int(indexu[lenn.index(max(lenn))]*100), int(max(ABCDDD_orig)*100), 1):
#     thold = thresholdd/100
#     unique = []
#     for i in range(len(DEPP)-1):
#         imp = []
#         if ABCDDD_orig[i] >= thold:
#             mins.append(min(DepEnt_pdf_Media_Out[:,i]))
#             maxs.append(max(DepEnt_pdf_Media_Out[:,i]))
#             for ipl in range(DepEnt_pdf_Media_Out[:,i].shape[0]):
#                 # if DepEnt_pdf_Media_Out[:,i][ipl] > 0.01:
#                     # print (ipl)
#                 imp.append(ipl)
#             import json
#             with open('Media_Dictionary_{}.json'.format(i), 'r') as fp:
#                 data = json.load(fp)
#             for ind in imp:
#                 q = ind/100
#                 threshold = q
#                 foldername = ("likemindedness_altered_in_degree{}".format(threshold))
#                 with open(foldername + "/likemindedlistofliststde", "rb") as lmlol7:
#                     likemindedlistofliststde = pickle.load(lmlol7)

#                 with open(foldername+"/likemindedlistoflistsnodes", "rb") as lmlol1:
#                     likemindedlistoflistsnodes = pickle.load(lmlol1)

#                 with open(foldername + "/likemindedlistoflists", "rb") as lmlol:
#                     likemindedlistoflists = pickle.load(lmlol)
#                 try:
#                     nodes = list(likemindedlistoflists[i])
#                     print (nodes)
#                     # print ("Homogeneity Index:{}".format(q))
#                     # print ("Entropy:{}".format(likemindedlistofliststde[i][0]))
#                     listoflists = []
#                     for qwe in nodes:
#                         if qwe != 0:
#                             if data[str(qwe)]:
#                                 listoflists.append(data[str(qwe)])
#                         listoflists.sort(key = len) 
#                     len_listoflists = []
#                     for abc in listoflists:
#                         len_listoflists.append(len(abc))
#                     listoflists_prime = []
#                     for abc in listoflists:
#                         if len(abc) <= min(len_listoflists):
#                             listoflists_prime.append(abc)
#                             for elem in abc:
#                                 if elem not in unique:
#                                     unique.append(elem)
#                 except KeyError:
#                     continue
#                     # print ("Threshold overlaps")
#                 except IndexError:
#                     continue
#                     # print ("Index Error")
#         else:
#             continue
#     if len(unique) > 0:
#         unique.sort()
#         counter = 0
#         TP = 0
#         FP = 0
#         FN = 0
#         TN = 0
#         if 7 in unique:
#             counter += 1
#         else:
#             TN += 1
#         if 8 in unique:
#             counter += 1
#         else:
#             TN += 1
#         TP = len(unique) - counter
#         FP = counter
#         FN = len(ground_truth) - TP - FP -TN
#         Precision = TP/(TP+FP)
#         Recall = TP/(TP+FN)
#         Accuracy = (TP+FP)/(TP+TN+FP+FN)
#         F_measure = (2*Recall*Precision)/(Recall+Precision)
#         F1.append(F_measure)
#         X.append(thold)
#         print ("Potentially True Media:{} when threshold is {}".format(unique, thold))
#         print ("Precision:{}, Recall:{}, Accuracy:{}, F1-Measure:{}".format(Precision, Recall, Accuracy, F_measure))
#     else:
#         X.append(thold)
#         F1.append(F_measure)

# mins = []
# maxs = []
# ground_truth = [1,2,3,4,5,6,7,8,9,10,11,12]
# thold = 0
# F1_ground = []
# X_ground = []
# for thresholdd in range(0, int(max(dd)*100), 1):
#     thold = thresholdd/100
#     unique = []
#     for i in range(len(DEPP)-1):
#         imp = []
#         if ddd[i] >= thold:
#             mins.append(min(DepEnt_pdf_Media_Out[:,i]))
#             maxs.append(max(DepEnt_pdf_Media_Out[:,i]))
#             for ipl in range(DepEnt_pdf_Media_Out[:,i].shape[0]):
#                 # if DepEnt_pdf_Media_Out[:,i][ipl] > 0.01:
#                     # print (ipl)
#                 imp.append(ipl)
#             import json
#             with open('Media_Dictionary_{}.json'.format(i), 'r') as fp:
#                 data = json.load(fp)
#             for ind in imp:
#                 q = ind/100
#                 threshold = q
#                 foldername = ("likemindedness_altered_in_degree{}".format(threshold))
#                 with open(foldername + "/likemindedlistofliststde", "rb") as lmlol7:
#                     likemindedlistofliststde = pickle.load(lmlol7)

#                 with open(foldername+"/likemindedlistoflistsnodes", "rb") as lmlol1:
#                     likemindedlistoflistsnodes = pickle.load(lmlol1)

#                 with open(foldername + "/likemindedlistoflists", "rb") as lmlol:
#                     likemindedlistoflists = pickle.load(lmlol)
#                 try:
#                     nodes = list(likemindedlistoflists[i])
#                     # print ("Homogeneity Index:{}".format(q))
#                     # print ("Entropy:{}".format(likemindedlistofliststde[i]))
#                     listoflists = []
#                     for qwe in nodes:
#                         if qwe != 0:
#                             if data[str(qwe)]:
#                                 listoflists.append(data[str(qwe)])
#                         listoflists.sort(key = len) 
#                     len_listoflists = []
#                     for abc in listoflists:
#                         len_listoflists.append(len(abc))
#                     listoflists_prime = []
#                     for abc in listoflists:
#                         if len(abc) <= min(len_listoflists):
#                             listoflists_prime.append(abc)
#                             for elem in abc:
#                                 if elem not in unique:
#                                     unique.append(elem)
#                 except KeyError:
#                     continue
#                     # print ("Threshold overlaps")
#                 except IndexError:
#                     continue
#                     # print ("Index Error")
#         else:
#             continue
#     if len(unique) > 0:
#         unique.sort()
#         counter = 0
#         TP = 0
#         FP = 0
#         FN = 0
#         TN = 0
#         if 7 in unique:
#             counter += 1
#         else:
#             TN += 1
#         if 8 in unique:
#             counter += 1
#         else:
#             TN += 1
#         TP = len(unique) - counter
#         FP = counter
#         FN = len(ground_truth) - TP - FP -TN
#         Precision = TP/(TP+FP)
#         Recall = TP/(TP+FN)
#         Accuracy = (TP+FP)/(TP+TN+FP+FN)
#         F_measure = (2*Recall*Precision)/(Recall+Precision)
#         F1_ground.append(F_measure)
#         X_ground.append(thold)
#         print ("Potentially True Media:{} when threshold is {}".format(unique, thold))
#         print ("Precision:{}, Recall:{}, Accuracy:{}, F1-Measure:{}".format(Precision, Recall, Accuracy, F_measure))
#     else:
#         X_ground.append(thold)
#         F1_ground.append(F_measure)

# plt.figure(8)
# plt.plot(X, F1, 'r-')
# plt.plot(X_ground, F1_ground, 'g-')
# plt.ylim(0,1)
# plt.xlabel("Threshold")
# plt.ylabel("F1 Score")
# plt.title("F1 Score")
# plt.savefig("000_All_Timestamps.png")
plt.show()
