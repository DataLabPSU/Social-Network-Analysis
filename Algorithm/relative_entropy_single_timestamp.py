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
import json
import os
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib
import statistics
matplotlib.rcParams.update({'font.size': 16})
# matplotlib.rcParams.update({'font.weight': 'bold'})
matplotlib.rcParams.update({'figure.autolayout': True})
np.random.seed(0)
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

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

for xyz in range(nott_pdf_Media_Out.shape[1]-1):
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

kk = np.zeros((nott_pdf_Media_Out.shape[0],nott_pdf_Media_Out.shape[1]))
for k in range(nott_pdf_Media_Out.shape[1]):
    conv_pmf = signal.fftconvolve(nott_pdf_Media_Out[:,k], noff_pdf_Media_Out[:,k],'same')
    kk[:,k] = conv_pmf/sum(conv_pmf)

ABCD_orig = []
ABCD_gaus = []
ABCD_actual = []
ABCD_gaus_all = []
DEPP = []
constants = [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
iterr = 0
skip = 3
acccc = []
preccc = []
reccc = []
F1111 = []
for constant in constants:
    foldername = "fscore_time_Bayesian/{}".format(constant)
    createFolder(foldername)
    folder_dir = os.getcwd() + "/" + foldername + "/"
    print (folder_dir)
    F111 = []
    precc = []
    recc = []
    accc = []
    while (iterr+constant) <=142:
        for abcdd in range(iterr, iterr+constant):
            gmm = mixture.GaussianMixture(n_components=2, covariance_type='full', max_iter=100).fit(DepEnt_pdf_Media_Outttt[1:,abcdd].reshape(-1,1))
            orig, gaus, DEP = plot_results(DepEnt_pdf_Media_Outttt[1:,abcdd].reshape(-1,1), gmm.predict(DepEnt_pdf_Media_Outttt[1:,abcdd].reshape(-1,1)), gmm.means_, gmm.covariances_, 0, 'Expectation-maximization', nott_pdf_Media_Out, noff_pdf_Media_Out, kk[1:,abcdd], abcdd)

            ABCD_orig.append(orig)
            ABCD_gaus.append(gaus)
            DEPP.append(DEP)

        ABCDD_orig = [ABCD_orig[jj][0] for jj in range(len(ABCD_orig))]
        ABCDDD_orig = [ABCD_orig[jj][1] for jj in range(len(ABCD_orig))]
        ABCDD_orig = [x for x in ABCDD_orig if ~np.isnan(x)]
        ABCDDD_orig = [y for y in ABCDDD_orig if ~np.isnan(y)]
        ABCDD_orig = [xx for xx in ABCDD_orig if ~np.isinf(xx)]
        ABCDDD_orig = [yy for yy in ABCDDD_orig if ~np.isinf(yy)]


        plt.figure(1,figsize=(8.0, 5.0))
        plt.subplot(411)
        sns.distplot(d[iterr:iterr+constant], hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "I(users who liked authentic information only)", color = 'green')
        # sns.distplot(dddd, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "Information gain distribution from entropy of users who have only liked False Media", color = 'red')
        plt.xlim(0.0,1.0)
        plt.ylim(0,10.0)
        plt.xlabel("Information Gain")
        plt.ylabel("Density")
        plt.legend(loc="upper right")


        plt.subplot(412)
        # sns.distplot(ddddd, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "Information gain from True Video Distribution between metrics", color = 'green')
        sns.distplot(dddddd[iterr:iterr+constant], hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "I(users who liked misinformation only)", color = 'red')
        plt.xlim(0.0,1.0)
        plt.ylim(0,10.0)
        plt.xlabel("Information Gain")
        plt.ylabel("Density")
        plt.legend(loc="upper right")


        plt.subplot(413)
        sns.distplot(dd[iterr:iterr+constant], hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "I(all users)", color = 'green')
        sns.distplot(ddd[iterr:iterr+constant], hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label = "I(all users)", color = 'red')
        plt.xlim(0.0,1.0)
        plt.ylim(0,10.0)
        plt.xlabel("Information Gain")
        plt.ylabel("Density")
        plt.legend(loc="upper right")

        plt.subplot(414)
        sns.distplot(ABCDD_orig, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label="I'(all users) [I(A)] from GMM", color = 'green')
        sns.distplot(ABCDDD_orig, hist = False, kde = True, norm_hist=True, kde_kws = {'shade': True, 'linewidth': 1.0, 'cumulative':False, 'bw':0.05}, label="I'(all users) [I(B)] from GMM", color = 'red')
        plt.xlim(0.0,1.0)
        plt.ylim(0,10.0)
        plt.xlabel("Information Gain")
        plt.ylabel("Density")
        plt.legend(loc="upper right")
        fig = plt.gcf()
        fig.set_size_inches((8.5, 11), forward=False)
        fig.savefig(folder_dir + "IG_{}-{}.png".format(iterr, iterr+constant), dpi=500)
        plt.close()
        
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
                for a in range(iterr,iterr+constant):
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

        plt.figure(2)
        plt.plot(rangee, F11, 'o-', color='black', label="F1")
        plt.plot(rangee, prec, 'o-', color='blue', label="precision")
        plt.plot(rangee, rec, 'o-', color='orange', label="recall")
        plt.plot(rangee, acc, 'o-', color='green', label="accuracy")

        plt.ylim(0,1)
        plt.xlabel("Threshold")
        plt.ylabel("Evaluation Metrics")
        plt.legend(loc="upper right")
        plt.title("{}-{} timestamps F1 Score".format(iterr, iterr+constant))
        fig = plt.gcf()
        fig.set_size_inches((8.5, 11), forward=False)
        fig.savefig(folder_dir + "Metrics_{}-{}.png".format(iterr, iterr+constant), dpi=500)
        # plt.savefig(folder_dir + "Metrics_{}-{}.png".format(iterr, iterr+constant), dpi=100)
        plt.close()
        F111.append(statistics.mean(F11))
        precc.append(statistics.mean(prec))
        recc.append(statistics.mean(rec))
        accc.append(statistics.mean(acc))
        np.save(folder_dir + "F1_{}-{}.npy".format(iterr, iterr+constant), F111)
        np.save(folder_dir + "Precision_{}-{}.npy".format(iterr, iterr+constant), precc)
        np.save(folder_dir + "Recall_{}-{}.npy".format(iterr, iterr+constant), recc)
        np.save(folder_dir + "Accuracy_{}-{}.npy".format(iterr, iterr+constant), accc)
        iterr += skip
    iterr = 0
#     F1111.append(statistics.mean(F111))
#     acccc.append(statistics.mean(accc))
#     reccc.append(statistics.mean(recc))
#     preccc.append(statistics.mean(precc))

# plt.figure(3)
# plt.plot(constants, F11, 'o-', color='black', label="F1")
# plt.plot(constants, prec, 'o-', color='blue', label="precision")
# plt.plot(constants, rec, 'o-', color='orange', label="recall")
# plt.plot(constants, acc, 'o-', color='green', label="accuracy")

# plt.ylim(0,1)
# plt.xlabel("Sliding Window")
# plt.ylabel("Evaluation Metrics")
# plt.legend(loc="upper right")
# plt.title("Sliding Window Vs. Evaluation Metrics")
# fig = plt.gcf()
# fig.set_size_inches((8.5, 11), forward=False)
# ff = ("fscore_time_Bayesian/")

# fig.savefig(ff + "Metrics_All.png", dpi=500)