import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib
import statistics
import os
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'figure.autolayout': True})

constants = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
iterr = 0
skip = 3
acccc = []
preccc = []
reccc = []
F1111 = []
for constant in constants:
    foldername = "fscore_time_Bayesian/{}".format(constant)
    # createFolder(foldername)
    folder_dir = os.getcwd() + "/" + foldername + "/"
    print (folder_dir)
    while (iterr+constant) <=142:
        F111 = np.load(folder_dir + "F1_{}-{}.npy".format(iterr, iterr+constant))
        precc = np.load(folder_dir + "Precision_{}-{}.npy".format(iterr, iterr+constant))
        recc = np.load(folder_dir + "Recall_{}-{}.npy".format(iterr, iterr+constant))
        acc = np.load(folder_dir + "Accuracy_{}-{}.npy".format(iterr, iterr+constant))
        iterr += skip
    iterr = 0
    F1111.append(statistics.mean(F111))
    acccc.append(statistics.mean(acc))
    reccc.append(statistics.mean(recc))
    preccc.append(statistics.mean(precc))

plt.figure(3)
plt.plot(constants, F1111, 'o-', color='black', label="F1")
plt.plot(constants, preccc, 'o-', color='blue', label="precision")
plt.plot(constants, reccc, 'o-', color='orange', label="recall")
plt.plot(constants, acccc, 'o-', color='green', label="accuracy")

plt.ylim(0,1)
plt.xlabel("Sliding Window")
plt.ylabel("Evaluation Metrics")
plt.legend(loc="upper right")
plt.title("Sliding Window Vs. Evaluation Metrics")
fig = plt.gcf()
fig.set_size_inches((8.5, 11), forward=False)
ff = ("fscore_time_Bayesian/")
plt.show()

# fig.savefig(ff + "Metrics_All.png", dpi=500)