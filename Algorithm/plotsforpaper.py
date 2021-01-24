import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import sys
import pickle
import numpy as np
import pandas as pd
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
            numberofnodes += len(timestampoftimestamps[i][j])
            for k in range(len(timestampoftimestamps[i][j])):
                numberoftrues += timestampoftimestamps[i][j][k][0]
                numberoffakes += timestampoftimestamps[i][j][k][1]
                numberofshares += timestampoftimestamps[i][j][k][2]
                credibility += timestampoftimestamps[i][j][k][3]
        temps1.append(numberoftrues)
        temps2.append(numberoffakes)
        temps3.append(numberofshares)
        temps4.append(credibility) 
        temps5.append(numberofnodes) 
        # print("numberoftrues: {}, numberoffakes: {}, numberofshares: {}, credibility: {}".format(numberoftrues, numberoffakes, numberofshares, credibility))
    # print("#############################End of {}th similarity metric timestamps###################################".format(q))
    return temps1, temps2, temps3, temps4, temps5


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
        # likemindedlistoflists = []
        threshold = q
        # foldername = ("likemindedness_{}".format(threshold))
        foldername = ("Media_Similarity_{}".format(threshold))

        with open("Media_Similarity_{}/timestampoftimestamps".format(threshold), "rb") as tsots:
            timestampoftimestamps = pickle.load(tsots)
        # with open("Media_Similarity_{}/likemindedlistoflists".format(threshold), "rb") as lmlol:
        #     likemindedlistoflists = pickle.load(lmlol)
        
        numoft,numoff,numofs,cred,non = data_splitter(timestampoftimestamps, q)
        number_of_trues.append(numoft)
        number_of_fakes.append(numoff)
        number_of_shares.append(numofs)
        credibilities.append(cred)
        number_of_nodes.append(non)
        meanz = np.mean(number_of_trues[-1])
        zzzz.append([meanz for i in range(len(number_of_trues[-1]))])
        meanzz = np.mean(number_of_fakes[-1])
        ZZZZ.append([meanzz for i in range(len(number_of_fakes[-1]))])

    X, Y = np.meshgrid(np.arange(len(number_of_trues[0])), np.arange(len(number_of_trues)))
    number_of_trues = np.asarray(number_of_trues)
    number_of_fakes = np.asarray(number_of_fakes)
    number_of_shares = np.asarray(number_of_shares)
    credibilities = np.asarray(credibilities)
    number_of_nodes = np.asarray(number_of_nodes)
    zzzz = np.asarray(zzzz)
    ZZZZ = np.asarray(ZZZZ)
    df1 = pd.DataFrame(data=zzzz[:,0])
    df1.to_csv('Average_True_Likes.csv', sep=",", header=None, float_format='%.2f', index=False)
    df2 = pd.DataFrame(data=ZZZZ[:,0])
    df2.to_csv('Average_Fake_Likes.csv', sep=",", header=None, float_format='%.2f', index=False)

    print ("Shapes: {}, {}, {}, {}, {}".format(number_of_trues.shape, number_of_fakes.shape, number_of_shares.shape, credibilities.shape, number_of_nodes.shape))

    fig1 = plt.figure()

    ax = plt.axes(projection='3d')
    #ax.scatter(X, Y, zzz, color = 'blue')
    ax.plot_wireframe(X, Y, number_of_trues, color = 'green')
    ax.plot_wireframe(X, Y, number_of_fakes, color = 'red')
    ax.plot_wireframe(X, Y, number_of_shares, color = 'yellow')
    # ax.plot_wireframe(X, Y, credibilities, color = 'blue')
    ax.plot_wireframe(X, Y, number_of_nodes, color = 'blue')


    ax.set_title("Media Distribution")
    ax.set_xlabel('Timestamps')
    ax.set_ylabel('Similarity Metric')
    ax.set_zlabel('Value')

    fig2 = plt.figure()

    ax = plt.axes(projection='3d')
    #ax.scatter(X, Y, zzz, color = 'blue')
    ax.plot_wireframe(X, Y, zzzz, color = 'green')
    ax.plot_wireframe(X, Y, ZZZZ, color = 'red')
    ax.set_title("Media Distribution")
    ax.set_xlabel('Timestamps')
    ax.set_ylabel('Similarity Metric')
    ax.set_zlabel('Value')

    plt.show()
if __name__ == "__main__":
    main()
        # (numberofshares/(len(timestampoftimestamps[i][f]))), (credibility/(len(timestampoftimestamps[i][f])))))
        # plt.subplots_adjust(hspace=.5)
        # plt.figure(figsize=(30, 15))
        # plt.title("Evolution of sub-graph {}".format(f+1))
        # plt.subplot(4, 1, 1)
        # plt.ylim(0.0, 1.0)
        # plt.plot(X, temps1, '-g', label="Percentage of true videos liked")
        # plt.xlabel("Timestamps")
        # plt.ylabel("True Video Likes")
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # plt.subplot(4, 1, 2)
        # plt.ylim(0.0, 1.0)
        # plt.plot(X, temps2, '-r', label="Percentage of fake videos liked")
        # plt.xlabel("Timestamps")
        # plt.ylabel("Fake Video Likes")
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # plt.subplot(4, 1, 3)
        # plt.ylim(0.0, 1.0)
        # plt.plot(X, temps3, '-y', label="Percentage of videos shared")
        # plt.xlabel("Timestamps")
        # plt.ylabel("Video Shares")
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # plt.subplot(4, 1, 4)
        # plt.ylim(0.0, 1.0)
        # plt.plot(X, temps4, '-b', label="Mean Credibility")
        # plt.xlabel("Timestamps")
        # plt.ylabel("Credibility Score")
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.subplots_adjust(hspace=.5)

        # plt.savefig("Media_Similarity_{}/Subgraph_{}.png".format(threshold, f+1))

        # plt.tight_layout()

        # plt.show()
        # plt.close()

    # X = np.linspace(1, 143, 143)
    # for f in range(range_min):
    #     temps1 = []
    #     temps2 = []
    #     temps3 = []
    #     temps4 = []
    #     for i in range(len(timestampoftimestamps)):
    #         numberoftrues = 0
    #         numberoffakes = 0
    #         numberofshares = 0
    #         credibility = 0
    #         temp = []
    #         for k in range(len(timestampoftimestamps[i][f])):
    #             numberoftrues = numberoftrues + timestampoftimestamps[i][f][k][0]/(19)
    #             numberoffakes = numberoffakes + timestampoftimestamps[i][f][k][1]/(21)
    #             numberofshares = numberofshares + timestampoftimestamps[i][f][k][2]/(40)
    #             credibility = credibility + timestampoftimestamps[i][f][k][3]

    #         # temps1.append((numberoftrues/(len(timestampoftimestamps[i][f]))))
    #         # temps2.append((numberoffakes/(len(timestampoftimestamps[i][f]))))
    #         # temps3.append((numberofshares/(len(timestampoftimestamps[i][f]))))
    #         # credibility = (credibility/(len(timestampoftimestamps[i][f])))
    #         # temps4.append(credibility)
    #         temps1.append(numberoftrues/(len(timestampoftimestamps[i][f])))
    #         temps2.append(numberoffakes/(len(timestampoftimestamps[i][f])))
    #         temps3.append(numberofshares/(len(timestampoftimestamps[i][f])))
    #         temps4.append(credibility/(len(timestampoftimestamps[i][f])))  
    #         print("numberoftrues: {}, numberoffakes: {}".format((numberoftrues/(len(timestampoftimestamps[i][f]))), (numberoffakes/(len(timestampoftimestamps[i][f])))))
    #         # (numberofshares/(len(timestampoftimestamps[i][f]))), (credibility/(len(timestampoftimestamps[i][f])))))
    #     # plt.subplots_adjust(hspace=.5)
    #     plt.figure(figsize=(30, 15))
    #     plt.title("Evolution of sub-graph {}".format(f+1))
    #     plt.subplot(4, 1, 1)
    #     plt.ylim(0.0, 1.0)
    #     plt.plot(X, temps1, '-g', label="Percentage of true videos liked")
    #     plt.xlabel("Timestamps")
    #     plt.ylabel("True Video Likes")
    #     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #     plt.subplot(4, 1, 2)
    #     plt.ylim(0.0, 1.0)
    #     plt.plot(X, temps2, '-r', label="Percentage of fake videos liked")
    #     plt.xlabel("Timestamps")
    #     plt.ylabel("Fake Video Likes")
    #     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #     plt.subplot(4, 1, 3)
    #     plt.ylim(0.0, 1.0)
    #     plt.plot(X, temps3, '-y', label="Percentage of videos shared")
    #     plt.xlabel("Timestamps")
    #     plt.ylabel("Video Shares")
    #     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #     plt.subplot(4, 1, 4)
    #     plt.ylim(0.0, 1.0)
    #     plt.plot(X, temps4, '-b', label="Mean Credibility")
    #     plt.xlabel("Timestamps")
    #     plt.ylabel("Credibility Score")
    #     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #     plt.subplots_adjust(hspace=.5)

    #     # plt.savefig("Media_Similarity_{}/Subgraph_{}.png".format(threshold, f+1))

    #     plt.tight_layout()

    #     # plt.show()
    #     plt.close()

