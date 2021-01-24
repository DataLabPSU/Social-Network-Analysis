import numpy as np
from numpy import genfromtxt
import sys
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
import pickle
import os
import time
import math

folder_name = "Data"
folder_dir = os.getcwd() + "/" + folder_name + "/"

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def normalization(data_matrix):
    for i in data_matrix:  # normalization
        counter = 0
        sum = 0
        for j in i:
            i[counter] = float(j/100.00)
            sum = sum + i[counter]
            counter = counter + 1
    return data_matrix

def thresholder(data_matrix, threshold, G):
    data = np.empty((data_matrix.shape[0], data_matrix.shape[1]))
    for ii in range(data_matrix.shape[0]):
        for jj in range(data_matrix.shape[1]):
            if (data_matrix[ii][jj] > threshold) and (G[ii][jj] == 1):
                if (jj != ii):
                    data[ii][jj] = 1
                else:
                    data[ii][jj] = 0
            else:
                data[ii][jj] = 0
    return data

def thresholderr(data_matrix, threshold, G):
    data = np.empty((data_matrix.shape[0], data_matrix.shape[1]))
    for ii in range(data_matrix.shape[0]):
        for jj in range(data_matrix.shape[1]):
            if (data_matrix[ii][jj] >= threshold) and (G[ii][jj] == 1):
                if (jj != ii):
                    data[ii][jj] = 1
                else:
                    data[ii][jj] = 0
            else:
                data[ii][jj] = 0
    return data

def combiner(tda, tdaa, tdaaa):
    if (tda.shape[0] == tdaa.shape[0]) and (tda.shape[1] == tdaa.shape[1]):
        combined = np.empty((tda.shape[0], tda.shape[1]))
        for x in range(tda.shape[0]):
            for y in range (tda.shape[1]):
                if (tda[x][y] == tdaa[x][y]):
                    combined[x][y] = 1
                else:
                    combined[x][y] = 0
    return combined

def indentropy(net, tds):
    states = len(net)
    Entropy = 0
    for node in net:
        row = tds[node]
        row = list(row)
        row = np.asarray(row)
        row = row[net]
        prob_state = np.count_nonzero(row[:] == 1)/states
        try:
            Entropy += prob_state * math.log(prob_state)
        except ValueError:
            Entropy += 0
    return -Entropy

def depentropy(net, tds):
    # states = len(net)
    states = tds.shape[0]
    Entropy = 0
    for node in net:
        row = tds[node]
        row = np.asarray(row)
        prob_state = np.count_nonzero(row[:] == 1)/states
        try:
            Entropy += prob_state * math.log(prob_state)
        except ValueError:
            # print ("0 Entropy")
            Entropy += 0
    return -Entropy

def subgrapher(subs, tdaa):
    subs = list(subs)
    E = indentropy(subs, tdaa)
    EE = depentropy(subs, tdaa)
    return E, EE

def dotmatrix(data, dataa, data_media_follower, data_media_follower_with_only_FA, data_media_follower_with_only_F, data_media_follower_with_FA_and_F, data_media_following, data_media_following_with_only_FA, data_media_following_with_only_F, data_media_following_with_FA_and_F, index, qq, G_FLWR, G_FLWG):    
    data = normalization(data) # node like-mindedness matrix -- in-degree -- follower
    dataa = normalization(dataa)  # node like-mindedness matrix -- out-degree -- following
    data_media_follower = normalization(data_media_follower)  # media like-mindedness matrix
    data_media_follower_with_only_FA = normalization(data_media_follower_with_only_FA)  # media like-mindedness matrix
    data_media_follower_with_only_F = normalization(data_media_follower_with_only_F)  # media like-mindedness matrix
    data_media_follower_with_FA_and_F = normalization(data_media_follower_with_FA_and_F)  # media like-mindedness matrix
    data_media_following = normalization(data_media_following)  # media like-mindedness matrix
    data_media_following_with_only_FA = normalization(data_media_following_with_only_FA)  # media like-mindedness matrix
    data_media_following_with_only_F = normalization(data_media_following_with_only_F)  # media like-mindedness matrix
    data_media_following_with_FA_and_F = normalization(data_media_following_with_FA_and_F)  # media like-mindedness matrix

    if qq == 0.0:
        # thresholdedd_data = thresholderr(data, qq, G_FLWR) #set threshold
        # thresholdedd_dataa = thresholderr(dataa, qq, G_FLWG)  # set threshold

        # thresholdedd_data_media_follower = thresholderr(data_media_follower, qq, G_FLWR)  # set threshold
        # thresholdedd_data_media_follower_with_only_FA = thresholderr(data_media_follower_with_only_FA, qq, G_FLWR)  # set threshold
        # thresholdedd_data_media_follower_with_only_F = thresholderr(data_media_follower_with_only_F, qq, G_FLWR)  # set threshold
        # thresholdedd_data_media_follower_with_FA_and_F = thresholderr(data_media_follower_with_FA_and_F, qq, G_FLWR)  # set threshold

        # thresholdedd_data_media_following = thresholderr(data_media_following, qq, G_FLWG)  # set threshold
        # thresholdedd_data_media_following_with_only_FA = thresholderr(data_media_following_with_only_FA, qq, G_FLWG)  # set threshold
        # thresholdedd_data_media_following_with_only_F = thresholderr(data_media_following_with_only_F, qq, G_FLWG)  # set threshold
        thresholdedd_data_media_following_with_FA_and_F = thresholderr(data_media_following_with_FA_and_F, qq, G_FLWG)  # set threshold

        # G = nx.from_numpy_array(thresholdedd_data)
        # G = nx.from_numpy_array(thresholdedd_dataa)

        # G = nx.from_numpy_array(thresholdedd_data_media_follower)
        # G = nx.from_numpy_array(thresholdedd_data_media_follower_with_only_FA)
        # G = nx.from_numpy_array(thresholdedd_data_media_follower_with_only_F)
        # G = nx.from_numpy_array(thresholdedd_data_media_follower_with_FA_and_F)

        # G = nx.from_numpy_array(thresholdedd_data_media_following)
        # G = nx.from_numpy_array(thresholdedd_data_media_following_with_only_FA)
        # G = nx.from_numpy_array(thresholdedd_data_media_following_with_only_F)
        G = nx.from_numpy_array(thresholdedd_data_media_following_with_FA_and_F)

    else:
        # thresholded_data = thresholder(data, qq, G_FLWR) #set threshold
        # thresholded_dataa = thresholder(dataa, qq, G_FLWG)  # set threshold

        # thresholded_data_media_follower = thresholder(data_media_follower, qq, G_FLWR)  # set threshold
        # thresholded_data_media_follower_with_only_FA = thresholder(data_media_follower_with_only_FA, qq, G_FLWR)  # set threshold
        # thresholded_data_media_follower_with_only_F = thresholder(data_media_follower_with_only_F, qq, G_FLWR)  # set threshold
        # thresholded_data_media_follower_with_FA_and_F = thresholder(data_media_follower_with_FA_and_F, qq, G_FLWR)  # set threshold

        # thresholded_data_media_following = thresholder(data_media_following, qq, G_FLWG)  # set threshold
        # thresholded_data_media_following_with_only_FA = thresholder(data_media_following_with_only_FA, qq, G_FLWG)  # set threshold
        # thresholded_data_media_following_with_only_F = thresholder(data_media_following_with_only_F, qq, G_FLWG)  # set threshold
        thresholded_data_media_following_with_FA_and_F = thresholder(data_media_following_with_FA_and_F, qq, G_FLWG)  # set threshold

        # comment if analyzing based on a single matrix type only
        # combined_matrix = combiner(thresholded_data, thresholded_dataa, thresholded_dataaa)

        # G = nx.from_numpy_array(thresholded_data)
        # G = nx.from_numpy_array(thresholded_dataa)

        # G = nx.from_numpy_array(thresholded_data_media_follower)
        # G = nx.from_numpy_array(thresholded_data_media_follower_with_only_FA)
        # G = nx.from_numpy_array(thresholded_data_media_follower_with_only_F)
        # G = nx.from_numpy_array(thresholded_data_media_follower_with_FA_and_F)

        # G = nx.from_numpy_array(thresholded_data_media_following)
        # G = nx.from_numpy_array(thresholded_data_media_following_with_only_FA)
        # G = nx.from_numpy_array(thresholded_data_media_following_with_only_F)
        G = nx.from_numpy_array(thresholded_data_media_following_with_FA_and_F)

    G.remove_nodes_from(list(nx.isolates(G)))
    # sub_graphs = list(nx.connected_component_subgraphs(G))
    sub_graphs = list(greedy_modularity_communities(G))
    noofnodes = []
    totaldegree = []
    avgdegree = []
    avgcc = []
    listtt = []
    dependent_entropy = []
    independent_entropy = []
    total_dependent_entropy = []
    total_independent_entropy = []
    de_sum = 0.0
    ie_sum = 0.0
    edges = []
    for i in range(len(sub_graphs)):
        s = 0
        cc = 0
        edges.append(G.edges(list(sub_graphs[i])))
        if qq == 0.0:
            e, ee = subgrapher(sub_graphs[i], thresholdedd_data_media_following_with_FA_and_F)
        else:
            e, ee = subgrapher(sub_graphs[i], thresholded_data_media_following_with_FA_and_F)
        de_sum = de_sum + ee
        ie_sum = ie_sum + e
        for j in sub_graphs[i]:
            s += G.degree[j]
            cc = nx.clustering(G,j)
        avg = float(s)/len(sub_graphs[i])
        avg_cc = float(cc)/len(sub_graphs[i])
        # print("The total number of edges in Sub-Graph_{} is: {}".format(i, s))
        # print("Average Degree for Sub-Graph_{} is: {}".format(i, avg))
        # print("Average clustering coefficient for Sub-Graph_{} is: {}".format(i, avg_cc))
        # print("# of nodes in Sub-Graph_{} is: {}".format(i, len(sub_graphs[i])))
        # print("Independent Entropy in Sub-Graph_{} is: {}".format(i, e))
        # print("Dependent Entropy in Sub-Graph_{} is: {}".format(i, ee))

        # N, K = sub_graphs[i].order(), sub_graphs[i].size()
        # avg_deg = float(K)/N
        noofnodes.append(len(sub_graphs[i]))
        totaldegree.append(s)
        avgdegree.append(avg)
        avgcc.append(avg_cc)
        listtt.append(sub_graphs[i])
        dependent_entropy.append(ee)
        independent_entropy.append(e)
    # listttt = [x for y in edges for x in y]
    total_dependent_entropy.append(de_sum)
    total_independent_entropy.append(ie_sum)
    # print("######################################################################################")
    # pos = nx.circular_layout(GG)  # positions for all nodes
    # if index == 143:
    #     nx.draw_circular(GG, with_labels = False)
    #     nx.draw_networkx_edges(GG, pos, edgelist=listttt, width=0.1, alpha=0.5, edge_color='r')
    #     plt.title("# of subgraphs {}".format(len(sub_graphs)))
    #     plt.savefig("Network_In_Degree_Similarity{}.png".format(qq))
    #     plt.close()
    #     G.clear()
    return noofnodes, totaldegree, avgdegree, avgcc, listtt, dependent_entropy, independent_entropy, total_dependent_entropy, total_independent_entropy
    # return None

##Eigen Decomposition of the like mindedness matrix
    # for row in data:
    #     node_list = []
    #     for rows in data:
    #         k = np.dot(row, rows)
    #         node_list.append(k)
    #     listoflists.append(node_list)
    # final = np.array(listoflists)
    # u, s, vh = np.linalg.svd(final, full_matrices=True)
    # S = u.dot(s).dot(vh)
    # print (S)

    # return None

def nodelistsformatter(file, s, dk):
    tlist = []
    file= file.astype(int)
    for f in dk[s+1]:
        tlist.append(file[f].tolist())
    return tlist

def offsetter(timestampedgraph):
    for j in range(len(timestampedgraph)):
        for k in range(len(timestampedgraph[j])):
            timestampedgraph[j][k] = timestampedgraph[j][k] + 1
    return timestampedgraph

def opener(f, threshold):

    with open(f+"/likemindedlistoflistsnodes", "rb") as lmlol1:
        likemindedlistoflistsnodes = pickle.load(lmlol1)
    # with open("Media_Similarity_{}/likemindedlistofliststd".format(threshold), "rb") as lmlol2:
    #     likemindedlistofliststd = pickle.load(lmlol2)
    # with open("Media_Similarity_{}/likemindedlistoflistsavgd".format(threshold), "rb") as lmlol3:
    #     likemindedlistoflistsavgd = pickle.load(lmlol3)
    with open(f + "/likemindedlistoflistsavgcc", "rb") as lmlol4:
        likemindedlistoflistsavgcc = pickle.load(lmlol4)
    with open(f + "/likemindedlistoflists", "rb") as lmlol:
        likemindedlistoflists = pickle.load(lmlol)
    with open(f + "/likemindedlistoflistsde", "rb") as lmlol5:
        likemindedlistoflistsde = pickle.load(lmlol5)
    with open(f + "/likemindedlistoflistsie", "rb") as lmlol6:
        likemindedlistoflistsie = pickle.load(lmlol6)
    with open(f + "/likemindedlistofliststde", "rb") as lmlol7:
        likemindedlistofliststde = pickle.load(lmlol7)
    with open(f + "/likemindedlistofliststie", "rb") as lmlol8:
        likemindedlistofliststie = pickle.load(lmlol8)

    # return likemindedlistoflistsnodes, likemindedlistofliststd, likemindedlistoflistsavgd, likemindedlistoflistsavgcc, likemindedlistoflists
    return likemindedlistoflistsnodes, likemindedlistoflistsavgcc, likemindedlistoflistsde, likemindedlistoflistsie, likemindedlistofliststde, likemindedlistofliststie, likemindedlistoflists


def main():
    for q in range(0,101):
        q = q/100
        likemindedlistoflists = []
        threshold = q
        print (threshold)
        # foldername = ("Node_altered_likemindedness_in_degree{}".format(threshold))
        # foldername = ("Node_altered_likemindedness_out_degree{}".format(threshold))

        # foldername = ("likemindedness_altered_in_degree{}".format(threshold))
        # foldername = "likemindedness_altered_in_degree_with_only_FA{}".format(threshold)
        # foldername = "likemindedness_altered_in_degree_with_FA_and_F{}".format(threshold)
        # foldername = ("likemindedness_altered_out_degree{}".format(threshold))
        # foldername = "likemindedness_altered_out_degree_with_only_FA{}".format(threshold)
        foldername = "likemindedness_altered_out_degree_with_FA_and_F{}".format(threshold)

        # foldername = ("likemindedness_in_degree{}".format(threshold))
        # foldername = "likemindedness_in_degree_with_only_FA{}".format(threshold)
        # foldername = "likemindedness_in_degree_with_FA_and_F{}".format(threshold)
        # foldername = ("likemindedness_out_degree{}".format(threshold))
        # foldername = "likemindedness_out_degree_with_only_FA{}".format(threshold)
        # foldername = "likemindedness_out_degree_with_FA_and_F{}".format(threshold)

        createFolder(foldername)
        folder_dir = os.getcwd() + "/" + folder_name + "/"

        with open("impressionlistfiles", "rb") as ilf:
            impressionlist = pickle.load(ilf)
        with open("credibilitylistfiles", "rb") as clf:
            credibilitylist = pickle.load(clf)
        with open("derivedlikesfiles", "rb") as dlf:
            derivedlist = pickle.load(dlf)
        with open("dictkeys", "rb") as dk:
            dictkeys = pickle.load(dk)
        nnodess = []
        tds = []
        avgds = []
        avgccs = []
        iee = []
        dee = []
        tdee =[]
        tiee = []
        for i in range(1,144,1):
            my_data_1 = genfromtxt('Follower_Node_Similarity_{}.csv'.format(i), delimiter=',') #in-degree
            my_data_2 = genfromtxt('Following_Node_Similarity_{}.csv'.format(i), delimiter=',') #out-degree
            my_data_3 = genfromtxt('Follower_Media_Similarity_{}.csv'.format(i), delimiter=',')
            my_data_4 = genfromtxt('Follower_Media_Similarity_{}_with_only_FA.csv'.format(i), delimiter=',')
            my_data_5 = genfromtxt('Follower_Media_Similarity_{}_with_only_F.csv'.format(i), delimiter=',')
            my_data_6 = genfromtxt('Follower_Media_Similarity_{}_with_FA_and_F.csv'.format(i), delimiter=',')
            my_data_7 = genfromtxt('Following_Media_Similarity_{}.csv'.format(i), delimiter=',')
            my_data_8 = genfromtxt('Following_Media_Similarity_{}_with_only_FA.csv'.format(i), delimiter=',')
            my_data_9 = genfromtxt('Following_Media_Similarity_{}_with_only_F.csv'.format(i), delimiter=',')
            my_data_10 = genfromtxt('Following_Media_Similarity_{}_with_FA_and_F.csv'.format(i), delimiter=',')
            G_follower = np.load("follower_matrix_{}.npy".format(i))
            G_following = np.load("following_matrix_{}.npy".format(i))


            my_data_1 = my_data_1.astype(float)
            my_data_2 = my_data_2.astype(float)
            my_data_3 = my_data_3.astype(float)
            my_data_4 = my_data_4.astype(float)
            my_data_5 = my_data_5.astype(float)
            my_data_6 = my_data_6.astype(float)
            my_data_7 = my_data_7.astype(float)
            my_data_8 = my_data_8.astype(float)
            my_data_9 = my_data_9.astype(float)
            my_data_10 = my_data_10.astype(float)
            
            nnodes, td, avgd, avgcc, listt, de, ie, tde, tie = dotmatrix(my_data_1, my_data_2, my_data_3, my_data_4, my_data_5, my_data_6, my_data_7, my_data_8, my_data_9, my_data_10, i, q, G_follower, G_following)
            nnodess.append(nnodes)
            tds.append(td)
            avgds.append(avgd)
            avgccs.append(avgcc)
            likemindedlistoflists.append(listt)
            iee.append(ie)
            dee.append(de)
            tdee.append(tde)
            tiee.append(tie)

        with open(foldername + '/likemindedlistoflistsnodes', "wb") as lmlol1: #Total number of nodes per Sub-Graph
            pickle.dump(nnodess, lmlol1)
        # with open("Media_Similarity_{}/likemindedlistofliststd".format(threshold), "wb") as lmlol2: #Total Degree per Sub-Graph
        #     pickle.dump(tds, lmlol2)
        # with open("Media_Similarity_{}/likemindedlistoflistsavgd".format(threshold), "wb") as lmlol3: #Average Degree per Sub-Graph
        #     pickle.dump(avgds, lmlol3)
        with open(foldername + "/likemindedlistoflistsavgcc", "wb") as lmlol4: #Average Clustering Co-efficient per Sub-Graph
            pickle.dump(avgccs, lmlol4)
        with open(foldername + "/likemindedlistoflists", "wb") as lmlol: #List of nodes in each Sub-Graph
            pickle.dump(likemindedlistoflists, lmlol)
        with open(foldername + "/likemindedlistoflistsde", "wb") as lmlol5: #Dependent Entropy per Sub-Graph
            pickle.dump(dee, lmlol5)
        with open(foldername + "/likemindedlistoflistsie", "wb") as lmlol6: #Independent Entropy per Sub-Graph
            pickle.dump(iee, lmlol6)
        with open(foldername + "/likemindedlistofliststde", "wb") as lmlol7: #Total Dependent Entropy
            pickle.dump(tdee, lmlol7)
        with open(foldername + "/likemindedlistofliststie", "wb") as lmlol8: #Total Independent Entropy
            pickle.dump(tiee, lmlol8)

        # time.sleep(60)
        a,b,c,d,e,f,g = opener(foldername, threshold)
        MasterList = []
        MasterList.append(a)
        MasterList.append(b)
        MasterList.append(c)
        MasterList.append(d)
        MasterList.append(e)
        MasterList.append(f)
        # MasterList.append(g)
        names = ["Number of nodes per community", "Average clustering coefficient per community", "Dependent Entropy", "Independent Entropy", "Total Dependent Entropy", "Total Independent, Entropy"]

        title_names = ["Number of nodes \n Media_Similarity Metric : {}".format(threshold),"Average clustering coefficient \n Media_Similarity Metric : {}".format(threshold), 
                       "Dependent Entropy \n Media_Similarity Metric : {}".format(threshold), "Independent Entropy \n Media_Similarity Metric : {}".format(threshold),
                       "Total Dependent Entropy \n Media_Similarity Metric : {}".format(threshold), "Total Independent Entropy \n Media_Similarity Metric : {}".format(threshold)]
        # title_names = ["Number of nodes \n Node_Similarity Metric : {}".format(threshold),"Average clustering coefficient \n Node_Similarity : {}".format(threshold), 
        #                "Dependent Entropy \n Node_Similarity Metric : {}".format(threshold), "Independent Entropy \n Node_Similarity Metric : {}".format(threshold),
        #                "Total Dependent Entropy \n Node_Similarity Metric : {}".format(threshold), "Total Independent Entropy \n Node_Similarity Metric : {}".format(threshold)]
        x = np.linspace(0, len(a), len(a))
        number = max(len(l) for l in a)
        symbols = ['-rs', '-g^', '-bo', '-y*', '-kh', '-m8', '-cD', '-g', '-rx', '-bp', '-kP', '-mH',
                   '-c,', '-gs', '-r^', '-yo', '-b*', '-mh', '-k8', '-gD', '-r', '-bx', '-rp', '-mP', 
                   '-kH', '-bs', '-b^', '-mo', '-r*', '-ch', '-b8', '-yD', '-b', '-kx', '-cp', '-gP', '-bH']
        categories = ['Community 1', 'Community 2', 'Community 3', 'Community 4', 'Community 5', 'Community 6',
                      'Community 7', 'Community 8', 'Community 9', 'Community 10', 'Community 11', 'Community 12',
                      'Community 13', 'Community 14', 'Community 15', 'Community 16', 'Community 17', 'Community 18',
                      'Community 19', 'Community 20', 'Community 21', 'Community 22', 'Community 23', 'Community 24',
                      'Community 25', 'Community 26', 'Community 27', 'Community 28', 'Community 29', 'Community 30']
        for index in range(len(MasterList)-2):
            for ll in range(len(MasterList[index])):
                num = 0
                if len(MasterList[index][ll]) != number:
                    num = number - len(MasterList[index][ll])
                    for a in range(num):
                        MasterList[index][ll].append(0)
                else:
                    continue
            plt.figure(figsize=(12, 8))
            
            for kk in range(number):
                KKK = []
                for kkk in MasterList[index]:
                    KKK.append(kkk[kk])
                plt.plot(x, KKK, symbols[kk], lw=1.5)
                plt.title(title_names[index])
                plt.xlabel("Timestamps")
                plt.ylabel(names[index])
                plt.legend(categories[:number], loc='center left', bbox_to_anchor=(1, 0.5))
                plt.tight_layout()
                plt.savefig(foldername+ "/Plot_{}_{}.png".format( index, threshold))
            plt.close()
        
        for index in range(4,len(MasterList)):
            plt.figure(figsize=(12, 8))
            KKKK = []
            for kkkk in MasterList[index]:
                KKKK.append(kkkk)
            plt.plot(x, KKKK, lw=1.5)
            plt.title(title_names[index])
            plt.xlabel("Timestamps")
            plt.ylabel(names[index])
            # plt.legend(categories[:number], loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.savefig(foldername+ "/Plot_{}_{}.png".format(index, threshold))
            plt.close()

        for j in range(len(dictkeys)):
            for k in range(len(dictkeys[j])):
                dictkeys[j][k] = dictkeys[j][k] - 1

        timestampoftimestamps = []

        for x in range(len(impressionlist)):

            file_contents_impressions = np.loadtxt(folder_dir+impressionlist[x+1], dtype=np.float, delimiter=" ")
            # file_contents_impressions = nodelistsformatter(file_contents_impressions, x, dictkeys) #list of lists containing the nodes corresponding to the following edgelist at timestamp (x+1)

            file_contents_credibility = np.loadtxt(folder_dir+credibilitylist[x+1], dtype=np.float, delimiter=" ") 
            # file_contents_credibility = nodelistsformatter(file_contents_credibility, x, dictkeys) # list of lists containing the nodes corresponding to the following edgelist at timestamp (x+1)

            file_contents_derived = np.loadtxt(folder_dir+derivedlist[x+1], dtype=np.float, delimiter=" ")
            # file_contents_derived = nodelistsformatter(file_contents_derived, x, dictkeys) # list of lists containing the nodes corresponding to the following edgelist at timestamp (x+1)
            # print (file_contents_derived[x])
            # print (file_contents_derived)
            k = g[x]
            # print (k)
            timestamp = []
            
            # print(len(k))

            for l in k:
                subgraphtimestamp = []

                for m in l:
                    # print(m)
                    if m not in l:
                        continue
                    else:
                        nodetimestamp = []
                        # print (file_contents_derived[m][2])
                        nodetimestamp.append(file_contents_derived[m][1])
                        nodetimestamp.append(file_contents_derived[m][2])
                        nodetimestamp.append(file_contents_impressions[m][4])
                        nodetimestamp.append(file_contents_credibility[m][1])
                        subgraphtimestamp.append(nodetimestamp)
                timestamp.append(subgraphtimestamp)
            timestampoftimestamps.append(timestamp)
            if len(impressionlist)-1 == len(timestampoftimestamps):
                break
            with open(foldername + "/timestampoftimestamps", "wb") as tsots: #Total Independent Entropy
                pickle.dump(timestampoftimestamps, tsots)

if __name__ == "__main__":
    main()
