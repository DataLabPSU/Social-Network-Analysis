import os
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import scipy as sc
from collections import defaultdict
import pandas as pd
import pickle
import json

folder_name = "Data"
folder_dir = os.getcwd() + "/" + folder_name + "/"

credibility_list = []
edgelist_list = []
impressions_list = []
follower_list = []
likes_list = []
labels_list = []
likes_dict = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [], '11': [], '12': []}
countofcounts = []

def label_dictionary_creater(ll, cc):
    Master_labels_list = []
    Master_labels_listt = []
    ld = {}
    # Reading non-uniform data from file into array with NumPy
    with open(folder_dir+ll[cc]) as f:
        a = f.read().splitlines()
    for xx in a:
        Master_labels_list.append(list(xx.split(' ')))

    for elements in Master_labels_list:
        tt = []
        for elementt in elements:
            tt.append(int(elementt))
        Master_labels_listt.append(tt)
    video_dict = create_video__dict(Master_labels_listt)

    for i in Master_labels_listt:
        # b = set([1,6,7,8,11])
        a = set(i[1:])
        if len(list(a)) > 0:
            likes_list = list(a)
            # print(likes_list)
            for item in range(len(likes_list)):
                likes_dict[str(likes_list[item])].append(i[0])
        else:
            likes_dict['0'].append(i[0])
        # c = a & b
        # print(list(a))
        # if len(c) > 0:
        #     ld[int(i[0])] = []
        # else:
        #     ld[int(i[0])] = i[1:]
    # print (ld.items())
    for n in likes_dict.keys():
        likes_dict[str(n)] = list(set(likes_dict[n]))

    # for x in likes_dict:
    #     print(x, ':', likes_dict[x])

    count = []
    for x in likes_dict:
        count.append(len(likes_dict[x]))

    countofcounts.append(count)
    # print (len(countofcounts))

    return video_dict, count

def create_video__dict(video_liked_list):
    video_dict = {}
    for lists in video_liked_list:
        if len(lists)==1:
            video_dict[lists[0]] = []
        else:
            video_dict[lists[0]] = lists[1:]

    return video_dict


def Masker(impressions_list, edgelist_list, credibility_list, follower_list, labels_list, counterr, dk, lk, dkfr, nfs):
    file_contents_impressions = np.loadtxt(folder_dir+impressions_list[counterr], dtype=np.float, delimiter=" ")
    masked_list_impressions = [] # nodes which were completely inactive during the study

    for impression in file_contents_impressions:
        if (impression[1] - 1) < 1 and impression[2] < 1 and (impression[3] - 1) < 1 and impression[4] < 1:
            masked_list_impressions.append(int(impression[0]))
        else:
            pass
    masked_list_impressions.sort()
    # print (counterr)
    file_contents_edgelist = np.loadtxt(folder_dir+edgelist_list[counterr], dtype=np.float, delimiter=" ")
    masked_list_edgelist = []
    file_credibility_list = np.loadtxt(folder_dir+credibility_list[counterr], dtype=np.float, delimiter=" ")
    check_array = list(range(1, file_credibility_list.shape[0]))

    for edge in file_contents_edgelist:
        masked_list_edgelist.append(int(edge[0]))
    either_or = list(set(check_array) - set(masked_list_edgelist)) # nodes which are either completely inactive or did not follow any other nodes
    no_following = list(set(either_or) - set(masked_list_impressions)) # nodes which did not follow any other nodes but only had followers
    either_or.sort()
    no_following.sort()
    # print (len(either_or))
    # print (len(no_following))

    graph = nx.read_edgelist(folder_dir+edgelist_list[counterr])
    dictionary_of_graph = nx.to_dict_of_lists(graph)
    dictionary_of_graph = {int(k): [int(i) for i in v] for k, v in dictionary_of_graph.items()}
    # nx.write_adjlist(graph, "following_{}.adjlist".format(counterr)) #writes the followers of a node to as the target

    following_dictionary = {}
    if file_contents_edgelist.shape[0] > 0:

        file_contents_edgelist_source = file_contents_edgelist[:, 0].tolist()
        file_contents_edgelist_targets = file_contents_edgelist[:, 1].tolist()

        unique = set(file_contents_edgelist_source)  
        # print (len(unique))      
        for each in unique:
            # following_dictionary[int(each)] = file_contents_edgelist_source.count(each) # to get a count on the number of nodes followed by each node
            temp = []
            filtered = filter(lambda i: file_contents_edgelist_source[i] == each, range(len(file_contents_edgelist_source)))
            for element in list(filtered):
                temp.append(int(file_contents_edgelist_targets[element]))
            temp = list(set(temp))
            temp.sort()
            # print (temp)
            following_dictionary[int(each)] = temp
        else:
            pass
        dk.append(list(following_dictionary.keys()))

    
    file_contents_follower = np.loadtxt(folder_dir+follower_list[counterr], dtype=np.float, delimiter=" ")
    graphh = nx.read_edgelist(folder_dir+follower_list[counterr])
    dictionary_of_graphh = nx.to_dict_of_lists(graphh)
    dictionary_of_graphh = {int(k): [int(i) for i in v] for k, v in dictionary_of_graphh.items()}

    # nx.write_adjlist(graphh, "follower_{}.adjlist".format(counterr)) #writes the followers of a node to as the target
    follower_dictionary = {}

    if file_contents_follower.shape[0]  > 0:
        file_contents_follower_source = file_contents_follower[:, 0].tolist()
        file_contents_follower_targets = file_contents_follower[:, 1].tolist()
        unique = set(file_contents_follower_source)        
        for each in unique:
            # follower_dictionary[int(each)] = file_contents_follower_source.count(each) # to get a count on the number of nodes followed by each node
            temp = []
            filtered = filter(lambda i: file_contents_follower_source[i] == each, range(len(file_contents_follower_source)))
            for element in list(filtered):
                temp.append(int(file_contents_follower_targets[element]))
            temp = list(set(temp))
            temp.sort()
            # print (temp)
            follower_dictionary[int(each)] = temp
        else:
            pass
        dkfr.append(list(follower_dictionary.keys()))
    # print (following_dictionary)
    # correlation_following(following_dictionary, counterr, dk, nfs)
    correlation_follower(follower_dictionary, counterr, dkfr, nfs)

    label_dict, cc = label_dictionary_creater(labels_list, counterr)
    lk.append(list(label_dict.keys()))

    # media_correlation_following(label_dict, counterr, lk, nfs, following_dictionary)
    # media_correlation_follower(label_dict, counterr, lk, nfs, follower_dictionary)
    return cc, label_dict
    # return None
    # return print("Nodes which were completely inactive during the study:\n {}\n Nodes which did not follow any other nodes but only had followers:\n {}\n Nodes which are either completely inactive or did not follow any other nodes:\n {}".format(masked_list_impressions, no_following, either_or))


def media_correlation_following(ldict, counterr, lkk, nfss, fwid):
    similarity_mat1 = np.zeros((620, 620))
    similarity_mat2 = np.zeros((620, 620))
    similarity_mat3 = np.zeros((620, 620))
    similarity_mat4 = np.zeros((620, 620))

    # for key in fwid.keys():
    #     if key in ldict:
    #         key_vid = ldict[key]
    #         for f in fwid[key]:
    #             if f in ldict:
    #                 foll_video = ldict[f]
    #                 num = set(key_vid) & set(foll_video)
    #                 den = set(key_vid) | set(foll_video)
    #                 similarity_mat[key-1][f-1] = float(len(num)/len(den))*100
    #print(similarity_mat)
    # print (ldict)
    # print ("###################################################################################################")
    # print (ldict.items())
    ldict_with_only_T = ldict.copy()
    ldict_with_only_T = {k: v if not 7 in v and not 8 in v else [] for k,v in ldict_with_only_T.items()}

    ldict_with_FA_and_F = ldict.copy()
    ldict_with_FA_and_F = {o: p if 8 in p and 7 in p else [] for o,p in ldict_with_FA_and_F.items()}

    for elementtt in fwid.keys():
        followees = fwid[elementtt]
        for elementttt in followees:
            num = set(ldict[elementttt]) & set(ldict[elementtt])
            den = set(ldict[elementttt]) | set(ldict[elementtt])
            if len(den) == 0:
                similarity_mat1[elementtt-1][elementttt-1] = 0
            else:
                similarity_mat1[elementtt-1][elementttt-1] = float(len(num)/len(den))*100

    df31 = pd.DataFrame(data=similarity_mat1)
    df31.to_csv('Following_Media_Similarity_{}.csv'.format(counterr), sep=",", header=False, float_format='%.2f', index=False)

    for i in range(similarity_mat1.shape[0]):
        plt.plot(-np.sort(-similarity_mat1[i]))
    plt.title("Media Node-Edge Homogeneity Distribution for Out Degrees")

    plt.xlabel("Nodes")
    plt.ylabel("Media Node-Edge Homogeneity %")
    plt.savefig("Media_Node-Edge_Homogeneity_Following_{}.png".format(counterr))
    plt.close()

    for elementtt in fwid.keys():
        followees = fwid[elementtt]
        for elementttt in followees:
            num = set(ldict_with_only_T[elementttt]) & set(ldict_with_only_T[elementtt])
            den = set(ldict_with_only_T[elementttt]) | set(ldict_with_only_T[elementtt])
            if len(den) == 0:
                similarity_mat2[elementtt-1][elementttt-1] = 0
            else:
                similarity_mat2[elementtt-1][elementttt-1] = float(len(num)/len(den))*100
    df32 = pd.DataFrame(data=similarity_mat2)
    df32.to_csv('Following_Media_Similarity_{}_with_only_T.csv'.format(counterr), sep=",", header=False, float_format='%.2f', index=False)

    for i in range(similarity_mat2.shape[0]):
        plt.plot(-np.sort(-similarity_mat2[i]))
    plt.title("Media Node-Edge Homogeneity Distribution for Out Degrees_with_only_T")

    plt.xlabel("Nodes")
    plt.ylabel("Media Node-Edge Homogeneity %")
    plt.savefig("Media_Node-Edge_Homogeneity_Following_{}_with_only_T.png".format(counterr))
    plt.close()

    for elementtt in fwid.keys():
        followees = fwid[elementtt]
        for elementttt in followees:
            num = set(ldict_with_FA_and_F[elementttt]) & set(ldict_with_FA_and_F[elementtt])
            den = set(ldict_with_FA_and_F[elementttt]) | set(ldict_with_FA_and_F[elementtt])
            if len(den) == 0:
                similarity_mat4[elementtt-1][elementttt-1] = 0
            else:
                similarity_mat4[elementtt-1][elementttt-1] = float(len(num)/len(den))*100

    df34 = pd.DataFrame(data=similarity_mat4)
    df34.to_csv('Following_Media_Similarity_{}_with_FA_and_F.csv'.format(counterr), sep=",", header=False, float_format='%.2f', index=False)

    for i in range(similarity_mat4.shape[0]):
        plt.plot(-np.sort(-similarity_mat4[i]))
    plt.title("Media Node-Edge Homogeneity Distribution for Out Degrees_with_FA_and_F")

    plt.xlabel("Nodes")
    plt.ylabel("Media Node-Edge Homogeneity %")
    plt.savefig("Media_Node-Edge_Homogeneity_Following_{}_with_FA_and_F.png".format(counterr))
    plt.close()
    return None

def media_correlation_follower(ldict, counterr, lkk, nfss, fwed):
    similarity_mat11 = np.zeros((620, 620))
    similarity_mat22 = np.zeros((620, 620))
    similarity_mat33 = np.zeros((620, 620))
    similarity_mat44 = np.zeros((620, 620))
    # for key in fwid.keys():
    #     if key in ldict:
    #         key_vid = ldict[key]
    #         for f in fwid[key]:
    #             if f in ldict:
    #                 foll_video = ldict[f]
    #                 num = set(key_vid) & set(foll_video)
    #                 den = set(key_vid) | set(foll_video)
    #                 similarity_mat[key-1][f-1] = float(len(num)/len(den))*100
    #print(similarity_mat)
    ldict_with_only_T = ldict.copy()
    ldict_with_only_T = {k: v if not 7 in v and not 8 in v else [] for k,v in ldict_with_only_T.items()}

    ldict_with_FA_and_F = ldict.copy()
    ldict_with_FA_and_F = {o: p if 8 in p and 7 in p else [] for o,p in ldict_with_FA_and_F.items()}

    for elementtt in fwed.keys():
        followers = fwed[elementtt]
        for elementttt in followers:
            num = set(ldict[elementttt]) & set(ldict[elementtt])
            den = set(ldict[elementttt]) | set(ldict[elementtt])
            if len(den) == 0:
                similarity_mat11[elementtt-1][elementttt-1] = 0
            else:
                similarity_mat11[elementtt-1][elementttt-1] = float(len(num)/len(den))*100
    df311 = pd.DataFrame(data=similarity_mat11)
    df311.to_csv('Follower_Media_Similarity_{}.csv'.format(counterr), sep=",", header=False, float_format='%.2f', index=False)

    for i in range(similarity_mat11.shape[0]):
        plt.plot(-np.sort(-similarity_mat11[i]))
    plt.title("Media Node-Edge Homogeneity Distribution for In Degrees")

    plt.xlabel("Nodes")
    plt.ylabel("Media Node-Edge Homogeneity %")
    plt.savefig("Media_Node-Edge_Homogeneity_Follower_{}.png".format(counterr))
    plt.close()

    for elementtt in fwed.keys():
        followers = fwed[elementtt]
        for elementttt in followers:
            num = set(ldict_with_only_T[elementttt]) & set(ldict_with_only_T[elementtt])
            den = set(ldict_with_only_T[elementttt]) | set(ldict_with_only_T[elementtt])
            if len(den) == 0:
                similarity_mat22[elementtt-1][elementttt-1] = 0
            else:
                similarity_mat22[elementtt-1][elementttt-1] = float(len(num)/len(den))*100
    df322 = pd.DataFrame(data=similarity_mat22)
    df322.to_csv('Follower_Media_Similarity_{}_with_only_T.csv'.format(counterr), sep=",", header=False, float_format='%.2f', index=False)

    for i in range(similarity_mat22.shape[0]):
        plt.plot(-np.sort(-similarity_mat22[i]))
    plt.title("Media Node-Edge Homogeneity Distribution for In Degrees_with_only_T")

    plt.xlabel("Nodes")
    plt.ylabel("Media Node-Edge Homogeneity %")
    plt.savefig("Media_Node-Edge_Homogeneity_Follower_{}_with_only_T.png".format(counterr))
    plt.close()

    for elementtt in fwed.keys():
        followers = fwed[elementtt]
        for elementttt in followers:
            num = set(ldict_with_FA_and_F[elementttt]) & set(ldict_with_FA_and_F[elementtt])
            den = set(ldict_with_FA_and_F[elementttt]) | set(ldict_with_FA_and_F[elementtt])
            if len(den) == 0:
                similarity_mat44[elementtt-1][elementttt-1] = 0
            else:
                similarity_mat44[elementtt-1][elementttt-1] = float(len(num)/len(den))*100

    df344 = pd.DataFrame(data=similarity_mat44)
    df344.to_csv('Follower_Media_Similarity_{}_with_FA_and_F.csv'.format(counterr), sep=",", header=False, float_format='%.2f', index=False)

    for i in range(similarity_mat44.shape[0]):
        plt.plot(-np.sort(-similarity_mat44[i]))
    plt.title("Media Node-Edge Homogeneity Distribution for In Degrees_with_FA_and_F")

    plt.xlabel("Nodes")
    plt.ylabel("Media Node-Edge Homogeneity %")
    plt.savefig("Media_Node-Edge_Homogeneity_Follower_{}_with_FA_and_F.png".format(counterr))
    plt.close()
    return None
    
def correlation_following(following_dictionary, counterr, dkk, nfss):
    similarity_mat = np.zeros((620, 620))
    # for key in fwid.keys():
    #     if key in ldict:
    #         key_vid = ldict[key]
    #         for f in fwid[key]:
    #             if f in ldict:
    #                 foll_video = ldict[f]
    #                 num = set(key_vid) & set(foll_video)
    #                 den = set(key_vid) | set(foll_video)
    #                 similarity_mat[key-1][f-1] = float(len(num)/len(den))*100
    #print(similarity_mat)
    dicttt = {}
    for k,v in following_dictionary.items():
        inner = {}
        for value in v:
            inner[value] = 1
        dicttt[k] = inner
    # dicttt = {kk:dict(zip(v, [1] * len(v))) for k,v in following_dictionary.items() for kk,vv in following_dictionary.items()}
    for kk, dd in dicttt.items():
        for ikk in dd:
            dd[ikk] = {'weight': 1}

    for i in range(1,621):
        if i not in dicttt.keys():
            dicttt[i]={}
        else:
            pass

    GG = nx.from_dict_of_dicts(dicttt)
    # G.remove_nodes_from(list(nx.isolates(G)))
    AA = nx.to_numpy_matrix(GG, nodelist=sorted(GG.nodes()))
    np.save("following_matrix_{}".format(counterr), AA)

    for elementtt in following_dictionary.keys():
        followees = following_dictionary[elementtt]
        for elementttt in followees:
            try:
                following_dictionary[elementttt]
                num = set(following_dictionary[elementttt]) & set(following_dictionary[elementtt])
                den = set(following_dictionary[elementttt]) | set(following_dictionary[elementtt])
                similarity_mat[elementtt-1][elementttt-1] = float(len(num)/len(den))*100
            except KeyError as e:
                print("Key Error")
                similarity_mat[elementtt-1][elementttt-1] = 0.0

    df3 = pd.DataFrame(data=similarity_mat)
    df3.to_csv('Following_Node_Similarity_{}.csv'.format(counterr), sep=",", header=False, float_format='%.2f', index=False)

    for i in range(similarity_mat.shape[0]):
        plt.plot(-np.sort(-similarity_mat[i]))
    plt.title("Node-Edge Homogeneity Distribution for Out Degrees")

    plt.xlabel("Nodes")
    plt.ylabel("Node-Edge Homogeneity %")
    plt.savefig("Node-Edge_Homogeneity_Following_{}.png".format(counterr))
    plt.close()
    return None

def correlation_follower(follower_dictionary, counterr, dkfrr, nfss):
    similarity_mat = np.zeros((620, 620))
    # for key in fwid.keys():
    #     if key in ldict:
    #         key_vid = ldict[key]
    #         for f in fwid[key]:
    #             if f in ldict:
    #                 foll_video = ldict[f]
    #                 num = set(key_vid) & set(foll_video)
    #                 den = set(key_vid) | set(foll_video)
    #                 similarity_mat[key-1][f-1] = float(len(num)/len(den))*100
    #print(similarity_mat)
    dictt = {}
    for k,v in follower_dictionary.items():
        inner = {}
        for value in v:
            inner[value] = 1
        dictt[k] = inner
    # dictt = {kk:dict(zip(v, [1] * len(v))) for k,v in follower_dictionary.items() for kk,vv in follower_dictionary.items()}
    # print (dictt.items())
    for k, d in dictt.items():
        for ik in d:
            d[ik] = {'weight': 1}

    for i in range(1,621):
        if i not in dictt.keys():
            dictt[i]={}
        else:
            pass
    G = nx.from_dict_of_dicts(dictt)
    # G.remove_nodes_from(list(nx.isolates(G)))
    A = nx.to_numpy_matrix(G, nodelist=sorted(G.nodes()))
    np.save("follower_matrix_{}".format(counterr), A)
    for elementtt in follower_dictionary.keys():
        followers = follower_dictionary[elementtt]
        for elementttt in followers:
            try:
                follower_dictionary[elementttt]
                num = set(follower_dictionary[elementttt]) & set(follower_dictionary[elementtt])
                den = set(follower_dictionary[elementttt]) | set(follower_dictionary[elementtt])
                similarity_mat[elementtt-1][elementttt-1] = float(len(num)/len(den))*100
            except KeyError as e:
                print("Key Error")
                similarity_mat[elementtt-1][elementttt-1] = 0.0

    df3 = pd.DataFrame(data=similarity_mat)
    df3.to_csv('Follower_Node_Similarity_{}.csv'.format(counterr), sep=",", header=False, float_format='%.2f', index=False)

    for i in range(similarity_mat.shape[0]):
        plt.plot(-np.sort(-similarity_mat[i]))
    plt.title("Node-Edge Homogeneity Distribution for In Degrees")

    plt.xlabel("Nodes")
    plt.ylabel("Node-Edge Homogeneity %")
    plt.savefig("Node-Edge_Homogeneity_Follower_{}.png".format(counterr))
    plt.close()
    return None

def namebreaker(filename):
    for char in filename:
        if char == "-":
            filename = filename.replace(char, "")
        elif char == "_":
            filename = filename.replace(char, "")
        else:
            None
    return filename

# def Influence_Calculator(self):

def main():
    with os.scandir(folder_dir) as entries:
        for entry in entries:  # segregate the entries as belonging to credibility list, impressions list and edgelist by altering their filenames to concatenate the date and time into ascending order
            if entry.name[:3] == "cre":
                entry_rename = namebreaker(entry.name)
                if entry.name != entry_rename:
                    entry_rename = folder_dir + entry_rename
                    os.rename(entry, entry_rename)
                credibility_list.append(entry.name)

            elif entry.name[:3] == "imp":
                entry_rename = namebreaker(entry.name)
                if entry.name != entry_rename:
                    entry_rename = folder_dir + entry_rename
                    os.rename(entry, entry_rename)
                impressions_list.append(entry.name)

            elif entry.name[:3] == "edg":
                entry_rename = namebreaker(entry.name)
                if entry.name != entry_rename:
                    entry_rename = folder_dir + entry_rename
                    os.rename(entry, entry_rename)
                edgelist_list.append(entry.name)
            
            elif entry.name[:3] == "fol":
                entry_rename = namebreaker(entry.name)
                if entry.name != entry_rename:
                    entry_rename = folder_dir + entry_rename
                    os.rename(entry, entry_rename)
                follower_list.append(entry.name)

            elif entry.name[:3] == "der":
                entry_rename = namebreaker(entry.name)
                if entry.name != entry_rename:
                    entry_rename = folder_dir + entry_rename
                    os.rename(entry, entry_rename)
                likes_list.append(entry.name)

            elif entry.name[:3] == "lab":
                entry_rename = namebreaker(entry.name)
                if entry.name != entry_rename:
                    entry_rename = folder_dir + entry_rename
                    os.rename(entry, entry_rename)
                labels_list.append(entry.name)

    credibility_list.sort()
    impressions_list.sort()
    edgelist_list.sort()
    follower_list.sort()
    likes_list.sort()
    labels_list.sort()

    # with open("credibilitylistfiles", "wb") as clf:
    #     pickle.dump(credibility_list, clf)
    # with open("impressionlistfiles", "wb") as ilf:
    #     pickle.dump(impressions_list, ilf)
    # with open("derivedlikesfiles", "wb") as dlf:
    #     pickle.dump(likes_list, dlf)

    counterrr = 0
    dictkeys_following = []
    labelkeys = []
    dictkeys_follower = []


    dictkeys = []
    temp = []
    countvideolikes = []
    for i in range(len(edgelist_list)):
        filee = edgelist_list[i]
        G = nx.read_edgelist(folder_dir+filee, nodetype=int)
        G = nx.convert_matrix.to_numpy_matrix(G)
        temp.append(G.shape)
    networkfinalsize = 342

    ld = []
    for fileee in range(len(credibility_list)):
    # for file in range(3):

        cc, ldd = Masker(impressions_list, edgelist_list, credibility_list, follower_list, labels_list, counterrr, dictkeys_following, labelkeys, dictkeys_follower, networkfinalsize)
        # Masker(impressions_list, edgelist_list, credibility_list, follower_list, labels_list, counterrr, dictkeys_following, labelkeys, dictkeys_follower, networkfinalsize)
        counterrr = counterrr + 1
        import json
        with open('Media_Dictionary_{}.json'.format(fileee), 'w') as fp:
            json.dump(ldd, fp)
        ld.append(ldd)
        countvideolikes.append(cc)
    # x = np.linspace(0, len(countvideolikes), len(countvideolikes))
    # number = max(len(l) for l in countvideolikes)
    # for kk in range(1,number):
    #     KKK = []
    #     for kkk in range(len(countvideolikes)):
    #         KKK.append(countvideolikes[kkk][kk])
    #     plt.plot(x, KKK, lw=1.5)
    #     plt.xlim(0.0, 145.0)
    #     plt.xlabel("Timestamps")
    #     plt.ylabel("Number of likes per category")
    #     plt.legend(['Audioless', 'International News', 'Domestic News', 'Political', 'Healthcare', 'Random', 'Misinformation\n(Deep Fake Videos)',
    #                 'Misinformation\n(Doctored Content)', 'Advertisement', 'Sports', 'Movie', 'Education'], loc='center left', bbox_to_anchor=(1, 0.5))
    #     plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()











