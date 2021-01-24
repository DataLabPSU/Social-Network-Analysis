import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


df11 = pd.read_csv("Data/impressionsnew20190405010722.txt", sep=" ", header=None)
df22 = pd.read_csv("Data/credibilitynew20190405010722.txt", sep=" ", header=None)
df2 = pd.read_csv("Data/followerlistnew20190405010722.txt",sep=" ", header=None)
followersonlylist = []
for i in df2[0]:
    followersonlylist.append(i)
followersonlylist.sort()
followersonlylist = set(followersonlylist)
followersonlylist = sorted(followersonlylist)
df221 = df22[0].tolist()
df222 = df22[1].tolist()

df111 = df11[0].tolist()
df112 = df11[2].tolist()

dic = {}
for each in followersonlylist:
    temp = []
    filtered = filter(lambda i: df221[i] == each, range(len(df221)))
    for element in list(filtered):
        temp.append(df222[element])
    temp = list(set(temp))
    temp.sort()
    dic[int(each)] = temp

dictt = {}
for each in followersonlylist:
    temp = []
    filtered = filter(lambda i: df111[i] == each, range(len(df111)))
    for element in list(filtered):
        temp.append(df112[element])
    temp = list(set(temp))
    temp.sort()
    dictt[int(each)] = temp

values_followers = np.array(list(dictt.values()))
values_credibility = np.array(list(dic.values()))
print (values_followers.shape)

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(values_followers, values_credibility)  # perform linear regression
Y_pred = linear_regressor.predict(values_followers)  # make predictions
print(linear_regressor.score(values_followers, values_credibility))
# print (dic.keys().tolist())
# print (df2[0])
plt.scatter(values_followers, values_credibility)
plt.plot(values_followers, Y_pred, color = 'red')
plt.title("# of Followers Vs. Credibility Score")
plt.xlabel("# of Followers per Node")
plt.ylabel("Credibility Score")
plt.savefig("Followers_CredibilityScore.png")
plt.show()
