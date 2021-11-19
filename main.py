# -*- coding: utf-8 -*-
#import seaborn as sns
from chapter8 import Chapter8
from KMeans import KMeans
import numpy as np
import matplotlib.pyplot as plt
import sklearn.mixture
import pandas as pd

# data=[0,1,0]
# file_A="./variable/chapter8/A.txt"
# file_B="./variable/chapter8/B.txt"
# rho=[1/3,1/3,1/3]
# c8=Chapter8(file_A,file_B,rho)
# #c8.forward_algorithm(data)
# #c8.backward_algorithm(data)
# #c8.viterbi_algorithm(data)

# file_A="./variable/chapter8/First_A.txt"
# file_B="./variable/chapter8/First_B.txt"
# rho=[1,0,0]
# c8=Chapter8(file_A,file_B,rho)
# # c8.backward_algorithm(data)
# c8.baum_welch_algorithm(data)
N = 1000
pi=[0.3,0.2,0.2,0.2,0.1]
eye=np.eye(2)
mu=[[-6, 6], [-6, -5], [0, 0], [4, 6], [4, -4]]
# cov=[[np.eye(2)] for i in range(len(mu))]
cov=np.array([[[1.0, -1.0], [-1.0, 2]],
            [[2, 0.4], [0.4, 0.5]],
            [[0.3, 0], [0, 0.8]],
            [[1.2, 0.4], [0.4, 0.5]],
            [[0.2, 0], [0, 0.2]]])
x=pd.DataFrame(index=[],columns=["X","Y"])
for i in range(len(mu)):
    temp=np.random.multivariate_normal(mu[i], cov[i], int(pi[i]*N))
    temp=pd.DataFrame(temp,columns=["X","Y"])
    x=pd.concat([x,temp],axis=0)
    # x.insert(0, column=["X","Y"],temp)
x=x.reset_index(drop=True)
x.to_csv("dataset.csv")
num=5
k=KMeans(x,num)
# fig=plt.figure(figsize=(10,10))
for i in range(20):
    k.Assign()
    k.Draw(i)
    k.newCenter()
k.MakeGif()