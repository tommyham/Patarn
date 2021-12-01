# -*- coding: utf-8 -*-
#import seaborn as sns
from chapter8 import Chapter8
# from KMeans import KMeans,Data
from ConvecCluster import ConvecCluster,Data
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

d=Data()
# data=d.makeDataSet()
data=d.readDataSet()
num=5
epoc=5000
# k=KMeans(data,num,epoc)
# k.DrawInitialScatter()
# for i in range(1,epoc+1):
#     k.Assign()
#     k.DrawScatter(i)
#     k.DrawError(i)
#     k.DrawAll(i)
#     k.newCenter()
#     # k.make2DGaussianDistribution()
#     # k.makeBoundary()
# k.MakeGif(k.outputfolder+k.folder,k.outputfolder+"/kmeans.gif")
# k.MakeGif(k.outputfolder+k.folderAll,k.outputfolder+"/kmeansall.gif")
# error=pd.DataFrame(np.array(k.error))
# error.to_csv("error.csv",mode="a",index=False)

size=[]
c=ConvecCluster(data)
c.DrawInitialScatter(0)
c.pxw()
for i in range(1,epoc+1):
    c.newPi()
    c.error()
    size.append(len(c.pi[c.pi>0]))
    if i in {1,5,100,1000,2000,3000,4000,5000,10000,15000,20000,25000,30000}:
        c.DrawScatter(i)
c.DrawLikehood()
print(sorted(c.pi,reverse=True)[:5])
pi=c.pi[c.pi>0]
print(sum(pi))