import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.mixture
import pandas as pd
import copy
import math
import Clustering as cl

# g=cl.Gibs()
# g.calculate()


d=cl.Data()
# data=d.makeDataSet(100)
data=d.readDataSet()
fig,ax=plt.subplots(1,1)
sns.scatterplot(data=data,x="X",y="Y",ax=ax,legend="brief")
ax.set_xlim(-6,6)
ax.set_ylim(-6,6)
mu=[]
lam=[]
class_num=data.pivot_table(index=["class"],aggfunc="size")
for i in range(len(class_num)):
    mu_i=data[data["class"]==i][{"X","Y"}].mean()
    mu.append(np.array(mu_i))
    sigma=data[data["class"]==i][{"X","Y"}].cov()
    lam.append(np.linalg.inv(sigma))
no=cl.NonParametric(data[{"X","Y"}])
prob=no.calLikeHood(data, mu, lam)
print(prob,prob.ln())
data["class"]=0
mi=[]
lam=[]
mu.append(np.array(data[{"X","Y"}].mean()))
sigma=data[{"X","Y"}].cov()
lam.append(np.linalg.inv(sigma))
prob=no.calLikeHood(data, mu, lam)
print(prob,prob.ln())

nonp=cl.NonParametric(data[{"X","Y"}])
nonp.calpxnew()
class_num,prob=nonp.epocProcessing(100)
# print(class_num,prob)
# nonp.upDataClass()
# nonp.upDataParams()