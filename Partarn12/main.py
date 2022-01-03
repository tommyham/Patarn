import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.mixture
import pandas as pd
import copy
import math
import Clustering as cl

g=cl.Gibs()
g.calculate()


d=cl.Data()
# data=d.makeDataSet()
data=d.readDataSet()
fig,ax=plt.subplots(1,1)
sns.scatterplot(data=data,x="X",y="Y",ax=ax,legend="brief")
ax.set_xlim(-6,6)
ax.set_ylim(-6,6)

# nonp=cl.NonParametric(data)
# for i in range(10):
# nonp.calpxnew()
# nonp.upDataClass()
# nonp.upDataParams()