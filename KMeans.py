from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sklearn.mixture
import pandas as pd
import random
import seaborn as sns
import os
import glob
from natsort import natsorted

# k-Meansを行うクラス（引数は入力データ(DataFrame型)，クラス数（整数））
class KMeans:
    def __init__(self,data,NumClass):
        self.data=data
        self.NumClass=NumClass
        self.ClassName=["c"+str(i)for i in range(NumClass)]
        
        # データの重心を決定する関数
        element=[random.randrange(len(data)) for i in range(NumClass)]
        centers=pd.DataFrame(index=[],columns=[])
        for i in range(NumClass):
            temp=data.loc[element[i]]
            centers=pd.concat([centers,temp],axis=1)
        centers=centers.T
        centers=centers.reset_index(drop=True)
        centers["class"]=self.ClassName
        self.centers=centers
        # self.fig=plt.figure(figsize=(10,10))
        # self.ax=self.fig.add_subplot(111)
        self.folder="./outputKMeans"
        self.ani=[]
        self.gifimage=[]
    
    # クラス分けしたデータから新しく重心を求める関数
    def newCenter(self):
        num=self.NumClass
        centers=pd.DataFrame(index=[],columns=[])
        for i in range(num):
            cluster=self.data[self.data["class"]==self.ClassName[i]]
            centers=pd.concat([centers,cluster.mean()],axis=1)
        centers=centers.T
        centers=centers.reset_index(drop=True)
        centers["class"]=self.ClassName
        self.centers=centers
    
    # 各データにクラスを割り当てる関数(引数は各クラスの重心座標)
    def Assign(self):
        distance=pd.DataFrame(index=[],columns=[])
        for i in range(self.NumClass):
            distance[self.ClassName[i]]=self.data[["X","Y"]].sub(np.array(self.centers.loc[i,["X","Y"]])).pow(2).sum(1).pow(0.5)
        self.data["class"]=distance.idxmin(axis=1)
        self.data=self.data.sort_values("class")
    
    def Draw(self,i):
        fig=plt.figure(figsize=(10,10))
        ax=fig.add_subplot(111)
        image=sns.scatterplot(x="X",y="Y",hue="class",data=self.data,size=1,markers="^",ax=ax)
        ax.scatter(self.centers["X"],self.centers["Y"],marker='^',color="black")
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        ax.get_legend().remove()
        ax.text(5,-10,"iter="+str(i),fontfamily="Times New Roman",fontsize=50)
        fig.savefig(self.folder+"/iter"+str(i))
    
    def MakeGif(self):
        # fig=plt.figure(figsize=(10,10))
        # ani=animation.ArtistAnimation(self.fig,self.gifimage,interval=300)
        # plt.show()
        # ani.save("kmeans.gif")
        # self.ani.save("kmeans.gif",writer="pillow")
        files=glob.glob(self.folder+"/*.png")
        files=natsorted(files)
        images=list(map(lambda file:Image.open(file),files))
        images[0].save("kmeans.gif",save_all=True,append_images=images[1:],duration=400,loop=0)
        