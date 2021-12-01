from PIL import Image
from scipy.stats import multivariate_normal
from scipy.special import comb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import sklearn.mixture
import pandas as pd
import random
from decimal import Decimal,ROUND_HALF_UP
import seaborn as sns
import os
import glob
from natsort import natsorted

# k-means法で用いるデータセットの生成と読み込みを行うクラス
class Data:
    def __init__(self):
        self.filename="dataset.csv"
        self.N = 1000
        self.pi=[0.3,0.2,0.2,0.2,0.1]
        self.mu=[[-6, 6], [-6, -5], [0, 0], [4, 6], [4, -4]]
        self.cov=np.array([[[1.0, -1.0], [-1.0, 2]],
                    [[2, 0.4], [0.4, 0.5]],
                    [[0.3, 0], [0, 0.8]],
                    [[1.2, 0.4], [0.4, 0.5]],
                    [[0.2, 0], [0, 0.2]]])
    
    def makeDataSet(self):
        data=pd.DataFrame(index=[],columns=["X","Y"])
        for i in range(len(self.mu)):
            temp=np.random.multivariate_normal(self.mu[i], self.cov[i], int(self.pi[i]*self.N))
            temp=pd.DataFrame(temp,columns=["X","Y"])
            data=pd.concat([data,temp],axis=0)
        data=data.reset_index(drop=True)
        data.to_csv("dataset.csv")
        
        return data
        
    def readDataSet(self):
        data=pd.read_csv(self.filename,index_col=0)
        
        return data

# k-Meansを行うクラス（引数は入力データ(DataFrame型)，クラス数（整数））
class KMeans:
    def __init__(self,data,NumClass,epoc):
        self.data=data
        self.NumClass=NumClass
        self.ClassName=["c"+str(i)for i in range(NumClass)]
        self.epoc=epoc
        
        # データの重心を決定する関数
        centers=pd.DataFrame(index=[],columns=[])
        
        # 直接値を指定する場合
        x=[-5,-1,0,2,5]
        y=[-5,-2,0,1,5]
        # x=[1,2,3,4,5]
        # y=[1,2,3,4,5]
        # x=[-5,0,5]
        # y=[-5,0,5]
        # x=[-5,-3,-1,0,2,3,5]
        # y=[-5,-3,-2,0,1,3,5]
        centers=pd.DataFrame(np.array([x,y]),index=["X","Y"])
        
        # 乱数で初期重心を決める場合
        # x=[]
        # y=[]
        # for i in range(NumClass):
        #     x.append(random.uniform(-10,10))
        #     y.append(random.uniform(-10,10))
        # centers=pd.DataFrame(np.array([x,y]),index=["X","Y"])
        
        # データ点から初期重心を決める場合
        # element=[random.randrange(len(data)) for i in range(NumClass)]
        # for i in range(NumClass):
        #     temp=data.loc[element[i]]
        #     centers=pd.concat([centers,temp],axis=1)
        
        centers=centers.T
        centers=centers.reset_index(drop=True)
        centers["class"]=self.ClassName
        self.centers=centers
        self.initialcenters=centers
        
        self.error=[]
        self.covariance=[[]for i in range(NumClass)]
        self.distribution=[[]for i in range(NumClass)]
        self.decisionBoundary=[[]for i in range(int(comb(NumClass,2)))]
        self.outputfolder="./Chapter10"
        self.folder="/outputKMeans"
        self.folderError="/outputError"
        self.folderAll="/outputALL"
        self.ani=[]
        self.gifimage=[]
    
    def make2DGaussianDistribution(self):
        num=self.NumClass
        # distribution=[[]for i in range(num)]
        x=y=np.arange(-10,10,0.1)
        X,Y=np.meshgrid(x,y)
        pos=np.dstack((X,Y))
        for i in range(num):
            mu=np.array(self.centers.loc[i,["X","Y"]]) # クラスiの平均ベクトル
            sigma=np.array(self.covariance[i]) # クラスiの分散共分散行列
            self.distribution[i]=multivariate_normal(mu,sigma).pdf(pos)
            # fig=plt.figure(figsize=(10,10))
            # ax=fig.add_subplot(111,projection='3d')
            # ax.plot_surface(X,Y,self.distribution[i],cmap="coolwarm",cstride=1,rstride=1)
        
        return True
    
    def makeBoundary(self):
        num=self.NumClass
        d=Data()
        x=y=np.arange(-10,10,0.1)
        X,Y=np.meshgrid(x,y)
        
        count=0
        for i in range(num):
            for j in range(i+1,num):
                self.decisionBoundary[count]=abs((d.pi[i]*self.distribution[i])-(d.pi[j]*self.distribution[j]))
                fig=plt.figure(figsize=(10,10))
                ax=fig.add_subplot(111,projection='3d')
                ax.plot_surface(X,Y,self.decisionBoundary[count],cmap="coolwarm",cstride=1,rstride=1)
                count+=1
        
        return True
    
    # クラス分けしたデータから新しく重心を求める関数
    def newCenter(self):
        num=self.NumClass
        centers=pd.DataFrame(index=[],columns=[])
        # covariance=[[]for i in range(num)]
        for i in range(num):
            cluster=self.data[self.data["class"]==self.ClassName[i]]
            self.covariance[i]=cluster.cov()
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
        self.error.append(distance.min(axis=1).sum())
        self.data["class"]=distance.idxmin(axis=1)
        self.data=self.data.sort_values("class")
        
    def DrawInitialScatter(self):
        fig=plt.figure(figsize=(10,10))
        plt.rcParams['font.family'] = 'Times New Roman'
        ax=fig.add_subplot(111)
        sns.scatterplot(x="X",y="Y",data=self.data,color="black",size=1,markers="^",ax=ax)
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        ax.get_legend().remove()
        ax.tick_params(labelsize=18)
        ax.text(5,-10,"iter="+str(0),fontsize=50)
        ax.set_xlabel("X",fontsize=20)
        ax.set_ylabel("Y",fontsize=20)
        fig.savefig(self.outputfolder+"/InitialScatter.png",dpi=300,bbox_inches="tight",pad_inches=0)
    
    # クラスタリング後の各散布図を描画する関数
    def DrawScatter(self,i):
        fig=plt.figure(figsize=(10,10))
        plt.rcParams['font.family'] = 'Times New Roman'
        ax=fig.add_subplot(111)
        sns.scatterplot(x="X",y="Y",hue="class",data=self.data,size=1,markers="^",ax=ax)
        ax.scatter(self.centers["X"],self.centers["Y"],s=500,marker='^',color="black")
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        ax.get_legend().remove()
        ax.text(5,-10,"iter="+str(i),fontsize=50)
        ax.tick_params(labelsize=18)
        ax.set_xlabel("X",fontsize=20)
        ax.set_ylabel("Y",fontsize=20)
        fig.savefig(self.outputfolder+self.folder+"/iter"+str(i),dpi=300,bbox_inches="tight",pad_inches=0)
        
    # クラスタリング後の量子化誤差をグラフ化する関数
    def DrawError(self,i):
        error=np.array(self.error)
        maximum=Decimal(str(max(error)))
        fig=plt.figure(figsize=(20,10))
        plt.rcParams['font.family'] = 'Times New Roman'
        ax=fig.add_subplot(111)
        x=np.arange(1,len(error)+1,1)
        ax.plot(x,error,marker='.',markersize=40,markeredgewidth=3,markerfacecolor="white")
        ax.set_xlim(0,self.epoc+2)
        maximum=int(maximum.quantize(Decimal('1E3'),rounding=ROUND_HALF_UP))
        maximum=5000
        y_tick=np.arange(0,maximum+1001,1000)
        ax.set_yticks(y_tick)
        ax.set_ylim(0,maximum+1000)
        ax.tick_params(labelsize=20)
        ax.set_xlabel("Epoc",fontsize=40)
        ax.set_ylabel("Error",fontsize=40)
        fig.savefig(self.outputfolder+self.folderError+"/iter"+str(i),dpi=300,bbox_inches="tight",pad_inches=0)
        
    def DrawAll(self,i):
        error=np.array(self.error)
        maximum=Decimal(str(max(error)))
        fig=plt.figure(figsize=(10,15))
        plt.rcParams['font.family'] = 'Times New Roman'
        spec=gridspec.GridSpec(ncols=1, nrows=2,height_ratios=[2,1])
        ax=[]
        ax.append(fig.add_subplot(spec[0]))
        sns.scatterplot(x="X",y="Y",hue="class",data=self.data,size=1,markers="^",ax=ax[0])
        ax[0].scatter(self.centers["X"],self.centers["Y"],s=500,marker='^',color="black")
        ax[0].set_xlim(-10,10)
        ax[0].set_ylim(-10,10)
        ax[0].get_legend().remove()
        ax[0].text(5,-10,"iter="+str(i),fontsize=50)
        ax[0].tick_params(labelsize=18)
        ax[0].set_xlabel("X",fontsize=40)
        ax[0].set_ylabel("Y",fontsize=40)
        
        ax.append(fig.add_subplot(spec[1]))
        x=np.arange(1,len(error)+1,1)
        ax[1].plot(x,error,marker='.',markersize=40,markeredgewidth=3,markerfacecolor="white")
        ax[1].set_xlim(0,self.epoc+2)
        maximum=int(maximum.quantize(Decimal('1E3'),rounding=ROUND_HALF_UP))
        maximum=5000
        y_tick=np.arange(0,maximum+1001,1000)
        ax[1].set_ylim(0,maximum+1000)
        ax[1].set_yticks(y_tick)
        ax[1].tick_params(labelsize=20)
        ax[1].set_xlabel("Epoc",fontsize=40)
        ax[1].set_ylabel("Error",fontsize=40)
        fig.savefig(self.outputfolder+self.folderAll+"/iter"+str(i),dpi=300,bbox_inches="tight",pad_inches=0)
        
        
    
    # 保存したグラフを全て読み込みGifにする関数
    def MakeGif(self,folder,output):
        # fig=plt.figure(figsize=(10,10))
        # ani=animation.ArtistAnimation(self.fig,self.gifimage,interval=300)
        # plt.show()
        # ani.save("kmeans.gif")
        # self.ani.save("kmeans.gif",writer="pillow")
        files=glob.glob(folder+"/*.png")
        files=natsorted(files)
        images=list(map(lambda file:Image.open(file),files))
        images[0].save(output,save_all=True,append_images=images[1:],duration=400,loop=0)
        