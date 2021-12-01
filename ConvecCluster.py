from PIL import Image
from scipy.stats import multivariate_normal
from scipy.special import comb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
import sklearn.mixture
import pandas as pd
import random
from decimal import Decimal,ROUND_HALF_UP
import seaborn as sns
import os
import glob
from natsort import natsorted


# 凸クラスタリング法で用いるデータセットの生成と読み込みを行うクラス
class Data:
    def __init__(self):
        self.filename="dataset.csv"
        self.N = 1000
        # self.pi=[0.4,0.2,0.2,0.1,0.1]
        # self.mu=[[-6, 6], [-5, -7], [0, 0], [4, 7], [5, -5]]
        # self.cov=np.array([[[0.8, -1.0], [-1.0, 0.2]],
        #             [[1.2, 0.3], [0.3, 0.4]],
        #             [[0.5, 0.3], [0.3, 1]],
        #             [[0.8, 0.6], [0.6, 0.6]],
        #             [[0.5, 0], [0, 0.5]]])
        
        self.pi=[0.3,0.2,0.2,0.1,0.1,0.1]
        self.mu=[[5, -4], [-2.5, -6], [-5, -4], [0, 0], [1, 6],[-2,7]]
        self.cov=np.array([[[1, 0.8], [0.8, 0.1]],
                    [[0.4, -0.2], [-0.2, 0.8]],
                    [[0.4, -0.1], [-0.1, 0.6]],
                    [[0.6, 0], [0, 0.6]],
                    [[0.5, 0], [0, 0.5]],
                    [[0.3,0],[0,0.3]]])
    
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

# 凸クラスタリング法
class ConvecCluster:
    # 初期値設定
    def __init__(self,data):
        self.data=data
        self.size=len(data)
        self.sigma=4 # 分散の初期値
        self.distance=[] #ユークリッドの距離の二乗の配列
        self.gausian=[] # f_ik
        self.pi=np.ones(self.size)/self.size # 事前確率の初期値
        self.pis=self.pi.copy() # 試行毎の事前確率を入れる配列
        self.threshold=5e-4 # 事前確率を強制で0にする閾値
        self.likehood=0 # 対数尤度
        self.likehoods=[] # 試行毎の対数尤度
        self.output="./chapter10/ConvecCluster" # クラスタリング後の画像を入れるフォルダ
        self.outputscatter=self.output+"/Scatter" # クラスタ結果を入れるフォルダ
    
    # 確率密度関数の計算(2次元のみ)
    def pxw(self):
        gridx=np.meshgrid(self.data["X"],self.data["X"])
        distanceX=gridx[0]-gridx[1]
        gridy=np.meshgrid(self.data["Y"],self.data["Y"])
        distanceY=gridy[0]-gridy[1]
        self.distance=distanceX**2+distanceY**2
        self.gausian=np.exp(-self.distance/(2*self.sigma))/(2*np.pi*self.sigma)
        
        return True
    
    # 事前確率の更新値を計算する関数
    def newPi(self):
        molecule=self.gausian*self.pi
        denominator=np.sum(molecule,axis=1)
        newPi=(np.sum(molecule.T/denominator,axis=1)/self.size)
        newPi=np.where(newPi<self.threshold,0,newPi)
        newPi=newPi/sum(newPi)
        self.pis=np.vstack([self.pis,newPi])
        self.pi=newPi
        
        return True
    
    # 対数尤度を計算する関数
    def error(self):
        self.likehood=sum(np.log(np.dot(self.gausian,self.pi)))
        self.likehoods.append(self.likehood)
        # print(self.likehood)
        
        return True
    
    def DrawInitialScatter(self,i):
        xmin=-10
        xmax=10
        fig=plt.figure(figsize=(10,10))
        plt.rcParams['font.family'] = 'Times New Roman'
        ax=fig.add_subplot(111)
        sns.scatterplot(x="X",y="Y",data=self.data,color="black",size=1,markers="^",ax=ax)
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(xmin,xmax)
        ax.get_legend().remove()
        ax.text(3,-10,"iter="+str(i),fontsize=40)
        ax.tick_params(labelsize=18)
        ax.set_xlabel("X",fontsize=20)
        ax.set_ylabel("Y",fontsize=20)
        fig.savefig(self.output+"/initial.png",dpi=300,bbox_inches="tight",pad_inches=0)
        
        return True
    
    def DrawScatter(self,i):
        ppi=72
        xmin=-10
        xmax=10
        size=self.sigma*3
        size=self.sigma*1.5
        
        pi=pd.DataFrame(self.pi)
        centers=self.data[pi[0]>0]
        pi=pi[pi[0]>0]
        pi=pi*50
        fig=plt.figure(figsize=(10,10))
        plt.rcParams['font.family'] = 'Times New Roman'
        ax=fig.add_subplot(111)
        sns.scatterplot(x="X",y="Y",data=self.data,color="black",size=1,markers="^",ax=ax)
        
        # x軸をベースに計算,y軸でも同じ。アスペクト比が違えばおかしくなる
        ax_length=ax.bbox.get_points()[1][0]-ax.bbox.get_points()[0][0]
        # dpi単位の長さをポイント単位に変換(dpiからインチ単位にし、インチからポイント単位に変換)
        ax_point = ax_length*ppi/fig.dpi
        # x軸の実スケールをポイント単位に変換
        xsize=xmax-xmin
        fact=ax_point/xsize
        # scatterのマーカーサイズは直径のポイントの二乗を描くため、実スケールの半径をポイントに変換し直径にしておく
        size*=2*fact
        
        ax.scatter(centers["X"],centers["Y"],s=size**2,facecolor="None",linewidths=pi[0],edgecolors="red")
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(xmin,xmax)
        ax.get_legend().remove()
        ax.text(3,-10,"iter="+str(i),fontsize=40)
        ax.tick_params(labelsize=18)
        ax.set_xlabel("X",fontsize=20)
        ax.set_ylabel("Y",fontsize=20)
        fig.savefig(self.outputscatter+"/iter"+str(i),dpi=300,bbox_inches="tight",pad_inches=0)
        
        return True
    
    def DrawLikehood(self):
        fig=plt.figure(figsize=(40,10))
        plt.rcParams['font.family'] = 'Times New Roman'
        ax=fig.add_subplot(111)
        x=np.arange(0,len(self.likehoods))
        ax.plot(x,self.likehoods)
        ax.set_xlim(0,4000)
        ax.set_ylim(-5000,-4000)
        ax.tick_params(labelsize=18)
        ax.set_xlabel("Epoc",fontsize=20)
        ax.set_ylabel("Error",fontsize=20)