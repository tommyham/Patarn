import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.mixture
import scipy.stats as stats
from scipy.stats import multivariate_normal,invwishart,wishart
from decimal import Decimal
import seaborn as sns
import pandas as pd
import copy
import math


# クラスタリングで用いるデータセットの生成と読み込みを行うクラス
class Data:
    def __init__(self):
        self.filename="dataset.csv"
        self.N = 500
        self.pi=[0.3,0.2,0.2,0.2,0.1]
        self.mu=[[-3, 4], [3, 4], [0, 0], [-3, -4], [4, -2]]
        self.cov=np.array([[[0.1, -0.4], [-0.4, 0.3]],
                    [[0.8, 0.5], [0.5, 0.6]],
                    [[0.2, 0], [0, 0.8]],
                    [[0.8, 0.2], [0.2, 0.5]],
                    [[0.3, 0], [0, 0.4]]])
    
    def makeDataSet(self,N):
        self.N=N
        data=pd.DataFrame(index=[],columns=["X","Y"])
        for i in range(len(self.mu)):
            temp=np.random.multivariate_normal(self.mu[i], self.cov[i], int(self.pi[i]*self.N))
            temp=pd.DataFrame(temp,columns=["X","Y"])
            temp["class"]=int(i)
            data=pd.concat([data,temp],axis=0)
        data=data.reset_index(drop=True)
        data.to_csv("dataset.csv")
        
        return data
        
    def readDataSet(self):
        data=pd.read_csv(self.filename,index_col=0)
        
        return data

class Gibs:
    def __init__(self):
        self.num=2
        self.a=0.6
        self.cov=[[1,self.a],[self.a,1]]
        self.mu=[0,0]
        self.x=[5.0,-5.0]
        self.epoc1=7
        self.epoc2=200
        
        # 2変量正規分布の95%区間を作図する関数
    def add_ellipse_CI95(self,ax, mean, cov, **kwargs):
        w, v = np.linalg.eig(cov)
        v1 = v[:, np.argmax(w)]
        angle = 180. / np.pi * np.arctan(v1[1]/v1[0])
        width = 2 * np.sqrt(np.max(w) * 5.991)
        height = 2 * np.sqrt(np.min(w) * 5.991)
        e = matplotlib.patches.Ellipse(mean, width, height, angle=angle, **kwargs)
        ax.add_artist(e)
    
    def calculate(self):
        x=copy.deepcopy(self.x)
        answer=[copy.deepcopy(x)]
        for i in range(self.epoc1):
            x[0]=np.random.normal(self.a*x[1],1)
            answer.append(copy.deepcopy(x))
            x[1]=np.random.normal(self.a*x[0],1)
            answer.append(copy.deepcopy(x))

        answer=np.array(answer).T
        x=answer[0]
        y=answer[1]
        fig,ax=plt.subplots(1,1)
        self.add_ellipse_CI95(ax,self.mu, self.cov, fc='none', ls='solid', ec='C0', lw='2.')
        ax.quiver(x[:-1],y[:-1],x[1:]-x[:-1],y[1:]-y[:-1],
                  angles="xy",scale_units="xy",scale=1,color="C0")
        ax.scatter(x,y,marker="o",c="none",edgecolors="C0")
        ax.set_xlim(-6,6)
        ax.set_ylim(-6,6)
        fig.savefig("gibs_process.png",bbox_inches="tight",pad_inches=0.1)

        x=copy.deepcopy(self.x)
        answer=[copy.deepcopy(x)]
        for i in range(self.epoc2):
            x[0]=np.random.normal(self.a*x[1],1)
            answer.append(copy.deepcopy(x))
            x[1]=np.random.normal(self.a*x[0],1)
            answer.append(copy.deepcopy(x))

        answer=np.array(answer).T
        x=answer[0]
        y=answer[1]
        fig,ax=plt.subplots(1,1)
        ax.scatter(x,y,marker="o",c="none",edgecolors="C0")
        self.add_ellipse_CI95(ax,self.mu, self.cov, fc='none', ls='solid', ec='C0', lw='2.')
        ax.set_xlim(-6,6)
        ax.set_ylim(-6,6)
        fig.savefig("gibs_plots.png",bbox_inches="tight",pad_inches=0.1)

class NonParametric:
    # dataはDataFrame型(必須)
    def __init__(self,data):
        self.data_colum=data.columns.values
        self.data=copy.deepcopy(data)
        self.num_patarn=[len(data)]
        
        self.class_num=1
        self.data["class"]=int(0)
        self.class_size=self.data.pivot_table(index=["class"],aggfunc="size")
        
        self.epoc=100
        self.alpha=1.0
        self.beta=1/3
        self.v=15
        self.S=[[0.1,0],[0,0.1]]
        self.average=self.data[self.data_colum].mean()
        self.mu_0=self.average
        self.mu=[self.average for i in range(self.class_num)]
        self.convariance=np.cov(data[self.data_colum].T)
        self.lam=[np.linalg.inv(self.convariance) for i in range(self.class_num)]
        self.prob_new=[]
        self.prob_max=0
    
    # 引数のデータのクラス数が変化したかの確認(変化あり：True、変化なし：False)
    def checkClassNum(self,xk,x_k):
        # 各クラスのデータ数の算出
        class_table=x_k.pivot_table(index=["class"],aggfunc="size")
        
        if sum(class_table.index==xk["class"]):
            return False
        
        return True
    
    def calSb(self,xk):
        dim=len(self.data_colum)
        a=np.array(xk-self.mu_0).reshape(dim,1)
        b=np.array(xk-self.mu_0).reshape(1,dim)
        inv_S_b=np.linalg.inv(self.S)+self.beta/(1+self.beta)*np.dot(a,b)
        
        S_b=np.linalg.inv(inv_S_b)
        
        return np.linalg.det(S_b)**((self.v+1)/2)
        
    
    def calpxnew(self):
        dimention=len(self.data_colum)
        factor=(self.beta/((1+self.beta)*np.pi))**(dimention/2)
        det_S=np.linalg.det(self.S)
        denominator=(det_S**(self.v/2))*math.gamma((self.v+1-dimention)/2)
        
        S_b=self.data[self.data_colum].apply(self.calSb,axis=1)
        
        prob=factor*S_b*math.gamma((self.v+1)/2)/denominator
        
        # for i in range(len(self.data)):
        #     xk=self.data[self.data_colum].loc[i]
        #     average_mu=self.average
        #     a=np.array(xk-average_mu).reshape(dimention,1)
        #     b=np.array(xk-average_mu).reshape(1,dimention)
        #     inv_S_b=np.linalg.inv(self.S)+self.beta/(1+self.beta)*np.dot(a,b)
        #     S_b=np.linalg.inv(inv_S_b)
        #     det_S_b=np.linalg.det(S_b)
            
        #     molecule=(det_S_b**((self.v+1)/2))*math.gamma((self.v+1)/2)
        #     prob_new=factor*molecule/denominator
            
        #     self.prob_new.append(prob_new)
        
        self.prob_new=np.array(prob)
        
        return True
    
    def calpxtheta(self,xk,k):
        dimention=len(self.data_colum)
        all_prob=[]
        for i in range(self.class_num):
            n_i=self.class_size[i]
            mu=np.array(self.mu[i]).reshape(dimention)
            lam=self.lam[i]
            inv_lam=np.linalg.inv(lam)
            prob=multivariate_normal(mu,inv_lam).pdf(xk)
            all_prob.append(prob)
        
        # average_mu=self.average
        # a=np.array(xk-average_mu).reshape(len(self.data_colum),1)
        # b=np.array(xk-average_mu).reshape(1,len(self.data_colum))
        # inv_S_b=np.linalg.inv(self.S)+self.beta/(1+self.beta)*np.dot(a,b)
        # S_b=np.linalg.inv(inv_S_b)
        # det_S=np.linalg.det(self.S)
        # det_S_b=np.linalg.det(S_b)
        
        # factor=(self.beta/((1+self.beta)*np.pi))**(dimention/2)
        # denominator=(det_S**(self.v/2)*((self.v+1)/2-1))
        # molecule=det_S_b**((self.v+1)/2)
        # prob_new=factor*molecule/denominator
        all_prob.append(self.prob_new[k])
        
        return np.array(all_prob)
    
    def calNewClass(self,k):
        data=copy.deepcopy(self.data)
        mu=copy.deepcopy(self.mu)
        lam=copy.deepcopy(self.lam)
        
        xk=data.loc[k]
        x_k=pd.concat([data.loc[:k-1],data.loc[k+1:]])
        
        if self.checkClassNum(xk,x_k):
            mu.pop(int(xk["class"]))
            lam.pop(int(xk["class"]))
            pre_class=x_k["class"]
            new_class=pre_class-(pre_class>xk["class"])*1
            x_k["class"]=new_class
            data["class"]=new_class
        
        probs=self.calpxtheta(xk[self.data_colum],k)
        probs=probs/sum(probs)
        probs=np.cumsum(probs)
        
        diff=probs-np.random.rand()
        new_class=min(np.where(diff>=0)[0])
        xk["class"]=int(new_class)
        data.loc[k]=xk
        
        return data,mu,lam
    
    def calMatrixPandas(self,x):
        dimention=len(self.data_colum)
        
        a=np.array(x).reshape(dimention,1)
        b=np.array(x).reshape(1,dimention)
        
        array=np.dot(a,b)
        
        return pd.Series(array.reshape(array.size))
    
    def calptheta(self,mu_i,mu_c,lam_c,lam_i,v_c,S_q):
        dimention=len(self.data_colum)
        
        gaus=0
        
        para1=np.sqrt(np.linalg.det(lam_c))/(2*np.pi)**(dimention/2)
        
        diff=mu_i-mu_c
        temp=np.dot(lam_c,diff)
        para2=np.exp(-np.dot(diff,temp)/2)
        
        gaus=para1*para2
        
        wishart=0
        
        molecule=0
        para1=np.linalg.det(lam_i)**((self.v-dimention-1)/2)
        inv_S_q=np.linalg.inv(S_q)
        para2=np.exp(-np.trace(np.dot(inv_S_q,lam_i))/2)
        molecule=para1*para2
        
        denominator=0
        para1=2**(self.v*dimention/2)
        para2=np.pi**(dimention*(dimention-1)/4)
        para3=np.linalg.det(S_q)**(self.v/2)
        gamma=[math.gamma((self.v+1-i)/2) for i in range(dimention)]
        para4=math.prod(gamma)
        denominator=para1*para2*para3*para4
        
        wishart=molecule/denominator
        
        return gaus*wishart
    
    def calInvWishart(self,v_c,S_q):
        lam_i=wishart(v_c,S_q).rvs()
        
        return lam_i
    
    def calInvGaus(self,mu_c,lam_c):
        inv_lam_c=np.linalg.inv(lam_c)
        
        mu_i=multivariate_normal(mu_c,inv_lam_c).rvs()
        
        return mu_i
        
    
    def calNewParams(self,data):
        dimention=len(self.data_colum)
        class_table=data.pivot_table(index=["class"],aggfunc="size")
        class_mu=[]
        class_lam=[]
        
        for i in range(len(class_table)):
            n_i=class_table[i]
            class_data=data[data["class"]==i][self.data_colum]
            average=class_data.mean()
            inv_S=np.linalg.inv(self.S)
            
            diff=class_data-average
            cal=diff.apply(self.calMatrixPandas,axis=1)
            para2=cal.sum()
            para2=np.array(para2).reshape(dimention,dimention)
            
            a=np.array(average-self.mu_0).reshape(dimention,1)
            b=np.array(average-self.mu_0).reshape(1,dimention)
            para3=n_i*self.beta/(n_i+self.beta)*np.dot(a,b)
            
            inv_S_q=inv_S+para2+para3
            S_q=np.linalg.inv(inv_S_q)
            v_c=self.v+n_i
            new_lam=wishart(v_c,S_q).rvs()
            # new_lam=np.linalg.inv(invwishart(v_c,S_q).rvs())
            
            mu_c=(n_i*average+self.beta*self.mu_0)/(n_i+self.beta)
            lam_c=(n_i+self.beta)*new_lam
            sigma=np.linalg.inv(lam_c)
            
            new_mu=multivariate_normal(mu_c,sigma).rvs()
            
            class_mu.append(pd.DataFrame(new_mu))
            class_lam.append(pd.DataFrame(new_lam))
        
        return class_mu,class_lam
    
    # 返り値はDecimal型
    def calEwens(self,class_table):
        molecule=0
        class_fc=class_table.apply(lambda s:Decimal(math.factorial(s-1)))
        molecule=Decimal(self.alpha**len(class_table))*np.prod(class_fc)
        
        denominator=0
        arrayD=[Decimal(i) for i in range(int(self.alpha),int(self.alpha+sum(class_table)))]
        denominator=np.prod(arrayD)
        
        return molecule/denominator
    
    # 返り値はDecimal型
    def calLikeHood(self,data,mu,lam):
        class_table=data.pivot_table(index=["class"],aggfunc="size")
        
        para1=self.calEwens(class_table)
        
        prob=[]
        for i in range(len(class_table)):
            data_i=data[data["class"]==i][self.data_colum]
            mu_i=mu[i]
            mu_i=np.array(mu_i).reshape(len(self.data_colum))
            lam_i=lam[i]
            sigma=np.linalg.inv(self.beta*lam_i)
            
            gaus=multivariate_normal(self.mu_0,sigma).pdf(mu_i)
            wish=wishart(self.v,self.S).pdf(lam_i)
            para2=Decimal(gaus*wish)
            
            sigma=np.linalg.inv(lam_i)
            func=lambda s:multivariate_normal(mu_i,sigma).pdf(s)
            gaus=data_i.apply(func,axis=1)
            gaus=[Decimal(i)for i in gaus]
            para3=np.prod(gaus)
            # prob=prob*para2*para3
            prob.append(para2*para3)
        
        return para1*np.prod(prob)
        
        
    def epocProcessing(self,N):
        data_size=len(self.data)
        epoc=N
        count=0
        
        class_num=[]
        prob_max=[]
        for i in range(epoc):
            for k in range(data_size):
                data,mu,lam=self.calNewClass(k)
                mu,lam=self.calNewParams(data)
                prob=self.calLikeHood(data, mu, lam)
                if prob>self.prob_max or i<1:
                # if True:
                    class_table=data.pivot_table(index=["class"],aggfunc="size")
                    self.prob_max=prob
                    self.data=data
                    self.class_num=len(class_table)
                    self.class_size=class_table
                    self.mu=mu
                    self.lam=lam
                    count=0
                else:
                    count+=1
                    pass
            class_num.append(self.class_num)
            prob_max.append(self.prob_max.ln())
            if i in {0,10,50,100}:
                sns.scatterplot(data=self.data,x="X",y="Y",hue="class",legend="brief")
        sns.scatterplot(data=self.data,x="X",y="Y",hue="class",legend="brief")
        
        return class_num,prob_max