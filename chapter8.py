# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
#import seaborn as sns
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
# import scipy.stats
# from sklearn import preprocessing
import glob

class Variable():
    def __init__(self,file_A,file_B):
        self.A=pd.read_table(file_A,sep=' ',header=None)
        self.B=pd.read_table(file_B,sep=' ',header=None)
        
class Cahpter8():
    def __init__(self,file_A,file_B,rho):
        self.rho=rho
        self.file_A=file_A
        self.file_B=file_B
    # 前向きアルゴリズムの計算を行う関数(引数は得られる出力系列)
    def forward_algorithm(self,data):
        v=Variable(self.file_A,self.file_B)
        temporary=0
        alpha=np.zeros(len(self.rho))
        all_alpha=[]
        for i in range(len(data)):
            temporary=0
            all_alpha.append([])
            if i==0:
                for j in range(len(alpha)):
                    temporary=self.rho[j]*v.B[data[i]][j]
                    all_alpha[i].append(temporary)
            else:
                for j in range(len(alpha)):
                    temporary=0
                    for k in range(len(alpha)):
                        temporary+=all_alpha[i-1][k]*v.A[j][k]
                    all_alpha[i].append(temporary*v.B[data[i]][j])
        answer=0
        for i in range(len(all_alpha[-1])):
            answer+=all_alpha[-1][i]
        print(answer)
        
        return all_alpha
    
    def backward_algorithm(self,data):
        v=Variable(self.file_A,self.file_B)
        temporary=0
        all_beta=[]
        all_beta.append([np.ones(len(self.rho))])
        for i in range(1,len(data)):
            temporary=0
            all_beta.insert(0,[])
            
            
    def viterbi_algorithm(self,data):
        v=Variable(self.file_A,self.file_B)
        temporary=0
        psi=[]
        Psi=[]
        psi.append([])
        for i in range(len(self.rho)):
            temporary=self.rho[i]*v.B[data[0]][i]
            psi[0].append(temporary)
        for i in range(1,len(data)):
            temporary=0
            process=np.zeros(len(psi[i-1]))
            psi.append([])
            Psi.append([])
            for j in range(len(self.rho)):
                for k in range(len(psi[i-1])):
                    process[k]=psi[i-1][k]*v.A[j][k]
                Psi[i-1].append(np.argmax(process))
                temporary=max(process)*v.B[data[i]][j]
                psi[i].append(temporary)
        Psi.append(np.argmax(psi[-1]))
        print(psi)
        print(Psi)
        
        s=[]
        s.insert(0,Psi[-1]+1)
        for i in range(1,len(Psi)):
            s.insert(0,Psi[-i-1][Psi[-1]]+1)
        print(s)
        
    def baum_welch_algorithm(self,data):
        # A=pd.read_table("./variable/chapter8/First_A.txt",sep=' ',header=None)
        # B=pd.read_table("./variable/chapter8/First_B.txt",sep=' ',header=None)
        # rho=pd.read_table("./variable/chapter8/First_rho.txt",sep=' ',header=None)
        v=Variable(self.file_A,self.file_B)
        print(v.A)
        print(v.B)
        print(self.rho)