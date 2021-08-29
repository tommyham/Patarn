# -*- coding: utf-8 -*-
#import seaborn as sns
from chapter8 import Chapter8

data=[0,1,0]
file_A="./variable/chapter8/A.txt"
file_B="./variable/chapter8/B.txt"
rho=[1/3,1/3,1/3]
c8=Chapter8(file_A,file_B,rho)
#c8.forward_algorithm(data)
#c8.backward_algorithm(data)
#c8.viterbi_algorithm(data)

file_A="./variable/chapter8/First_A.txt"
file_B="./variable/chapter8/First_B.txt"
rho=[1,0,0]
c8=Chapter8(file_A,file_B,rho)
# c8.backward_algorithm(data)
c8.baum_welch_algorithm(data)