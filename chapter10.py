import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GMM

class Clustering:
    def __init__(self):
        pass
    
    def gmm(self):
        N = 300
        x = np.concatenate([np.random.multivariate_normal([-2, 0], np.eye(2), N / 3),
                       np.random.multivariate_normal([0, 5], np.eye(2), N / 3),
                       np.random.multivariate_normal([2, 3], np.eye(2), N / 3)])
        
        return x
