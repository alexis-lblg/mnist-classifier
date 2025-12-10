from network import Network
from data   import Data
import matplotlib.pyplot as plt
import torch 

if __name__ == '__main__':
    reseau = Network()
    da = Data()
    da.training(128, reseau)
    da.test(reseau, 10000)  