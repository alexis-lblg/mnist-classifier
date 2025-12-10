import gzip
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from network import Network

class Data:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.base_dir, "data", "mnist.pkl.gz")

        with gzip.open(self.data_path, "rb") as f:
            self.training_data, self.validation_data, self.test_data = pickle.load(f, encoding="latin1")
        self.image_t, self.label_t = self.training_data
        self.image_tes, self.label_tes = self.test_data
        self.label_tes2 = self.label_tes
        self.label_t = torch.tensor(self.change_format(self.label_t))
        self.label_tes = torch.tensor(self.change_format(self.label_tes))
    
    def training(self, taille_B, res): 
        losses = []
        for epoch in range(5):  
            for i in range(0, len(self.image_t), taille_B):

                batch_x = torch.from_numpy(self.image_t[i:i + taille_B]).float()
                batch_y = self.label_t[i:i + taille_B]
                
                res.forward(batch_x)
                
                if i % (taille_B * 10) == 0:
                    loss = res.compute_loss(res.A[-1], batch_y)
                    losses.append(loss.item())
                
                res.backward(batch_y)

        self.graph_loss(losses)

    def graph_loss(self, losses):
        plt.plot(losses)
        plt.title("loss during the training")
        plt.xlabel("Number of steps")
        plt.ylabel("Loss")
        plt.show()

    def test(self, res, t=20):
        batch_x = torch.from_numpy(self.image_tes[0:t]).float()
        R = res.forward(batch_x)
        
        correct = 0
        for i in range(len(R)):
            predite = torch.argmax(R[i])
            vrai = self.label_tes2[i]
            print(f"Predite: {predite}, Réelle: {vrai}")
            if predite == vrai:
                correct += 1
        
        prec = correct / len(R) * 100
        print(f"Précision: {prec:.2f}%")
        return prec
    
    def change_format(self, label):
        temp = []
        for element in label:
            temp_2 = []
            for i in range(10):
                if element == i:
                    temp_2.append(1)
                else:
                    temp_2.append(0)
            temp.append(temp_2)
        return temp