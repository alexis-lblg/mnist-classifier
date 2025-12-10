import torch 

class Network:
    def __init__(self): 
        self.layers = [784, 32, 16, 10]
        self.W = [
            torch.randn(self.layers[i+1], self.layers[i]) * (2 / self.layers[i])**0.5
            for i in range(len(self.layers)-1)
            ]
        self.b = [
            torch.zeros(self.layers[i+1])
            for i in range(len(self.layers)-1)
            ]
                               
    def reLU(self, z):
        return torch.clamp(z, min=0.0)
    
    def reLU_derive(self, z):
        return (z > 0).float()
    
    def forward(self, X): 
        X = X.float()
        self.A = [X]  
        self.Z = []

        for i in range(len(self.W)):
            Z = self.A[-1] @ self.W[i].t() + self.b[i] 
            if i != (len(self.W)-1):
                A = self.reLU(Z)
            else:
                A = self.softmax(Z)
            self.Z.append(Z)   
            self.A.append(A)   
        return self.A[-1]
    
    def softmax(self, x): 
        x_shift = x - x.max(dim=1, keepdim=True).values 
        exp = torch.exp(x_shift)
        return exp / exp.sum(dim=1, keepdim=True)
    
    def compute_loss(self, y_pred, y_true):
        return -torch.mean(torch.sum(y_true * torch.log(y_pred + 1e-8), dim=1))
    
    def backward(self, y_true, lr = 0.04): 
        y_pred = self.A[-1]  
        batch_size = y_pred.shape[0]
        
        dZ = (y_pred - y_true.float()) / batch_size  
        
        for i in reversed(range(len(self.W))):
            A_prev = self.A[i]
            
            dW = dZ.t() @ A_prev
            db = dZ.sum(dim=0)
            
            self.W[i] -= lr * dW
            self.b[i] -= lr * db

            if i > 0:
                dA = dZ @ self.W[i]  
                dZ = dA * self.reLU_derive(self.Z[i-1])

    