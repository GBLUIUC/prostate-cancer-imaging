from unicodedata import numeric
import torch 
import numpy as np

def quadratic_weighted_kappa(probs, target, n_classes=6, epsilon=1e-10):
    
    W = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            W[i][j] = np.power(i - j, 2)
    
    # Calculate the Weights, Observed, and Expected matrices
    W = torch.from_numpy(W)
    O = torch.matmul(torch.trasnpose(target, 0, 1), probs)
    E = torch.matmul(target.sum(dim=0).view(-1,1), probs.sum(dim=0).view(1,-1)) / O.sum()
    
    numerator = (W * O).sum()
    denominator = (W * E).sum() + epsilon

    return  numerator / denominator