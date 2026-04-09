import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    return orth_vec

def generate_etf_vector(in_channels, num_classes):

    orth_vec = generate_random_orthogonal_matrix(in_channels, num_classes)

    i_nc_nc = torch.eye(num_classes)
    one_nc_nc = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))

    etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                        math.sqrt(num_classes / (num_classes - 1)))
    
    return etf_vec
