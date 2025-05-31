import torch

from utils import *
import torch.nn.functional as F
import scipy
from scipy.spatial.distance import cdist
import torch.nn.functional as F

import numpy as np

import torch.nn as nn

def knn_affinity(image, n_neighbors=[20, 10], distance_weights=[2.0, 0.1]):
    """Computes a KNN-based affinity matrix. Note that this function requires pymatting"""
    try:
        from pymatting.util.kdtree import knn
    except:
        raise ImportError(
            'Please install pymatting to compute KNN affinity matrices:\n'
            'pip3 install pymatting'
        )
    device = image.device
    h, w = image.shape[:2]
    r, g, b = image.reshape(-1, 3).T
    n = w * h
    r, g, b = r.to(device), g.to(device), b.to(device)
    
    x = torch.repeat_interleave(torch.linspace(0, 1, w).to(device), h)
    y = torch.cat([torch.linspace(0, 1, h)] * w).to(device)

    i, j = [], [] 

    for k, distance_weight in zip(n_neighbors, distance_weights):
        f = torch.stack(
            [r, g, b, distance_weight * x, distance_weight * y],
             axis=1,
             out=torch.zeros((n, 5), dtype=torch.float32).to(device)
        ).to(device) 
        
        distances, neighbors = knn(f.cpu().numpy(), f.cpu().numpy(), k=k)
        
        distances = torch.tensor(distances)
        neighbors = torch.tensor(neighbors)

        i.append(torch.repeat_interleave(torch.arange(n), k))
        j.append(neighbors.view(-1))

    ij = torch.cat(i + j)
    ji = torch.cat(j + i)
    coo_data = torch.ones(2 * sum(n_neighbors) * n)

    W = scipy.sparse.csr_matrix((coo_data.cpu().numpy(), (ij.cpu().numpy(), ji.cpu().numpy())), (n, n))
    return torch.tensor(W.toarray())

def rw_affinity(image, sigma=0.033, radius=1):
    """Computes a random walk-based affinity matrix. Note that this function requires pymatting"""
    try:
        from pymatting.laplacian.rw_laplacian import _rw_laplacian
    except:
        raise ImportError(
            'Please install pymatting to compute RW affinity matrices:\n'
            'pip3 install pymatting'
        )
    h, w = image.shape[:2]
    n = h * w
    values, i_inds, j_inds = _rw_laplacian(image, sigma, radius)
    W = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), shape=(n, n))
    return W

def get_diagonal(W, threshold: float=1e-12):
    if not isinstance(W, torch.Tensor):
        W = torch.tensor(W, dtype=torch.float32)

    D = torch.matmul(W, torch.ones(W.shape[1], dtype=W.dtype).to(W.device))
    D[D < threshold] = 1.0  # Prevent division by zero.

    D_diag = torch.diag(D)
    return D_diag

class EigenLoss(nn.Module):
    def __init__(self, eigen_cluster):
        super(EigenLoss, self).__init__()
        
        self.eigen_cluster = eigen_cluster

    def normalized_laplacian(self, L, D):
        D_inv_sqrt = torch.inverse(torch.sqrt(D))
        D_inv_sqrt = D_inv_sqrt.diagonal(dim1=-2, dim2=-1)
        
        D_inv_sqrt_diag = torch.diag_embed(D_inv_sqrt)

        L_norm = torch.bmm(D_inv_sqrt_diag, torch.bmm(L, D_inv_sqrt_diag))
        
        return L_norm

    def batch_trace(self,tensor):
        diagonals = torch.diagonal(tensor, dim1=1, dim2=2)
        trace_values = torch.sum(diagonals, dim=1)
        return trace_values

    def eigen(self, lap, K: int):
        _, eigenvectors_all = torch.linalg.eigh(lap, UPLO='U')
        eigenvectors = eigenvectors_all[:, :, :K]
        eigenvectors = eigenvectors.float()

        for k in range(eigenvectors.shape[0]):
            if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  
                eigenvectors[k] = 0 - eigenvectors[k]
                
        return eigenvectors
    
    def pairwise_distances(self, x, y=None):
        x_norm = (x**2).sum(1).view(-1, 1)

        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)
        
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf)

    def compute_color_affinity(self, image, sigma_c=30, radius=1):
        H, W, _ = image.shape
        # N = H * W
        pixels = image.view(-1, 3).float() / 255.0 
        color_distances = self.pairwise_distances(pixels)
        W_color = torch.exp(-color_distances**2 / (2 * sigma_c**2))
        
        y, x = np.mgrid[:H, :W]
        coords = np.c_[y.ravel(), x.ravel()]
        
        spatial_distances = cdist(coords, coords, metric='euclidean')
        
        W_color[spatial_distances > radius] = 0
        return W_color
    
    def laplacian(self, adj, W):
        adj = (adj * (adj > 0))
        max_values = adj.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        adj = adj / max_values 
        w_combs = W.to(adj.device)
        
        max_values = w_combs.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        w_combs = w_combs / max_values 

        W_comb = w_combs + adj
        D_comb = torch.stack([get_diagonal(w_comb) for w_comb in W_comb])
        L_comb = D_comb - W_comb
        lap = self.normalized_laplacian(L_comb, D_comb)
        return lap
    
    def color_affinity(self, img):
        color = []
        for img_ in img:
            normalized_image = img_ / 255.0 
            pixels = normalized_image.view(-1, 3)
            color_distances = torch.cdist(pixels, pixels, p=2.0)
            color_affinity = torch.exp(-color_distances ** 2 / (2 * (0.1 ** 2)))  
            color.append(color_affinity)
            
        aff_color = torch.stack(color)
        return aff_color
    
    def laplacian_matrix(self, img, image_feat, 
                         image_color_lambda=0, which_color_matrix='knn'):
        threshold_at_zero = True

        if threshold_at_zero:
            image_feat = (image_feat * (image_feat > 0))

        max_values = image_feat.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        image_feat = image_feat / max_values 
        
        if image_color_lambda > 0:
            img_resize = F.interpolate(img, size=(28, 28), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
            max_values = img_resize.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            img_norm = img_resize / max_values
            
            affinity_matrices = []

            for img_norm_b in img_norm:
                if which_color_matrix == 'knn':
                    W_lr = knn_affinity(img_norm_b)
                elif which_color_matrix == 'rw':
                    W_lr = rw_affinity(img_norm_b)
                affinity_matrices.append(W_lr)
            W_color = torch.stack(affinity_matrices).to(image_feat.device)

            W_comb = image_feat + W_color * image_color_lambda
        else:
            W_comb = image_feat

        D_comb = torch.stack([get_diagonal(w_comb) for w_comb in W_comb])
        L_comb = D_comb - W_comb
        lap = self.normalized_laplacian(L_comb, D_comb)
        return lap

    def lalign(self, img, Y, adj_code):
        # if code_neg_torch is None:
        
        size = 14 if Y.shape[1] == 196 else 28
        img = F.interpolate(img, size=(size, size), mode='bilinear', align_corners=False).permute(0,2,3,1)
        
        color_W = self.color_affinity(img)
        nor_adj_lap = self.laplacian(adj_code, color_W)
            
        nor_adj_lap_detach = torch.clone(nor_adj_lap.detach()) 
        eigenvectors = self.eigen(nor_adj_lap_detach, K=self.eigen_cluster) 
        
        return eigenvectors
    
    def forward(self, img, feat, code):
        feat = F.normalize(feat, p=2.0, dim=-1)
        adj_code = torch.bmm(code, code.transpose(1,2))

        eigenvectors = self.lalign(img, feat, adj_code)
        return eigenvectors 
