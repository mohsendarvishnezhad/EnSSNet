"""

In the Name of Allah
Mohsen Darvishnezhad – 2025/05/06
SP-EnSSNet-RF with Forward Feature Selection:
The Proposed Adaptive Rollback-Based Mutual Information Feature Selection 

"""

import numpy as np
import random
import time
import cv2
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as transforms
from torchvision import models
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    mutual_info_score,
    )
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.init as init
# --------------------------
# 1. Generate Synthetic PolSAR Data
# --------------------------

# Load PolSAR data and GTM
data = scipy.io.loadmat('/PolSAR.mat')
gtm = scipy.io.loadmat('/GTM.mat')


PolSAR = data['PolSAR']  # PolSAR image data
gt = gtm['GTM']  # GTM

# Display one band of PolSAR data (e.g., Band 1)
plt.figure(figsize=(8, 6))
plt.imshow(PolSAR[:, :, 0], cmap='gray')
plt.colorbar()
plt.title("PolSAR Band 1")
plt.show()

# Display the GTM
plt.figure(figsize=(8, 6))
plt.imshow(gt, cmap='jet')  # 'jet' provides better visualization for categorical data
plt.colorbar()  # Add a color legend
plt.title("GTM")
plt.show()


# --------------------------
# 2. Patch Extraction and Preprocessing
# --------------------------
def extract_patch(PolSAR, center, patch_size):
    """Extract a cube patch from the PolSAR given the center and patch size."""
    h, w, _ = PolSAR.shape
    half = patch_size // 2
    c_i, c_j = center
    # Limit to image boundaries
    start_i = max(c_i - half, 0)
    end_i = min(c_i + half, h)
    start_j = max(c_j - half, 0)
    end_j = min(c_j + half, w)
    patch = PolSAR[start_i:end_i, start_j:end_j, :]
    # If patch size is less than required, pad it
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        patch = cv2.copyMakeBorder(patch,
                                   top=half - (c_i - start_i),
                                   bottom=half - (end_i - c_i),
                                   left=half - (c_j - start_j),
                                   right=half - (end_j - c_j),
                                   borderType=cv2.BORDER_REFLECT)
    return patch

def apply_pca(patch, n_components=3):
    """Apply PCA to reduce the Polarimetric dimension of the patch to n_components."""
    h, w, bands = patch.shape
    patch_2d = patch.reshape(-1, bands)
    pca = PCA(n_components=n_components)
    patch_reduced = pca.fit_transform(patch_2d)
    patch_reduced = patch_reduced.reshape(h, w, n_components)
    # Normalize to [0, 1]
    patch_reduced = (patch_reduced - patch_reduced.min()) / (patch_reduced.max() - patch_reduced.min() + 1e-8)
    return patch_reduced


# --------------------------
# 3. Data Augmentation for Generating Two Views
# --------------------------
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(28),  # Crop to a smaller size
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

def generate_two_views(patch):
    """
    Generate two different augmented views from the input patch (numpy array).
    The patch is assumed to be PCA-reduced and then augmented.
    """
    patch_uint8 = (patch * 255).astype(np.uint8)
    view1 = data_transforms(patch_uint8)
    view2 = data_transforms(patch_uint8)
    return view1, view2

# --------------------------
# 4. Calculate Difficulty of Each Patch
# --------------------------


# Function to calculate difficulty for a single pixel based on the provided formula
def calculate_pixel_difficulty(H_i, alpha_i):
    # Calculate difficulty using the formula
    difficulty = np.sqrt(H_i**2 + (1 - np.abs((alpha_i - 60) / 60))**2)
    return difficulty

# Function to calculate difficulty for the entire patch
def calculate_patch_difficulty(patch):
    height, width, _ = patch.shape
    difficulties = []

    # Loop through every pixel in the patch
    for i in range(height):
        for j in range(width):
            # Get H_i from the first band (H_i is the value at the first band of the pixel)
            H_i = patch[i, j, 0]  # Value from the first band (H_i)

            # Get alpha_i from the second band (alpha_i is the value at the second band of the pixel)
            alpha_i = patch[i, j, 1]  # Value from the second band (α_i)

            # Calculate difficulty for this pixel
            pixel_difficulty = calculate_pixel_difficulty(H_i, alpha_i)
            difficulties.append(pixel_difficulty)

    # Calculate the average difficulty for the patch
    patch_difficulty = np.mean(difficulties)
    return patch_difficulty



# --------------------------
# 6. Self-Supervised Dataset Definition
# --------------------------

class PolSARSelfSupervisedFullDataset(Dataset):
    def __init__(self, PolSAR, patch_size=32):
        """
        Extract patches from the entire PolSAR image, one for each pixel.
        """
        self.PolSAR = PolSAR
        self.patch_size = patch_size
        self.height, self.width, _ = PolSAR.shape

    def __len__(self):
        # Return total number of pixels in the image
        return self.height * self.width

    def __getitem__(self, idx):
        # Convert the 1D index to 2D coordinates (i, j)
        i = idx // self.width
        j = idx % self.width
        center = (i, j)

        # Extract a patch centered at (i, j)
        patch = extract_patch(self.PolSAR, center, self.patch_size)
        # Apply PCA to reduce Polarimetric bands to 3 for compatibility with EfficientNet-B0
        patch_pca = apply_pca(patch, n_components=3)
        # Generate two different augmented views of the patch
        view1, view2 = generate_two_views(patch_pca)
        return view1, view2


# --------------------------
# 7. Deep Curriculum Algorthm
# --------------------------

class CurriculumBatchSampler(Sampler):
    def __init__(self, difficulties, batch_size):
        """
        A sampler that generates batches ordered from easy to hard based on difficulty scores.

        Args:
            difficulties (list or array): Difficulty score for each sample (lower = easier).
            batch_size (int): Number of samples in each mini-batch.
        """
        self.difficulties = np.array(difficulties)
        self.batch_size = batch_size

        # Sort all sample indices based on difficulty (easy to hard)
        self.sorted_indices = np.argsort(self.difficulties)

    def __iter__(self):
        """
        Yield batches in curriculum learning order: easy to hard.
        """
        for i in range(0, len(self.sorted_indices), self.batch_size):
            batch = self.sorted_indices[i:i + self.batch_size]
            # Sort each batch internally by difficulty (just to be sure)
            batch = sorted(batch, key=lambda idx: self.difficulties[idx])
            yield batch

    def __len__(self):
        """
        Total number of batches.
        """
        return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size


# --------------------------
# 8. Define the EnSSNet Model Based on EfficientNet-B0 and GCN
# --------------------------

#   1- Build sparse normalized adjacency (per Batch)

def build_sparse_knn_adj_tensor(features, k=49):
    """
    Build a sparse normalized adjacency matrix for each sample in the batch.
    Args:
        features: Tensor of shape (B, C, M, M)
        k: number of neighbors per node
    Returns:
        List of length B of sparse adjacency matrices, each of shape (N, N) where N=M*M
    """
    B, C, M, _ = features.shape
    num_nodes = M * M
    device = features.device
    adj_list = []

    for b in range(B):
        # 1. Flatten to (num_nodes, C)
        x = features[b].view(C, -1).transpose(0, 1)  # (N, C)

        # 2. Pairwise distances
        dist = torch.cdist(x, x)             # (N, N)
        dist.fill_diagonal_(float('inf'))

        # 3. k-NN indices
        knn = dist.topk(k, largest=False).indices  # (N, k)

        # 4. Build COO indices
        row = torch.arange(num_nodes, device=device).unsqueeze(1).repeat(1, k).flatten()  # (N*k,)
        col = knn.flatten()                                                               # (N*k,)
        vals = torch.ones_like(row, dtype=torch.float, device=device)

        # 5. Build and coalesce sparse adjacency
        A_sparse = torch.sparse_coo_tensor(
            indices=torch.stack([row, col], dim=0),
            values=vals,
            size=(num_nodes, num_nodes)
        ).coalesce()  # merge duplicates & sort

        # 6. Symmetric normalization: A_norm = D^-1/2 A D^-1/2
        deg = torch.sparse.sum(A_sparse, dim=1).to_dense()          # (N,)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # multiply each A_sparse.value by deg_inv_sqrt[i]*deg_inv_sqrt[j]
        idx = A_sparse.indices()
        norm_vals = A_sparse.values() * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]
        A_norm = torch.sparse_coo_tensor(idx, norm_vals, size=A_sparse.size()).coalesce()

        adj_list.append(A_norm)

    return adj_list


# 2- Batch-level GCN Layer

class BatchGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k_neighbors, patch_size):
        super().__init__()
        self.k = k_neighbors
        self.patch_size = patch_size  # M
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.batchnorm = nn.BatchNorm2d(out_channels) 
        self.dropout = nn.Dropout2d(0.2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # He initialization for conv1x1 weights
        init.kaiming_normal_(self.conv1x1.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1x1.bias is not None:
            nn.init.constant_(self.conv1x1.bias, 0)

    def forward(self, X):
        """
        X: Tensor of shape (B, C, M, M)
        returns: Tensor of shape (B, out_channels)
        """
        B, C, M, _ = X.shape
        num_nodes = M * M

        # 1. Build sparse normalized adjacencies for whole batch
        adj_list = build_sparse_knn_adj_tensor(X, k=self.k)  # list of B sparse (N,N)

        out = []
        for b in range(B):
            A_norm = adj_list[b]               # sparse (N, N)
            x = X[b].view(C, -1).transpose(0, 1)  # (N, C)

            # 2. GCN propagation: Z = A_norm @ x  -> (N, C)
            Z = torch.sparse.mm(A_norm, x)     # (N, C)

            # 3. Reshape back to (1, C, M, M)
            Z = Z.transpose(0, 1).view(1, C, M, M)

            # 4. 1×1 conv + ReLU + global avg pool → (1, out_channels)
            Fg = self.conv1x1(Z)
            Fg = F.relu(Fg)
            Fg = self.batchnorm(Fg)
            Fg = self.dropout(Fg)
            Fg = self.pool(Fg).view(1, -1)

            out.append(Fg)

        return torch.cat(out, dim=0)  # (B, out_channels)


# 3. Self Supervised Model With GCN-Efficientnet Net-B0

class EnSSNetWithGCNModel(nn.Module):
    def __init__(self, gcn_in_channels, gcn_out_channels, patch_size, k_neighbors=49):
        super().__init__()
        # EfficientNet-B0 backbone up to block 10
        self.backbone = torch.hub.load(
            'rwightman/gen-efficientnet-pytorch',
            'efficientnet_b0',
            pretrained=False
        )
        self.stem = nn.Sequential(
            self.backbone.conv_stem, self.backbone.bn1, self.backbone.act1
        )
        self.blocks = nn.Sequential(*self.backbone.blocks[:10])

        # Batch-level GCN
        self.batch_gcn = BatchGCNLayer(
            in_channels=gcn_in_channels,
            out_channels=gcn_out_channels,
            k_neighbors=k_neighbors,
            patch_size=patch_size
        )

    def forward(self, x):
        # x: (B, C, M, M)
        # 1. CNN path
        fe = self.stem(x)
        fe = self.blocks(fe)
        fe = fe.flatten(1)  # (B, CNN_feat_dim)

        # 2. Batch-level GCN path
        fg = self.batch_gcn(x)  # (B, gcn_out_channels)

        # 3. Concatenate
        return torch.cat([fe, fg], dim=1)  # (B, CNN_feat_dim + gcn_out_channels)

# --------------------------
# 9. Self-Supervised Loss Function
# --------------------------
def self_supervised_loss(z_a, z_b, lambd=0.005):
    """
    Compute the self-supervised loss:
      L = sum_i (1 - C_ii)^2 + λ * sum_{i≠j} C_ij^2
    where C is the cross-correlation matrix between two feature sets.
    """
    batch_size, feat_dim = z_a.shape
    # Compute cross-correlation matrix
    c = torch.mm(z_a.T, z_b) / batch_size  # Shape: (feat_dim, feat_dim)

    # Diagonal loss: make diagonal values close to 1
    on_diag = torch.diagonal(c).add(-1).pow(2).sum()
    # Off-diagonal loss: make off-diagonal values close to 0
    off_diag = (c - torch.diag(torch.diagonal(c))).pow(2).sum()

    loss = on_diag + lambd * off_diag
    return loss

class WeightedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, weight_decay=0, grad_weights=None, **kwargs):
        # Convert params to a list so we can iterate multiple times
        params = list(params)
        
        # If grad_weights is not provided, default to 1.0 for each parameter
        if grad_weights is None:
            grad_weights = [1.0 for _ in params]

        super().__init__(params, lr=lr, weight_decay=weight_decay, **kwargs)
        self.grad_weights = grad_weights

    def step(self, closure=None):
        if self.grad_weights is not None:
            param_idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.data.mul_(self.grad_weights[param_idx])
                    param_idx += 1
        super().step(closure)


# --------------------------
# 10. Self-Supervised Training Function
# --------------------------

def train_with_cumulative_curriculum(patches, difficulties, batch_size, model, optimizer, device, epoch, total_epochs):
    """
    Cumulative Curriculum Learning Training Loop

    Args:
        patches: list of (view1, view2) tensor pairs
        difficulties: list of difficulty scores (float) for each patch
        batch_size: size of each internal mini-batch (e.g. 64)
        model: the neural network model
        optimizer: optimizer for model parameters
        device: 'cuda' or 'cpu'
        epoch: current epoch number (for logging)
        total_epochs: total number of epochs (for logging)
    """
    model.train()

    # Step 1: Shuffle all sample indices
    indices = list(range(len(patches)))
    random.shuffle(indices)

    # Step 2: Split shuffled indices into mini-batches
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    # Step 3: Compute average difficulty for each batch
    batch_difficulties = [np.mean([difficulties[idx] for idx in batch]) for batch in batches]

    # Step 4: Sort batch indices by their difficulty (ascending)
    sorted_batch_indices = np.argsort(batch_difficulties)

    # Step 5: Cumulative curriculum loop
    cumulative_indices = []

    for iteration, batch_idx in enumerate(sorted_batch_indices, 1):
        # Add the current batch to cumulative training data
        cumulative_indices.extend(batches[batch_idx])

        # Get corresponding patches and difficulty scores
        selected_with_difficulty = [(patches[i], difficulties[i]) for i in cumulative_indices]

        # Sort all cumulative patches by difficulty (ascending)
        selected_with_difficulty.sort(key=lambda x: x[1])
        selected_patches = [item[0] for item in selected_with_difficulty]

        # Divide sorted cumulative data into internal mini-batches
        for i in range(0, len(selected_patches), batch_size):
            mini_batch = selected_patches[i:i + batch_size]
            if len(mini_batch) < 2:
                continue  # skip too-small batches

            # Unpack views and move to device
            views1, views2 = zip(*mini_batch)
            views1 = torch.stack(views1).to(device)
            views2 = torch.stack(views2).to(device)

            # Training step
            optimizer.zero_grad()
            z_a = model(views1)
            z_b = model(views2)
            loss = self_supervised_loss(z_a, z_b, lambd=0.005)
            loss.backward()
            optimizer.step()

        # Log progress
        print(f"Epoch [{epoch}/{total_epochs}] Iteration [{iteration}/{len(batches)}] - Trained on {len(cumulative_indices)} samples Loss: {loss.item():.4f}")

    return model


# --------------------------
# 11. Feature Extraction and Classification 
# --------------------------
def extract_features(model, PolSAR, patch_size, centers=None, device='cuda'):
    """
    Extract deep features for specified pixel centers.
    centers: list of tuples (i, j)
    """
    model.eval()
    features_list = []
    with torch.no_grad():
        for center in centers:
            patch = extract_patch(PolSAR, center, patch_size)
            patch_pca = apply_pca(patch, n_components=3)
            # Convert to tensor and apply necesPolSARy transforms
            patch_tensor = transforms.ToTensor()( (patch_pca*255).astype(np.uint8) ).unsqueeze(0).to(device)
            feat = model(patch_tensor)
            feat = feat.cpu().numpy().flatten()
            features_list.append(feat)
    return np.array(features_list)

def extract_Polarimetric_vector(PolSAR, center):
    """Return the Polarimetric vector at the given center pixel."""
    i, j = center
    Polarimetric_vec = PolSAR[i, j, :]
    return Polarimetric_vec


def get_labeled_centers(gt, sample_ratio=0.01, seed=42):
    """
    Randomly select labeled centers (pixels) for each class based on a ratio.
    
    Parameters:
    - gt: 2D numpy array of ground truth labels.
    - sample_ratio: fraction of labeled samples per class to select (e.g., 0.01 for 1%).
    - seed: random seed for reproducibility.
    
    Returns:
    - centers: list of (row, col) tuples of selected samples.
    - labels: numpy array of labels corresponding to the centers.
    """
    random.seed(seed)
    centers = []
    labels = []
    h, w = gt.shape
    
    classes = np.unique(gt)  # Automatically find the classes present in gt

    for cls in classes:
        cls_indices = np.argwhere(gt == cls)
        n_samples = max(1, int(len(cls_indices) * sample_ratio))
        selected = random.sample(cls_indices.tolist(), n_samples)
        for s in selected:
            centers.append(tuple(s))
            labels.append(cls)
    
    return centers, np.array(labels)


# --------------------------
# 12. Polarimetric Features
# --------------------------

def extract_raw_Polarimetric_features(PolSAR_subset, centers):
    """
    Extract raw Polarimetric features (direct pixel values) from the given PolSAR subset
    at specified spatial centers.

    Parameters:
        PolSAR_subset (np.ndarray): PolSAR data for one Polarimetric group [H, W, bands].
        centers (list of tuples): List of (i, j) spatial coordinates.

    Returns:
        np.ndarray: Array of shape [num_samples, num_bands] with Polarimetric vectors.
    """
    features = []
    for i, j in centers:
        features.append(PolSAR_subset[i, j, :])
    return np.array(features)


# --------------------------
# 13. Padding Array
# --------------------------

def apply_padding(features, patch=3):
    """
    Apply padding to the feature vector to convert it to a 2D matrix.

    features: 1D array of features (feature_dim,)
    patch_size: Size of the patch (3x3 for example, resulting in a 3x3 matrix)

    Returns: 2D matrix (patch_size x patch_size x feature_dim)
    """
    # Get the dimension of the feature vector
    feature_dim = features.shape[0]  # Should be 1024 for example

    # Initialize a zero matrix for the padded features
    padded_features = np.zeros((patch, patch, feature_dim))  # Initialize a zero matrix

    # Fill the center of the patch with the features
    padded_features[patch // 2, patch // 2, :] = features  # Place feature vector in the center
    return padded_features

# Applying padding on the combined features:
def apply_padding_to_features(features_list, patch=3):
    """
    Apply padding to all features in the list and return the padded features.

    features_list: List of feature vectors (N x feature_dim)
    patch_size: Size of the patch (3x3 for example)

    Returns: List of padded feature matrices (N x patch_size x patch_size x feature_dim)
    """
    padded_features_list = []

    for features in features_list:
        padded_features = apply_padding(features, patch)
        padded_features_list.append(padded_features)

    return np.array(padded_features_list)


# --------------------------------
# 14. Local Bainary Graph Algorithm
# --------------------------------


from scipy.spatial.distance import cdist
from scipy.linalg import eigh

def build_graph_matrix(patch_reshaped, k=4):
    """
    Constructs adjacency matrix A using k-nearest neighbors based on Euclidean distance.
    :param patch_reshaped: numpy array of shape (N, C), where N is number of pixels and C is number of Polarimetric bands
    :param k: number of nearest neighbors
    :return: adjacency matrix A of shape (N, N)
    """
    N = patch_reshaped.shape[0]

    # Compute Euclidean distance between all pixel vectors
    distances = cdist(patch_reshaped, patch_reshaped, metric='euclidean')

    # For each pixel, keep only k nearest neighbors (excluding self)
    A = np.zeros((N, N))
    for i in range(N):
        idx = np.argsort(distances[i])[1:k+1]  # Skip self (0th index)
        A[i, idx] = 1
        A[idx, i] = 1  # Make it symmetric

    return A

def compute_degree_matrix(A):
    """
    Constructs diagonal degree matrix D from adjacency matrix A.
    :param A: adjacency matrix of shape (N, N)
    :return: degree matrix D of shape (N, N)
    """
    D = np.diag(np.sum(A, axis=1))
    return D

def compute_laplacian(D, A):
    """
    Computes unnormalized graph Laplacian matrix L = D - A.
    :param D: degree matrix
    :param A: adjacency matrix
    :return: Laplacian matrix L
    """
    return D - A

def compute_feature_fusion(patch, output_channels=40, k=4):
    """
    Applies graph-based feature fusion on a given patch.
    :param patch: numpy array of shape (H, W, C)
    :param output_channels: number of output fused Polarimetric dimensions
    :param k: number of nearest neighbors in the graph
    :return: fused features of shape (H*W, output_channels)
    """
    H, W, C = patch.shape
    N = H * W

    # Step 1: Reshape patch to (N, C)
    X = patch.reshape(N, C)

    # Step 2: Construct graph adjacency matrix A
    A = build_graph_matrix(X, k=k)

    # Step 3: Compute degree matrix D
    D = compute_degree_matrix(A)

    # Step 4: Compute Laplacian matrix L = D - A
    L = compute_laplacian(D, A)

    # Step 5: Compute P = X^T * D * X
    P = X.T @ D @ X  # Shape: (C, C)

    # Regularization for numerical stability
    epsilon = 1e-5
    P += epsilon * np.eye(P.shape[0])

    # Step 6: Compute Q = X^T * L * X
    Q = X.T @ L @ X  # Shape: (C, C)

    # Step 7: Solve generalized eigenvalue problem Qw = λPw
    eigvals, eigvecs = eigh(Q, P)

    # Step 8: Select top 'output_channels' eigenvectors (with largest eigenvalues)
    sorted_idx = np.argsort(eigvals)[::-1]
    top_eigvecs = eigvecs[:, sorted_idx[:output_channels]]  # Shape: (C, output_channels)

    # Step 9: Multiply reshaped patch (N, C) with weight matrix (C, output_channels)
    fused_features = X @ top_eigvecs  # Resulting shape: (N, output_channels)

    return fused_features  # shape: (H*W, output_channels)

def batch_feature_fusion(patches, output_channels=40, k=4):
    """
    Applies graph-based feature fusion to a batch of patches.
    :param patches: numpy array of shape (N, H, W, C)
    :param output_channels: number of output Polarimetric features
    :param k: number of neighbors for graph construction
    :return: numpy array of shape (N, H*W, output_channels)
    """
    N, H, W, C = patches.shape
    fused_batch = np.zeros((N, H * W, output_channels))

    for i in range(N):
        fused_batch[i] = compute_feature_fusion(patches[i], output_channels=output_channels, k=k)


    return fused_batch

# --------------------------------
# 15. Feature Selection : Forward
# --------------------------------

def forward_rollback_with_test_mi(X_train, y_train, X_test, y_test):
    """
    Forward feature selection with rollback.
    Ranking is based on MI(feature, pseudo-labels on test set).
    Only features with MI >= 0.4 * max(MI) are considered.
    """

    # Step 0: Train initial classifier and get pseudo-labels on test set
    init_clf = RandomForestClassifier(n_estimators=500, random_state=42)
    init_clf.fit(X_train, y_train)
    y_int = init_clf.predict(X_test)  # Preliminary pseudo-labels on test set
    print("Initial classifier trained. Obtained pseudo-labels on test set.\n")

    # Step 1: Compute MI(feature, y_int) using X_test
    mi = mutual_info_classif(X_test, y_int, discrete_features=False)
    mi_max = np.max(mi)
    selected_candidates = [i for i, m in enumerate(mi) if m >= 0.4 * mi_max]

    if len(selected_candidates) == 0:
        selected_candidates = [np.argmax(mi)]  # Ensure at least one feature

    print("Selected candidates based on MI threshold (>= 40% max):")
    for i in selected_candidates:
        print(f"  F{i}: MI = {mi[i]:.4f}")

    # Rank selected candidates by MI descending
    ranked = sorted(selected_candidates, key=lambda i: mi[i], reverse=True)
    print("Feature ranking (desc):", [f"F{i}" for i in ranked], "\n")

    # Step 2: Initialize
    selected = []
    best_acc = 0.0
    history = []
    print(f"Initialization: S^0 = {{}} (empty), α^0 = {best_acc:.4f}\n")

    # Step 3: Forward selection with rollback
    for step, feat in enumerate(ranked, start=1):
        print(f"--- Step {step}: consider adding F{feat} ---")
        cand_set = selected + [feat]

        clf = RandomForestClassifier(n_estimators=500, random_state=42)
        clf.fit(X_train[:, cand_set], y_train)
        y_pred = clf.predict(X_test[:, cand_set])
        cand_acc = accuracy_score(y_test, y_pred)
        print(f"After adding F{feat}: set = {[f'F{i}' for i in cand_set]}, acc = {cand_acc:.4f}")

        if cand_acc >= best_acc:
            selected = cand_set
            best_acc = cand_acc
            print(f"✅ Accepted. S^{step} = {[f'F{i}' for i in selected]}, α^{step} = {best_acc:.4f}\n")
        else:
            print(f"⚠ Accuracy dropped ({cand_acc:.4f} < {best_acc:.4f}); entering rollback")
            improved = True
            while improved:
                improved = False
                for f in sorted(selected, key=lambda i: mi[i]):
                    red_set = [x for x in selected if x != f] + [feat]
                    clf.fit(X_train[:, red_set], y_train)
                    acc_red = accuracy_score(y_test, clf.predict(X_test[:, red_set]))
                    print(f"  Try replace F{f}: set = {[f'F{i}' for i in red_set]}, acc = {acc_red:.4f}")
                    if acc_red > best_acc:
                        selected = red_set
                        best_acc = acc_red
                        improved = True
                        print(f"  🔄 Replacement improved! New S = {[f'F{i}' for i in selected]}, α = {best_acc:.4f}\n")
                        break
                if not improved:
                    print(f"Rollback complete. No improvement. S^{step} = {[f'F{i}' for i in selected]}, α^{step} = {best_acc:.4f}\n")

        history.append({
            'step': step,
            'added': feat,
            'selected': selected.copy(),
            'accuracy': best_acc
        })

    print(f"Final selected set S^Final = {[f'F{i}' for i in selected]}, final accuracy = {best_acc:.4f}")
    return selected, history


def print_elapsed_time(start_time):
    end_time = time.time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"Elapsed time: {minutes} minutes and {seconds} seconds")



# --------------------------
# 16. Main Function to Run the Entire Process
# --------------------------


def main():


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Self-supervised training using full PolSAR
    print("ِPatch Ranking Uisng The Deep Curriculum Learning Model Has Been Started")
    ss_dataset = PolSARSelfSupervisedFullDataset(PolSAR, patch_size=32)

    # List of patches and their difficulty scores
    patches = []
    difficulties = []
    print(f" Number of Pathces: {len(ss_dataset)}")

    for idx in range(len(ss_dataset)):
        view1, view2 = ss_dataset[idx]
        patch = view1.numpy()  # For example, use view1 to compute difficulty
        difficulty = calculate_patch_difficulty(patch)
        # Print the calculated difficulty
        # print(f"Difficulty of the patch: {difficulty}")
        patches.append((view1, view2))
        difficulties.append(difficulty)

    # Neural Network Model
    K, M, _ = view1.shape
    model = EnSSNetWithGCNModel(gcn_in_channels=3,gcn_out_channels=128,patch_size= M,k_neighbors=49).to(device)
    # Define the WAdam optimizer
    optimizer = WeightedAdam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")  


    print("Training Process Has Been Started")

    start_time = time.time()
    epochs = 100
    total_epochs = 100
    
    # Define the Cosine Annealing learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)  # T_max = total number of epochs
    
    # Training loop with scheduler
    for epoch in range(1, total_epochs + 1):

      print(f"Training Epoch {epoch}/{total_epochs}")
      model = train_with_cumulative_curriculum(patches=patches,
        difficulties=difficulties,
        batch_size=128,
        model=model,
        optimizer=optimizer,
        device=device,
        epoch=epoch,
        total_epochs=total_epochs)
      
      scheduler.step()  # Update the learning rate at the end of each epoch

    print("The Model Has been Trained")

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Training completed in {minutes} minutes and {seconds} seconds")

    # Select approximately 1% of labeled samples from each class in the ground truth 'gt'.
    # The function automatically detects how many classes exist.
    # Returns:
        #   centers: list of (row, col) tuples of selected pixels
        #   labels: numpy array of corresponding class labels
    centers, labels = get_labeled_centers(gt, sample_ratio=0.01)
    all_centers = [(i, j) for i in range(PolSAR.shape[0]) for j in range(PolSAR.shape[1]) if gt[i, j] > 0]

    patch_sizes = [32,64,128]
    final_ensemble_probs = None
    total_weights = 0.0

    print("Starting The Second Phase of The Proposed Method: Feature Level and View Level Ensemble Strategy")

    num_bands = PolSAR.shape[2]
    band_mutual_info = np.zeros((num_bands, num_bands))

    for i in range(num_bands):
        for j in range(i+1, num_bands):
            band_mutual_info[i, j] = mutual_info_score(PolSAR[:, :, i].flatten(), PolSAR[:, :, j].flatten())
            band_mutual_info[j, i] = band_mutual_info[i, j]

    print("Mutual information between bands calculated.")


    # Assume all_model_probs will store probability outputs of each group
    all_model_probs = []

    for band_idx in range(num_bands):
        print(f"Creating Group for Band {band_idx + 1} of {num_bands}")

        mutual_info_scores = band_mutual_info[band_idx]
        sorted_band_indices = np.argsort(mutual_info_scores)[-20:]
        group_bands = [band_idx] + sorted_band_indices.tolist()
        print(f"Group {band_idx+1}: {len(group_bands)} Polarimetric Features")

        PolSAR_subset = PolSAR[:, :, group_bands]

        Polarimetric_features = extract_raw_Polarimetric_features(PolSAR_subset, centers)
        all_Polarimetric_features = extract_raw_Polarimetric_features(PolSAR_subset, all_centers)

        group_features_list = []
        group_all_features_list = []

        for ps in patch_sizes:
            print(f"View Level Ensemble: Patch Size {ps}")

            start_time = time.time()
            features = extract_features(model, PolSAR_subset, patch_size=ps, centers=centers, device=device)
            print(f"Deep Features of {ps} Patch Size: {features.shape[-1]} Features")
            print_elapsed_time(start_time)
            group_features_list.append(features)

            start_time = time.time()
            all_features = extract_features(model, PolSAR_subset, patch_size=ps, centers=all_centers, device=device)
            print_elapsed_time(start_time)
            group_all_features_list.append(all_features)

        combined_features = np.concatenate(group_features_list + [Polarimetric_features], axis=1)
        combined_all_features = np.concatenate(group_all_features_list + [all_Polarimetric_features], axis=1)

        x_train = combined_features
        x_test = combined_all_features
        y_train = labels
        y_test = np.array([gt[i, j] for (i, j) in all_centers])

        selected_feats, log_history = forward_rollback_with_test_mi(x_train, y_train, x_test, y_test)
        x_train_selected = x_train[:, selected_feats]
        x_test_selected = x_test[:, selected_feats]
        


        start_time = time.time()
        clf = RandomForestClassifier(n_estimators=500, random_state=42)
        clf.fit(x_train_selected, y_train)
        print(f"RF Classifier Has Been Trained For Group {band_idx + 1}")
        print_elapsed_time(start_time)

        probs = clf.predict_proba(x_test_selected)
        all_model_probs.append(probs)

    # Estimate class priors from label distribution
    y_test = np.array([gt[i, j] for (i, j) in all_centers])
    num_classes = probs.shape[1]
    hist = np.bincount(y_test, minlength=num_classes)
    class_priors = hist / np.sum(hist)

    # Final ensemble using Bayesian Voting
    def bayesian_voting(all_model_probs, class_priors):
        F = len(all_model_probs)
        N, C = all_model_probs[0].shape
        eps = 1e-12
        probs_stack = np.stack(all_model_probs, axis=0) + eps  # (F, N, C)
        product_likelihoods = np.prod(probs_stack, axis=0)     # (N, C)
        weighted_posteriors = product_likelihoods * class_priors  # (N, C)
        denominator = np.sum(weighted_posteriors, axis=1, keepdims=True) + eps
        final_probs = weighted_posteriors / denominator
        final_pred = np.argmax(final_probs, axis=1)
        return final_probs, final_pred

    final_ensemble_probs, final_pred = bayesian_voting(all_model_probs, class_priors)



    classified_map = np.zeros(gt.shape, dtype=int)
    valid_idx = np.argwhere(gt > 0)
    for idx, (i, j) in enumerate(valid_idx):
        classified_map[i, j] = final_pred[idx]

    gt_flat = gt[gt > 0].flatten()
    oa = accuracy_score(gt_flat, final_pred) * 100
    conf_matrix = confusion_matrix(gt_flat, final_pred)
    report = classification_report(gt_flat, final_pred)

    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)
    print(f"Overall Accuracy (OA): {oa:.2f}%")

    plt.figure(figsize=(8, 6))
    plt.imshow(classified_map, cmap='jet')
    plt.colorbar()
    plt.title("Ensemble Classification Map (Patch Sizes: 32, 64)")
    plt.show()


if __name__ == '__main__':
    main()