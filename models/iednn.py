import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StraightThroughBinarize(torch.autograd.Function):
    """
    Straight-through estimator for binary activation.
    Forward pass: sign function, backward pass: identity
    """
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class SpinProjection(nn.Module):
    """
    Projects continuous features to binary spins using straight-through estimator
    """
    def __init__(self, feature_dim, spin_dim):
        super(SpinProjection, self).__init__()
        self.feature_dim = feature_dim
        self.spin_dim = spin_dim
        
        # Linear projection to spin dimension
        self.projection = nn.Linear(feature_dim, spin_dim)
        
    def forward(self, features):
        # Project to spin dimension
        projected = self.projection(features)
        
        # Binarize using straight-through estimator
        spins = StraightThroughBinarize.apply(projected)
        
        return spins

class SpinInteraction(nn.Module):
    """
    Learnable sparse symmetric interaction matrix for spin-spin couplings
    """
    def __init__(self, spin_dim, sparsity_lambda=0.01):
        super(SpinInteraction, self).__init__()
        self.spin_dim = spin_dim
        self.sparsity_lambda = sparsity_lambda
        
        # Symmetric interaction matrix (upper triangular)
        self.J_upper = nn.Parameter(torch.randn(spin_dim, spin_dim) * 0.1)
        
        # External field (bias) for each spin
        self.h = nn.Parameter(torch.zeros(spin_dim))
        
    def forward(self, spins):
        # Create symmetric matrix from upper triangular
        J = torch.triu(self.J_upper, diagonal=1)
        J = J + J.t()  # Make symmetric
        
        # Store for visualization
        self.J = J
        
        return J, self.h
    
    def l1_regularization(self):
        """L1 regularization for sparsity"""
        return self.sparsity_lambda * torch.norm(self.J_upper, p=1)

class EnergyPooling(nn.Module):
    """
    Computes Ising energy from spin configurations and interactions
    """
    def __init__(self):
        super(EnergyPooling, self).__init__()
        
    def forward(self, spins, J, h):
        batch_size = spins.size(0)
        
        # Compute pairwise interaction energy: -0.5 * sum_ij J_ij * s_i * s_j
        # Using matrix multiplication: s^T * J * s
        interaction_energy = -0.5 * torch.bmm(
            torch.bmm(spins.unsqueeze(1), J.unsqueeze(0).expand(batch_size, -1, -1)),
            spins.unsqueeze(2)
        ).squeeze()
        
        # Compute field energy: -sum_i h_i * s_i
        field_energy = -torch.sum(spins * h.unsqueeze(0), dim=1)
        
        # Total energy
        total_energy = interaction_energy + field_energy
        
        return total_energy

class FeatureExtractor(nn.Module):
    """
    Standard deep network for feature extraction (CNN or MLP)
    """
    def __init__(self, input_dim, feature_dim, arch_type='MLP'):
        super(FeatureExtractor, self).__init__()
        self.arch_type = arch_type
        
        if arch_type == 'CNN':
            # For image data
            self.conv1 = nn.Conv2d(input_dim[0], 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.25)
            
            # Calculate flattened dimension
            conv_output_size = 64 * (input_dim[1] // 4) * (input_dim[2] // 4)
            self.fc1 = nn.Linear(conv_output_size, 128)
            self.fc2 = nn.Linear(128, feature_dim)
            
        else:
            # For tabular data
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, feature_dim)
            self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        if self.arch_type == 'CNN':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.dropout(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
        return x

class IEDNN(nn.Module):
    """
    Ising-Enhanced Deep Neural Network (I-EDNN)
    
    Architecture:
    Input -> Feature Extractor -> Spin Projection -> Interaction Graph -> Energy Pooling -> Output
    """
    def __init__(self, input_dim, feature_dim=64, spin_dim=32, n_classes=2, 
                 sparsity_lambda=0.01, temperature=1.0, arch_type='MLP'):
        super(IEDNN, self).__init__()
        
        self.feature_dim = feature_dim
        self.spin_dim = spin_dim
        self.n_classes = n_classes
        self.temperature = temperature
        
        # Components
        self.feature_extractor = FeatureExtractor(input_dim, feature_dim, arch_type)
        self.spin_projection = SpinProjection(feature_dim, spin_dim)
        self.spin_interaction = SpinInteraction(spin_dim, sparsity_lambda)
        self.energy_pooling = EnergyPooling()
        
        # Task-specific head
        self.classifier = nn.Sequential(
            nn.Linear(1, 32),  # Energy is scalar
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes)
        )
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Project to binary spins
        spins = self.spin_projection(features)
        
        # Get interaction matrix and biases
        J, h = self.spin_interaction(spins)
        
        # Compute energy
        energy = self.energy_pooling(spins, J, h)
        
        # Classification
        logits = self.classifier(energy.unsqueeze(1))
        
        return logits
    
    def get_sparsity_loss(self):
        """Get L1 regularization loss for sparsity"""
        return self.spin_interaction.l1_regularization()
    
    def get_spin_configuration(self, x):
        """Get spin configuration for visualization"""
        with torch.no_grad():
            features = self.feature_extractor(x)
            spins = self.spin_projection(features)
            return spins
    
    def get_interaction_matrix(self):
        """Get learned interaction matrix"""
        with torch.no_grad():
            if hasattr(self.spin_interaction, 'J'):
                return self.spin_interaction.J.cpu().numpy()
            else:
                # Create dummy input to trigger forward pass
                dummy_input = torch.zeros(1, self.spin_dim)
                _, _ = self.spin_interaction(dummy_input)
                return self.spin_interaction.J.cpu().numpy()
