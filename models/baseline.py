import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineNN(nn.Module):
    """
    Standard baseline neural network for comparison with I-EDNN
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], n_classes=2, arch_type='MLP'):
        super(BaselineNN, self).__init__()
        self.arch_type = arch_type
        self.n_classes = n_classes
        
        if arch_type == 'CNN':
            # For image data
            self.conv1 = nn.Conv2d(input_dim[0], 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout_conv = nn.Dropout2d(0.25)
            
            # Calculate flattened dimension
            conv_output_size = 64 * (input_dim[1] // 4) * (input_dim[2] // 4)
            
            # Fully connected layers
            layers = []
            prev_dim = conv_output_size
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5)
                ])
                prev_dim = hidden_dim
            
            # Output layer
            layers.append(nn.Linear(prev_dim, n_classes))
            
            self.fc_layers = nn.Sequential(*layers)
            
        else:
            # For tabular data
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ])
                prev_dim = hidden_dim
            
            # Output layer
            layers.append(nn.Linear(prev_dim, n_classes))
            
            self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.arch_type == 'CNN':
            # Convolutional layers
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.dropout_conv(x)
            
            # Flatten for fully connected layers
            x = x.view(x.size(0), -1)
            
            # Fully connected layers
            x = self.fc_layers(x)
        else:
            # Fully connected layers only
            x = self.layers(x)
            
        return x

class MLPBaseline(nn.Module):
    """
    Simple MLP baseline with configurable architecture
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], n_classes=2, 
                 dropout_rate=0.3, activation='relu'):
        super(MLPBaseline, self).__init__()
        
        self.activation = getattr(F, activation)
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(hidden_dims) - 1:  # No dropout before last hidden layer
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, n_classes))
        
        self.network = nn.ModuleList(layers)
        
    def forward(self, x):
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                # Apply activation except for last layer
                if i < len(self.network) - 1:
                    x = self.activation(x)
            else:  # Dropout layer
                x = layer(x)
        
        return x

class CNNBaseline(nn.Module):
    """
    CNN baseline for image data
    """
    def __init__(self, input_channels=1, n_classes=10, img_size=28):
        super(CNNBaseline, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout2d = nn.Dropout2d(0.25)
        self.dropout1d = nn.Dropout(0.5)
        
        # Calculate size after convolutions and pooling
        conv_output_size = 128 * (img_size // 8) * (img_size // 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)
        
    def forward(self, x):
        # Convolutional layers with pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout2d(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1d(x)
        x = F.relu(self.fc2(x))
        x = self.dropout1d(x)
        x = self.fc3(x)
        
        return x

class ResidualBlock(nn.Module):
    """
    Residual block for deeper baseline networks
    """
    def __init__(self, in_dim, out_dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Skip connection
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()
            
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        
        # Add skip connection
        out += identity
        out = F.relu(out)
        
        return out

class DeepMLPBaseline(nn.Module):
    """
    Deep MLP with residual connections for fair comparison
    """
    def __init__(self, input_dim, hidden_dim=128, n_blocks=3, n_classes=2, dropout_rate=0.3):
        super(DeepMLPBaseline, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, dropout_rate) 
            for _ in range(n_blocks)
        ])
        
        # Output layer
        self.output = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = F.relu(self.input_proj(x))
        
        for block in self.blocks:
            x = block(x)
        
        x = self.dropout(x)
        x = self.output(x)
        
        return x
