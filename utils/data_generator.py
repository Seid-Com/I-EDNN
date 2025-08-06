import numpy as np
import torch
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def generate_synthetic_data(n_samples=1000, n_features=20, task_type='classification', 
                          noise=0.1, random_state=42):
    """
    Generate synthetic data for testing I-EDNN
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of input features
        task_type: 'classification' or 'regression'
        noise: Noise level in the data
        random_state: Random seed for reproducibility
    
    Returns:
        X, y: Input features and targets
    """
    np.random.seed(random_state)
    
    if task_type == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            n_redundant=n_features // 4,
            n_clusters_per_class=1,
            class_sep=1.0,
            random_state=random_state
        )
    else:  # regression
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            noise=noise * 10,
            random_state=random_state
        )
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def generate_ising_like_data(n_samples=1000, lattice_size=8, temperature_range=(0.5, 3.5), 
                           random_state=42):
    """
    Generate Ising model configurations for classification
    
    Args:
        n_samples: Number of configurations to generate
        lattice_size: Size of the square lattice
        temperature_range: Range of temperatures to sample from
        random_state: Random seed
    
    Returns:
        X, y: Lattice configurations and phase labels (0=paramagnetic, 1=ferromagnetic)
    """
    np.random.seed(random_state)
    
    T_critical = 2.269  # Critical temperature for 2D Ising model
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # Sample random temperature
        T = np.random.uniform(*temperature_range)
        
        # Generate Ising configuration using simple Monte Carlo
        config = monte_carlo_ising(lattice_size, T, steps=1000)
        
        # Flatten configuration
        X.append(config.flatten())
        
        # Label based on temperature relative to critical point
        y.append(1 if T < T_critical else 0)
    
    return np.array(X), np.array(y)

def monte_carlo_ising(size, temperature, steps=1000, J=1.0):
    """
    Simple Metropolis Monte Carlo for 2D Ising model
    
    Args:
        size: Lattice size (size x size)
        temperature: Temperature parameter
        steps: Number of Monte Carlo steps
        J: Coupling strength
    
    Returns:
        lattice: Final spin configuration
    """
    beta = 1.0 / temperature
    
    # Initialize random configuration
    lattice = 2 * np.random.randint(2, size=(size, size)) - 1
    
    for _ in range(steps):
        for _ in range(size * size):
            # Pick random site
            i, j = np.random.randint(0, size, 2)
            
            # Calculate energy change for spin flip
            neighbors = (lattice[(i+1) % size, j] + 
                        lattice[(i-1) % size, j] + 
                        lattice[i, (j+1) % size] + 
                        lattice[i, (j-1) % size])
            
            delta_E = 2 * J * lattice[i, j] * neighbors
            
            # Metropolis acceptance criterion
            if delta_E < 0 or np.random.random() < np.exp(-beta * delta_E):
                lattice[i, j] *= -1
    
    return lattice

def generate_mnist_data(n_samples_per_class=500, classes=[0, 1], flatten=True):
    """
    Load MNIST data for binary classification
    
    Args:
        n_samples_per_class: Number of samples per class
        classes: Which classes to include
        flatten: Whether to flatten images
    
    Returns:
        X_train, X_test, y_train, y_test: Train/test splits
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Filter to specific classes
    train_indices = []
    test_indices = []
    
    train_class_counts = {cls: 0 for cls in classes}
    test_class_counts = {cls: 0 for cls in classes}
    
    # Collect training indices
    for idx, (_, label) in enumerate(train_dataset):
        if label in classes and train_class_counts[label] < n_samples_per_class:
            train_indices.append(idx)
            train_class_counts[label] += 1
    
    # Collect test indices (smaller subset)
    test_samples_per_class = n_samples_per_class // 5
    for idx, (_, label) in enumerate(test_dataset):
        if label in classes and test_class_counts[label] < test_samples_per_class:
            test_indices.append(idx)
            test_class_counts[label] += 1
    
    # Extract data
    X_train = []
    y_train = []
    for idx in train_indices:
        image, label = train_dataset[idx]
        if flatten:
            image = image.view(-1).numpy()
        else:
            image = image.numpy()
        X_train.append(image)
        y_train.append(classes.index(label))  # Convert to 0, 1, 2, ...
    
    X_test = []
    y_test = []
    for idx in test_indices:
        image, label = test_dataset[idx]
        if flatten:
            image = image.view(-1).numpy()
        else:
            image = image.numpy()
        X_test.append(image)
        y_test.append(classes.index(label))
    
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def generate_spiral_data(n_samples=1000, noise=0.1, random_state=42):
    """
    Generate 2D spiral data for visualization
    
    Args:
        n_samples: Number of samples per class
        noise: Noise level
        random_state: Random seed
    
    Returns:
        X, y: 2D coordinates and class labels
    """
    np.random.seed(random_state)
    
    X = np.zeros((2 * n_samples, 2))
    y = np.zeros(2 * n_samples, dtype=int)
    
    for class_idx in range(2):
        ix = range(n_samples * class_idx, n_samples * (class_idx + 1))
        r = np.linspace(0.0, 1, n_samples)
        t = np.linspace(class_idx * 4, (class_idx + 1) * 4, n_samples) + \
            np.random.randn(n_samples) * noise
        
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = class_idx
    
    return X, y

def generate_xor_data(n_samples=1000, noise=0.1, random_state=42):
    """
    Generate XOR-like data that tests non-linear separability
    
    Args:
        n_samples: Number of samples
        noise: Noise level
        random_state: Random seed
    
    Returns:
        X, y: Input features and XOR labels
    """
    np.random.seed(random_state)
    
    # Generate random points in [-1, 1] x [-1, 1]
    X = np.random.uniform(-1, 1, (n_samples, 2))
    
    # Add noise
    X += np.random.normal(0, noise, X.shape)
    
    # XOR labels
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    
    return X, y

def visualize_data(X, y, title="Data Visualization"):
    """
    Visualize 2D data with class labels
    
    Args:
        X: 2D input features
        y: Class labels
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    
    classes = np.unique(y)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, cls in enumerate(classes):
        mask = y == cls
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[i % len(colors)], 
                   label=f'Class {cls}', 
                   alpha=0.6)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Example usage and testing
if __name__ == "__main__":
    # Test synthetic data generation
    X_cls, y_cls = generate_synthetic_data(n_samples=500, task_type='classification')
    print(f"Classification data shape: {X_cls.shape}, {y_cls.shape}")
    print(f"Class distribution: {np.bincount(y_cls)}")
    
    X_reg, y_reg = generate_synthetic_data(n_samples=500, task_type='regression')
    print(f"Regression data shape: {X_reg.shape}, {y_reg.shape}")
    
    # Test Ising data generation
    X_ising, y_ising = generate_ising_like_data(n_samples=100, lattice_size=4)
    print(f"Ising data shape: {X_ising.shape}, {y_ising.shape}")
    print(f"Phase distribution: {np.bincount(y_ising)}")
    
    # Test 2D data for visualization
    X_spiral, y_spiral = generate_spiral_data(n_samples=200)
    print(f"Spiral data shape: {X_spiral.shape}")
    
    X_xor, y_xor = generate_xor_data(n_samples=200)
    print(f"XOR data shape: {X_xor.shape}")
