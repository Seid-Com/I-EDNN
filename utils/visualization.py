import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import seaborn as sns

def plot_interaction_matrix(J_matrix, threshold=0.01, title="Spin Interaction Matrix"):
    """
    Visualize the learned interaction matrix J
    
    Args:
        J_matrix: Symmetric interaction matrix
        threshold: Threshold for significant interactions
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Heatmap of interaction matrix
    fig.add_trace(go.Heatmap(
        z=J_matrix,
        colorscale='RdBu',
        zmid=0,
        showscale=True,
        colorbar=dict(title="Coupling Strength")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Spin Index",
        yaxis_title="Spin Index",
        width=600,
        height=600
    )
    
    return fig

def plot_interaction_graph(J_matrix, threshold=0.01, title="Spin Interaction Graph"):
    """
    Visualize interaction matrix as a graph
    
    Args:
        J_matrix: Symmetric interaction matrix
        threshold: Threshold for edge creation
        title: Plot title
    
    Returns:
        Plotly figure
    """
    # Create graph from interaction matrix
    G = nx.Graph()
    n_spins = J_matrix.shape[0]
    
    # Add nodes
    for i in range(n_spins):
        G.add_node(i)
    
    # Add edges for significant interactions
    for i in range(n_spins):
        for j in range(i+1, n_spins):
            if abs(J_matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=abs(J_matrix[i, j]))
    
    # Get layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Extract edge information
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2]['weight'])
    
    # Extract node information
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [f"Spin {i}" for i in G.nodes()]
    
    # Create plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=15, color='lightblue', line=dict(width=2, color='black')),
        text=node_text,
        textposition="middle center",
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=600,
        height=600
    )
    
    return fig

def plot_energy_landscape(model, X_sample, n_points=50):
    """
    Plot energy landscape for 2D data
    
    Args:
        model: Trained I-EDNN model
        X_sample: Sample data points
        n_points: Grid resolution
    
    Returns:
        Plotly figure
    """
    if X_sample.shape[1] != 2:
        raise ValueError("Energy landscape visualization only supports 2D data")
    
    # Create grid
    x_min, x_max = X_sample[:, 0].min() - 1, X_sample[:, 0].max() + 1
    y_min, y_max = X_sample[:, 1].min() - 1, X_sample[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n_points),
        np.linspace(y_min, y_max, n_points)
    )
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Compute energies
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.FloatTensor(grid_points)
        features = model.feature_extractor(grid_tensor)
        spins = model.spin_projection(features)
        J, h = model.spin_interaction(spins)
        energies = model.energy_pooling(spins, J, h)
    
    energies = energies.numpy().reshape(xx.shape)
    
    # Create contour plot
    fig = go.Figure()
    
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, n_points),
        y=np.linspace(y_min, y_max, n_points),
        z=energies,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Energy")
    ))
    
    # Add sample points
    fig.add_trace(go.Scatter(
        x=X_sample[:, 0],
        y=X_sample[:, 1],
        mode='markers',
        marker=dict(size=8, color='red', line=dict(width=1, color='white')),
        name='Data Points'
    ))
    
    fig.update_layout(
        title="Energy Landscape",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        width=600,
        height=600
    )
    
    return fig

def plot_training_curves(training_history):
    """
    Plot training curves for comparison
    
    Args:
        training_history: Dictionary with training histories
    
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy')
    )
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (model_name, history) in enumerate(training_history.items()):
        color = colors[i % len(colors)]
        
        # Training loss
        fig.add_trace(
            go.Scatter(x=list(range(len(history['train_loss']))), 
                      y=history['train_loss'],
                      name=f'{model_name} Train Loss',
                      line=dict(color=color, dash='solid')),
            row=1, col=1
        )
        
        # Validation loss
        fig.add_trace(
            go.Scatter(x=list(range(len(history['val_loss']))), 
                      y=history['val_loss'],
                      name=f'{model_name} Val Loss',
                      line=dict(color=color, dash='dash')),
            row=1, col=2
        )
        
        # Training accuracy
        fig.add_trace(
            go.Scatter(x=list(range(len(history['train_acc']))), 
                      y=history['train_acc'],
                      name=f'{model_name} Train Acc',
                      line=dict(color=color, dash='solid')),
            row=2, col=1
        )
        
        # Validation accuracy
        fig.add_trace(
            go.Scatter(x=list(range(len(history['val_acc']))), 
                      y=history['val_acc'],
                      name=f'{model_name} Val Acc',
                      line=dict(color=color, dash='dash')),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Training Progress Comparison",
        height=600,
        showlegend=True
    )
    
    return fig

def plot_spin_configuration(spins, title="Spin Configuration"):
    """
    Visualize binary spin configuration
    
    Args:
        spins: 1D array of binary spins
        title: Plot title
    
    Returns:
        Plotly figure
    """
    # Reshape to square grid if possible
    n_spins = len(spins)
    grid_size = int(np.sqrt(n_spins))
    
    if grid_size * grid_size == n_spins:
        spin_grid = spins.reshape(grid_size, grid_size)
    else:
        # Use 1D visualization
        spin_grid = spins.reshape(1, -1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=spin_grid,
        colorscale=[[0, 'blue'], [0.5, 'white'], [1, 'red']],
        zmid=0,
        showscale=True,
        colorbar=dict(title="Spin Value", tickvals=[-1, 0, 1])
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Spin Index",
        yaxis_title="Spin Index" if spin_grid.shape[0] > 1 else "",
        width=400,
        height=400
    )
    
    return fig

def plot_feature_importance(model, feature_names=None):
    """
    Plot feature importance based on interaction strengths
    
    Args:
        model: Trained I-EDNN model
        feature_names: Optional feature names
    
    Returns:
        Plotly figure
    """
    # Get interaction matrix
    J_matrix = model.get_interaction_matrix()
    
    # Compute feature importance as sum of absolute interactions
    importance = np.sum(np.abs(J_matrix), axis=1)
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance))]
    
    # Sort by importance
    sorted_indices = np.argsort(importance)[::-1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[feature_names[i] for i in sorted_indices],
        y=importance[sorted_indices],
        marker_color='skyblue'
    ))
    
    fig.update_layout(
        title="Feature Importance (Based on Interaction Strengths)",
        xaxis_title="Features",
        yaxis_title="Importance Score",
        xaxis_tickangle=-45
    )
    
    return fig

def plot_sparsity_evolution(sparsity_history, title="Sparsity Evolution"):
    """
    Plot evolution of sparsity during training
    
    Args:
        sparsity_history: List of sparsity values over epochs
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(sparsity_history))),
        y=sparsity_history,
        mode='lines+markers',
        name='Sparsity',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Sparsity (%)",
        showlegend=False
    )
    
    return fig

def visualize_ising_configuration(lattice, temperature=None, title="Ising Configuration"):
    """
    Visualize 2D Ising model configuration
    
    Args:
        lattice: 2D array of spins (+1 or -1)
        temperature: Optional temperature value
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=lattice,
        colorscale='RdBu',
        zmid=0,
        showscale=True,
        colorbar=dict(title="Spin", tickvals=[-1, 0, 1])
    ))
    
    if temperature is not None:
        title += f" (T={temperature:.2f})"
    
    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="y",
        width=500,
        height=500,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    
    return fig

# Example usage
if __name__ == "__main__":
    # Test interaction matrix visualization
    J_test = np.random.randn(10, 10) * 0.1
    J_test = (J_test + J_test.T) / 2  # Make symmetric
    np.fill_diagonal(J_test, 0)  # Zero diagonal
    
    fig = plot_interaction_matrix(J_test)
    fig.show()
    
    # Test Ising configuration
    lattice = 2 * np.random.randint(2, size=(20, 20)) - 1
    fig = visualize_ising_configuration(lattice, temperature=2.0)
    fig.show()
