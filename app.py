import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import sys
import time

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from models.iednn import IEDNN
from models.baseline import BaselineNN
from utils.data_generator import generate_mnist_data, generate_synthetic_data
from utils.visualization import plot_interaction_matrix, plot_energy_landscape, plot_training_curves
from utils.training import train_model, evaluate_model
from physics.ising_model import IsingModel
from physics.annealing import SimulatedAnnealing

# Set page config
st.set_page_config(
    page_title="I-EDNN: Ising-Enhanced Deep Neural Network",
    page_icon="🧲",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'baseline_model' not in st.session_state:
    st.session_state.baseline_model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False

def main():
    st.title("🧲 Ising-Enhanced Deep Neural Network (I-EDNN)")
    st.markdown("""
    A physics-inspired neural network architecture that integrates principles from statistical physics 
    into deep learning using the Ising model for structured and efficient learning.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Overview", "Data Generation", "Model Configuration", "Training", "Results & Analysis", "Physics Visualization"]
    )
    
    if page == "Overview":
        show_overview()
    elif page == "Data Generation":
        show_data_generation()
    elif page == "Model Configuration":
        show_model_configuration()
    elif page == "Training":
        show_training()
    elif page == "Results & Analysis":
        show_results()
    elif page == "Physics Visualization":
        show_physics_visualization()

def show_overview():
    st.header("I-EDNN Architecture Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Components")
        st.markdown("""
        1. **Feature Extractor**: Standard deep network (CNN/MLP) for continuous feature extraction
        2. **Spin Projection Layer**: Converts continuous features to binary spins using sign activation
        3. **Interaction Graph**: Learnable sparse symmetric matrix J for spin-spin couplings
        4. **Energy Pooling**: Computes Ising energy from spin configurations and interactions
        5. **Annealing Optimizer**: Hybrid training combining gradient descent with simulated annealing
        6. **Task-Specific Head**: Classification/regression output layer
        """)
    
    with col2:
        st.subheader("Physics Inspiration")
        st.markdown("""
        The Ising model from statistical physics describes magnetic systems with binary spins:
        
        **Energy Function:**
        ```
        E = -∑(i,j) J_ij * s_i * s_j - ∑_i h_i * s_i
        ```
        
        - **J_ij**: Interaction strengths between spins
        - **s_i**: Binary spin variables (+1 or -1)
        - **h_i**: External magnetic field on each spin
        
        I-EDNN learns these interaction patterns from data to capture feature dependencies.
        """)

    # Architecture diagram would go here
    st.subheader("Benefits of Physics-Inspired Learning")
    st.markdown("""
    - **Structured Representations**: Binary spins create interpretable feature interactions
    - **Sparsity**: L1 regularization learns only essential feature connections  
    - **Energy-Based**: Global optimization through simulated annealing
    - **Inductive Bias**: Physics principles guide learning for better generalization
    """)

def show_data_generation():
    st.header("Data Generation")
    
    st.markdown("""
    Choose a dataset type to train and evaluate I-EDNN models.
    """)
    
    data_type = st.selectbox(
        "Select Data Type:",
        ["Synthetic Classification", "Synthetic Regression", "MNIST Digits", "Ising-Like Data", "Spiral Data", "XOR Data"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Parameters")
        
        # Initialize variables with defaults
        n_samples = 1000
        n_features = 20
        noise_level = 0.1
        samples_per_class = 500
        digit_classes = [0, 1]
        lattice_size = 8
        temp_min = 0.5
        temp_max = 3.5
        
        if data_type in ["Synthetic Classification", "Synthetic Regression"]:
            n_samples = st.slider("Number of samples", 100, 5000, 1000)
            n_features = st.slider("Number of features", 5, 50, 20)
            noise_level = st.slider("Noise level", 0.0, 1.0, 0.1)
            
        elif data_type == "MNIST Digits":
            samples_per_class = st.slider("Samples per class", 100, 1000, 500)
            digit_classes = st.multiselect("Select digits", list(range(10)), [0, 1])
            
        elif data_type == "Ising-Like Data":
            n_samples = st.slider("Number of samples", 100, 2000, 1000)
            lattice_size = st.slider("Lattice size", 4, 16, 8)
            temp_min = st.slider("Min temperature", 0.5, 2.0, 0.5)
            temp_max = st.slider("Max temperature", 2.5, 5.0, 3.5)
            
        else:  # Spiral or XOR
            n_samples = st.slider("Number of samples", 200, 2000, 1000)
            noise_level = st.slider("Noise level", 0.0, 0.5, 0.1)
    
    with col2:
        if st.button("Generate Data"):
            with st.spinner("Generating data..."):
                try:
                    if data_type == "Synthetic Classification":
                        from utils.data_generator import generate_synthetic_data
                        X, y = generate_synthetic_data(
                            n_samples=n_samples, 
                            n_features=n_features, 
                            task_type='classification',
                            noise=noise_level
                        )
                        st.session_state.X, st.session_state.y = X, y
                        st.session_state.data_generated = True
                        st.success(f"Generated {len(X)} samples with {X.shape[1]} features")
                        
                    elif data_type == "Synthetic Regression":
                        from utils.data_generator import generate_synthetic_data
                        X, y = generate_synthetic_data(
                            n_samples=n_samples, 
                            n_features=n_features, 
                            task_type='regression',
                            noise=noise_level
                        )
                        st.session_state.X, st.session_state.y = X, y
                        st.session_state.data_generated = True
                        st.success(f"Generated {len(X)} samples with {X.shape[1]} features")
                        
                    elif data_type == "MNIST Digits":
                        from utils.data_generator import generate_mnist_data
                        X_train, X_test, y_train, y_test = generate_mnist_data(
                            n_samples_per_class=samples_per_class,
                            classes=digit_classes,
                            flatten=True
                        )
                        # Combine train and test for simplicity
                        X = np.vstack([X_train, X_test])
                        y = np.hstack([y_train, y_test])
                        st.session_state.X, st.session_state.y = X, y
                        st.session_state.data_generated = True
                        st.success(f"Loaded MNIST data: {len(X)} samples")
                        
                    elif data_type == "Ising-Like Data":
                        from utils.data_generator import generate_ising_like_data
                        X, y = generate_ising_like_data(
                            n_samples=n_samples,
                            lattice_size=lattice_size,
                            temperature_range=(temp_min, temp_max)
                        )
                        st.session_state.X, st.session_state.y = X, y
                        st.session_state.data_generated = True
                        st.success(f"Generated {len(X)} Ising configurations")
                        
                    elif data_type == "Spiral Data":
                        from utils.data_generator import generate_spiral_data
                        X, y = generate_spiral_data(n_samples=n_samples//2, noise=noise_level)
                        st.session_state.X, st.session_state.y = X, y
                        st.session_state.data_generated = True
                        st.success(f"Generated {len(X)} spiral data points")
                        
                    elif data_type == "XOR Data":
                        from utils.data_generator import generate_xor_data
                        X, y = generate_xor_data(n_samples=n_samples, noise=noise_level)
                        st.session_state.X, st.session_state.y = X, y
                        st.session_state.data_generated = True
                        st.success(f"Generated {len(X)} XOR data points")
                        
                except Exception as e:
                    st.error(f"Error generating data: {str(e)}")
    
    # Display data if available
    if st.session_state.data_generated:
        st.subheader("Generated Data")
        X, y = st.session_state.X, st.session_state.y
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Shape**: {X.shape}")
            st.write(f"**Classes**: {len(np.unique(y))}")
            st.write(f"**Class distribution**: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        with col2:
            if X.shape[1] == 2:  # 2D data can be visualized
                fig = px.scatter(
                    x=X[:, 0], y=X[:, 1], color=y.astype(str),
                    title="Data Visualization",
                    labels={'x': 'Feature 1', 'y': 'Feature 2', 'color': 'Class'}
                )
                st.plotly_chart(fig, use_container_width=True)

def show_model_configuration():
    st.header("Model Configuration")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first in the 'Data Generation' tab.")
        return
    
    X, y = st.session_state.X, st.session_state.y
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    st.markdown(f"""
    **Dataset Info:**
    - Features: {n_features}
    - Classes: {n_classes}
    - Samples: {len(X)}
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("I-EDNN Configuration")
        
        feature_dim = st.slider("Feature dimension", 16, 128, 64)
        spin_dim = st.slider("Spin dimension", 8, 64, 32)
        sparsity_lambda = st.slider("Sparsity regularization", 0.001, 0.1, 0.01)
        temperature = st.slider("Temperature parameter", 0.1, 2.0, 1.0)
        
        arch_type = st.selectbox("Architecture type", ["MLP", "CNN"] if n_features > 100 else ["MLP"])
        
    with col2:
        st.subheader("Baseline Configuration")
        
        if arch_type == "MLP":
            hidden_dims = st.multiselect(
                "Hidden layer sizes", 
                [16, 32, 64, 128, 256], 
                default=[64, 32]
            )
        else:
            hidden_dims = [128, 64]  # Default for CNN
        
        dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.2)
    
    if st.button("Initialize Models"):
        with st.spinner("Initializing models..."):
            try:
                # Initialize I-EDNN
                from models.iednn import IEDNN
                
                if arch_type == "CNN" and n_features > 100:
                    # Assume square images
                    img_size = int(np.sqrt(n_features))
                    input_dim = (1, img_size, img_size)
                else:
                    input_dim = n_features
                
                iednn_model = IEDNN(
                    input_dim=input_dim,
                    feature_dim=feature_dim,
                    spin_dim=spin_dim,
                    n_classes=n_classes,
                    sparsity_lambda=sparsity_lambda,
                    temperature=temperature,
                    arch_type=arch_type
                )
                
                # Initialize baseline
                from models.baseline import BaselineNN
                baseline_model = BaselineNN(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    n_classes=n_classes,
                    arch_type=arch_type
                )
                
                st.session_state.model = iednn_model
                st.session_state.baseline_model = baseline_model
                
                st.success("Models initialized successfully!")
                
                # Display model architectures
                st.subheader("Model Architectures")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**I-EDNN Architecture:**")
                    st.text(str(iednn_model))
                
                with col2:
                    st.write("**Baseline Architecture:**")
                    st.text(str(baseline_model))
                    
            except Exception as e:
                st.error(f"Error initializing models: {str(e)}")

def show_training():
    st.header("Model Training")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first.")
        return
    
    if st.session_state.model is None:
        st.warning("Please configure models first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Parameters")
        epochs = st.slider("Epochs", 10, 200, 50)
        batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
        learning_rate = st.selectbox("Learning rate", [0.0001, 0.001, 0.01], index=1)
        train_split = st.slider("Train split", 0.6, 0.9, 0.8)
        
    with col2:
        st.subheader("Training Options")
        train_iednn = st.checkbox("Train I-EDNN", value=True)
        train_baseline = st.checkbox("Train Baseline", value=True)
        use_gpu = st.checkbox("Use GPU (if available)", value=False)
    
    if st.button("Start Training"):
        if not (train_iednn or train_baseline):
            st.error("Please select at least one model to train.")
            return
        
        X, y = st.session_state.X, st.session_state.y
        
        # Split data
        split_idx = int(train_split * len(X))
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.LongTensor(y_val)
        
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        training_history = {}
        
        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            if train_iednn:
                status_text.text("Training I-EDNN...")
                from utils.training import train_model
                
                def progress_callback(epoch, total_epochs):
                    progress_bar.progress(epoch / total_epochs * (0.5 if train_baseline else 1.0))
                
                iednn_history = train_model(
                    st.session_state.model,
                    X_train, y_train, X_val, y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    device=device,
                    progress_callback=progress_callback
                )
                training_history['I-EDNN'] = iednn_history
            
            if train_baseline:
                status_text.text("Training Baseline...")
                from utils.training import train_model
                
                def progress_callback(epoch, total_epochs):
                    base_progress = 0.5 if train_iednn else 0.0
                    progress_bar.progress(base_progress + epoch / total_epochs * 0.5)
                
                baseline_history = train_model(
                    st.session_state.baseline_model,
                    X_train, y_train, X_val, y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    device=device,
                    progress_callback=progress_callback
                )
                training_history['Baseline'] = baseline_history
            
            st.session_state.training_history = training_history
            progress_bar.progress(1.0)
            status_text.text("Training completed!")
            st.success("Training completed successfully!")
            
            # Show final results
            st.subheader("Training Results")
            for model_name, history in training_history.items():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        f"{model_name} Final Train Loss",
                        f"{history['train_loss'][-1]:.4f}"
                    )
                with col2:
                    st.metric(
                        f"{model_name} Final Val Accuracy",
                        f"{history['val_acc'][-1]:.4f}"
                    )
            
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            progress_bar.progress(0.0)
            status_text.text("Training failed!")

def show_results():
    st.header("Results & Analysis")
    
    if st.session_state.training_history is None:
        st.warning("Please train models first.")
        return
    
    # Training curves
    st.subheader("Training Curves")
    from utils.visualization import plot_training_curves
    
    fig = plot_training_curves(st.session_state.training_history)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison
    if len(st.session_state.training_history) > 1:
        st.subheader("Model Comparison")
        
        comparison_data = []
        for model_name, history in st.session_state.training_history.items():
            comparison_data.append({
                'Model': model_name,
                'Final Train Loss': f"{history['train_loss'][-1]:.4f}",
                'Final Val Loss': f"{history['val_loss'][-1]:.4f}",
                'Final Train Acc': f"{history['train_acc'][-1]:.4f}",
                'Final Val Acc': f"{history['val_acc'][-1]:.4f}"
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
    
    # I-EDNN specific analysis
    if st.session_state.model is not None:
        st.subheader("I-EDNN Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Interaction matrix
            try:
                J_matrix = st.session_state.model.get_interaction_matrix()
                from utils.visualization import plot_interaction_matrix
                fig = plot_interaction_matrix(J_matrix)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.error("Could not generate interaction matrix visualization")
        
        with col2:
            # Feature importance
            try:
                from utils.visualization import plot_feature_importance
                fig = plot_feature_importance(st.session_state.model)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.error("Could not generate feature importance plot")
    
    # Sparsity evolution
    if 'I-EDNN' in st.session_state.training_history and st.session_state.training_history['I-EDNN'].get('sparsity'):
        st.subheader("Sparsity Evolution")
        from utils.visualization import plot_sparsity_evolution
        fig = plot_sparsity_evolution(st.session_state.training_history['I-EDNN']['sparsity'])
        st.plotly_chart(fig, use_container_width=True)

def show_physics_visualization():
    st.header("Physics Visualization")
    
    st.markdown("""
    Explore the physics concepts behind I-EDNN through interactive simulations.
    """)
    
    tab1, tab2, tab3 = st.tabs(["Ising Model Simulation", "Simulated Annealing", "Spin Interactions"])
    
    with tab1:
        st.subheader("2D Ising Model")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            lattice_size = st.slider("Lattice size", 10, 50, 20)
            temperature = st.slider("Temperature", 0.5, 5.0, 2.27)
            coupling_J = st.slider("Coupling strength J", 0.1, 2.0, 1.0)
            external_field = st.slider("External field h", -1.0, 1.0, 0.0)
            n_steps = st.slider("Simulation steps", 100, 5000, 1000)
            
            if st.button("Run Ising Simulation"):
                from physics.ising_model import IsingModel
                
                with st.spinner("Running simulation..."):
                    ising = IsingModel(
                        size=lattice_size,
                        J=coupling_J,
                        h=external_field,
                        temperature=temperature
                    )
                    
                    ising.equilibrate(n_steps // 2)
                    ising.simulate(n_steps, record_interval=10)
                    
                    st.session_state.ising_model = ising
        
        with col2:
            if 'ising_model' in st.session_state:
                ising = st.session_state.ising_model
                
                from utils.visualization import visualize_ising_configuration
                fig = visualize_ising_configuration(ising.lattice, temperature)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show observables
                st.write(f"**Final Magnetization**: {ising.magnetization()}")
                st.write(f"**Final Energy**: {ising.total_energy():.2f}")
                
                if len(ising.energy_history) > 1:
                    fig_energy = go.Figure()
                    fig_energy.add_trace(go.Scatter(
                        y=ising.energy_history,
                        mode='lines',
                        name='Energy'
                    ))
                    fig_energy.update_layout(title="Energy vs Time", yaxis_title="Energy")
                    st.plotly_chart(fig_energy, use_container_width=True)
    
    with tab2:
        st.subheader("Simulated Annealing Optimization")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_spins = st.slider("Number of spins", 10, 50, 20)
            initial_temp = st.slider("Initial temperature", 1.0, 20.0, 10.0)
            final_temp = st.slider("Final temperature", 0.01, 1.0, 0.01)
            cooling_schedule = st.selectbox("Cooling schedule", ["exponential", "linear", "logarithmic"])
            annealing_steps = st.slider("Annealing steps", 100, 2000, 1000)
            
            if st.button("Run Annealing"):
                from physics.annealing import SimulatedAnnealing
                
                with st.spinner("Running annealing..."):
                    # Create random problem
                    J_matrix = np.random.randn(n_spins, n_spins) * 0.1
                    J_matrix = (J_matrix + J_matrix.T) / 2
                    np.fill_diagonal(J_matrix, 0)
                    
                    sa = SimulatedAnnealing(
                        initial_temp=initial_temp,
                        final_temp=final_temp,
                        cooling_schedule=cooling_schedule
                    )
                    
                    best_spins, best_energy, history = sa.optimize_ising(
                        J_matrix, n_steps=annealing_steps
                    )
                    
                    st.session_state.annealing_history = history
                    st.session_state.best_energy = best_energy
        
        with col2:
            if 'annealing_history' in st.session_state:
                history = st.session_state.annealing_history
                
                # Temperature and energy evolution
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Temperature Schedule", "Energy Evolution")
                )
                
                fig.add_trace(
                    go.Scatter(y=history['temperature'], name='Temperature'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(y=history['energy'], name='Current Energy'),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(y=history['best_energy'], name='Best Energy'),
                    row=2, col=1
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.write(f"**Best Energy Found**: {st.session_state.best_energy:.4f}")
    
    with tab3:
        st.subheader("Spin Interaction Networks")
        
        if st.session_state.model is not None:
            try:
                J_matrix = st.session_state.model.get_interaction_matrix()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    from utils.visualization import plot_interaction_matrix
                    fig = plot_interaction_matrix(J_matrix, title="Learned Interactions")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    threshold = st.slider("Interaction threshold", 0.001, 0.1, 0.01)
                    from utils.visualization import plot_interaction_graph
                    fig = plot_interaction_graph(J_matrix, threshold=threshold)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                strong_interactions = np.sum(np.abs(J_matrix) > threshold)
                sparsity = (1 - strong_interactions / (J_matrix.shape[0]**2 - J_matrix.shape[0])) * 100
                
                st.write(f"**Strong Interactions**: {strong_interactions}")
                st.write(f"**Sparsity**: {sparsity:.1f}%")
                
            except Exception as e:
                st.error(f"No trained I-EDNN model available: {str(e)}")
        else:
            st.info("Train an I-EDNN model first to visualize learned interactions.")

if __name__ == "__main__":
    main()
        