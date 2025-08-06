import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import time

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, 
                learning_rate=0.001, device='cpu', progress_callback=None):
    """
    Train a model with given data
    
    Args:
        model: PyTorch model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        progress_callback: Callback function for progress updates
    
    Returns:
        Dictionary with training history
    """
    model.to(device)
    model.train()
    
    # Determine task type
    if len(torch.unique(torch.LongTensor(y_train))) <= 10:  # Classification
        criterion = nn.CrossEntropyLoss()
        task_type = 'classification'
    else:  # Regression
        criterion = nn.MSELoss()
        task_type = 'regression'
        y_train = y_train.float()
        y_val = y_val.float()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [] if task_type == 'classification' else [],
        'val_acc': [] if task_type == 'classification' else [],
        'sparsity': [] if hasattr(model, 'get_sparsity_loss') else []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            
            if task_type == 'classification':
                loss = criterion(outputs, batch_y.long())
                preds = torch.argmax(outputs, dim=1)
            else:
                loss = criterion(outputs.squeeze(), batch_y)
                preds = outputs.squeeze()
            
            # Add sparsity regularization for I-EDNN
            if hasattr(model, 'get_sparsity_loss'):
                sparsity_loss = model.get_sparsity_loss()
                loss += sparsity_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(batch_y.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                
                if task_type == 'classification':
                    loss = criterion(outputs, batch_y.long())
                    preds = torch.argmax(outputs, dim=1)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)
                    preds = outputs.squeeze()
                
                val_loss += loss.item()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if task_type == 'classification':
            train_acc = accuracy_score(train_targets, train_preds)
            val_acc = accuracy_score(val_targets, val_preds)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
        else:
            train_acc = r2_score(train_targets, train_preds)
            val_acc = r2_score(val_targets, val_preds)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Track sparsity for I-EDNN
        if hasattr(model, 'get_interaction_matrix'):
            J_matrix = model.get_interaction_matrix()
            sparsity = np.mean(np.abs(J_matrix) < 0.01) * 100
            history['sparsity'].append(sparsity)
        
        # Progress callback
        if progress_callback:
            progress_callback(epoch + 1, epochs)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            metric_name = "Accuracy" if task_type == 'classification' else "R²"
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train {metric_name}: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val {metric_name}: {val_acc:.4f}")
            if history['sparsity']:
                print(f"  Sparsity: {history['sparsity'][-1]:.1f}%")
    
    return history

def evaluate_model(model, X_test, y_test, device='cpu'):
    """
    Evaluate a trained model
    
    Args:
        model: Trained PyTorch model
        X_test, y_test: Test data
        device: Device to evaluate on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.to(device)
    model.eval()
    
    # Determine task type
    if len(torch.unique(y_test)) <= 10:  # Classification
        task_type = 'classification'
    else:  # Regression
        task_type = 'regression'
        y_test = y_test.float()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        # Process in batches to handle large datasets
        batch_size = 256
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size].to(device)
            batch_y = y_test[i:i+batch_size]
            
            outputs = model(batch_X)
            
            if task_type == 'classification':
                preds = torch.argmax(outputs, dim=1)
            else:
                preds = outputs.squeeze()
            
            predictions.extend(preds.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    metrics = {}
    
    if task_type == 'classification':
        metrics['accuracy'] = accuracy_score(targets, predictions)
        
        # Additional classification metrics
        unique_labels = np.unique(targets)
        if len(unique_labels) == 2:  # Binary classification
            from sklearn.metrics import precision_score, recall_score, f1_score
            metrics['precision'] = precision_score(targets, predictions, average='binary')
            metrics['recall'] = recall_score(targets, predictions, average='binary')
            metrics['f1_score'] = f1_score(targets, predictions, average='binary')
        else:  # Multi-class
            from sklearn.metrics import precision_score, recall_score, f1_score
            metrics['precision'] = precision_score(targets, predictions, average='macro')
            metrics['recall'] = recall_score(targets, predictions, average='macro')
            metrics['f1_score'] = f1_score(targets, predictions, average='macro')
    
    else:  # Regression
        metrics['mse'] = mean_squared_error(targets, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2_score'] = r2_score(targets, predictions)
        
        # Mean Absolute Error
        metrics['mae'] = np.mean(np.abs(targets - predictions))
    
    return metrics

def cross_validate_model(model_class, model_params, X, y, cv_folds=5, 
                        train_params=None, device='cpu'):
    """
    Perform cross-validation
    
    Args:
        model_class: Model class to instantiate
        model_params: Parameters for model initialization
        X, y: Full dataset
        cv_folds: Number of CV folds
        train_params: Training parameters
        device: Device to train on
    
    Returns:
        Dictionary with CV results
    """
    from sklearn.model_selection import KFold
    
    if train_params is None:
        train_params = {'epochs': 30, 'batch_size': 32, 'learning_rate': 0.001}
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_scores = []
    fold_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Training fold {fold + 1}/{cv_folds}")
        
        # Split data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Convert to tensors
        X_train_fold = torch.FloatTensor(X_train_fold)
        y_train_fold = torch.LongTensor(y_train_fold)
        X_val_fold = torch.FloatTensor(X_val_fold)
        y_val_fold = torch.LongTensor(y_val_fold)
        
        # Initialize model
        model = model_class(**model_params)
        
        # Train model
        history = train_model(
            model, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
            device=device, **train_params
        )
        
        # Evaluate
        metrics = evaluate_model(model, X_val_fold, y_val_fold, device=device)
        
        cv_scores.append(metrics)
        fold_histories.append(history)
    
    # Aggregate results
    cv_results = {}
    
    # Average metrics across folds
    for metric in cv_scores[0].keys():
        scores = [fold[metric] for fold in cv_scores]
        cv_results[f'{metric}_mean'] = np.mean(scores)
        cv_results[f'{metric}_std'] = np.std(scores)
    
    cv_results['fold_scores'] = cv_scores
    cv_results['fold_histories'] = fold_histories
    
    return cv_results

def hyperparameter_search(model_class, param_grid, X_train, y_train, X_val, y_val,
                         train_params=None, device='cpu'):
    """
    Simple grid search for hyperparameters
    
    Args:
        model_class: Model class to instantiate
        param_grid: Dictionary of parameters to search
        X_train, y_train: Training data
        X_val, y_val: Validation data
        train_params: Training parameters
        device: Device to train on
    
    Returns:
        Best parameters and results
    """
    from itertools import product
    
    if train_params is None:
        train_params = {'epochs': 30, 'batch_size': 32, 'learning_rate': 0.001}
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    best_score = -np.inf
    best_params = None
    results = []
    
    for param_combo in param_combinations:
        # Create parameter dictionary
        params = dict(zip(param_names, param_combo))
        print(f"Testing parameters: {params}")
        
        try:
            # Initialize model
            model = model_class(**params)
            
            # Train model
            history = train_model(
                model, X_train, y_train, X_val, y_val,
                device=device, **train_params
            )
            
            # Evaluate
            metrics = evaluate_model(model, X_val, y_val, device=device)
            
            # Use accuracy for classification, R² for regression
            score = metrics.get('accuracy', metrics.get('r2_score', 0))
            
            results.append({
                'params': params,
                'score': score,
                'metrics': metrics,
                'history': history
            })
            
            if score > best_score:
                best_score = score
                best_params = params
            
            print(f"  Score: {score:.4f}")
            
        except Exception as e:
            print(f"  Failed: {str(e)}")
            continue
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results
    }

# Example usage
if __name__ == "__main__":
    # Example of training a simple model
    from models.baseline import BaselineNN
    from utils.data_generator import generate_synthetic_data
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=1000, n_features=20)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Initialize model
    model = BaselineNN(input_dim=20, hidden_dims=[64, 32], n_classes=2)
    
    # Train model
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=20)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print("Final metrics:", metrics)
