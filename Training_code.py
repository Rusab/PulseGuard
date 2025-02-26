import os
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

# Set random seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ECGDataset(Dataset):
    """Dataset for loading and preprocessing ECG data."""
    
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.X[idx], dtype=torch.float32)
        
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return sample, label
        else:
            return sample

class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward network."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward network
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class TransformerModel(nn.Module):
    """Transformer model for ECG classification."""
    
    def __init__(self, sequence_length=5000, num_leads=12, embed_dim=64, num_heads=8, 
                 ff_dim=128, num_transformer_blocks=4, dropout=0.2):
        super(TransformerModel, self).__init__()
        
        # Initial projection
        self.projection = nn.Sequential(
            nn.Conv1d(num_leads, embed_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate sequence length after projection
        projected_seq_len = sequence_length // 4  # After Conv1d with stride=2 and MaxPool1d with stride=2
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_blocks)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, num_leads]
        # Transpose for 1D convolution [batch_size, num_leads, sequence_length]
        x = x.transpose(1, 2)
        
        # Apply initial projection
        x = self.projection(x)
        
        # Transpose back for transformer [batch_size, sequence_length, embed_dim]
        x = x.transpose(1, 2)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
            
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x

class CNNLSTMModel(nn.Module):
    """CNN-LSTM hybrid model for ECG classification."""
    
    def __init__(self, sequence_length=5000, num_leads=12, dropout=0.3):
        super(CNNLSTMModel, self).__init__()
        
        # CNN feature extraction
        self.cnn_layers = nn.Sequential(
            # First CNN block
            nn.Conv1d(num_leads, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Second CNN block
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Third CNN block
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate sequence length after CNN layers
        self.lstm_input_size = 256
        self.lstm_hidden_size = 128
        self.lstm_seq_len = sequence_length // 8  # After 3 MaxPool layers with stride=2
        
        # Bidirectional LSTM layers
        self.lstm_layers = nn.Sequential(
            nn.LSTM(input_size=self.lstm_input_size,
                    hidden_size=self.lstm_hidden_size,
                    num_layers=2,
                    dropout=dropout,
                    bidirectional=True,
                    batch_first=True)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * 2, 64),  # * 2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, num_leads]
        batch_size = x.shape[0]
        
        # Transpose for 1D convolution [batch_size, num_leads, sequence_length]
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        x = self.cnn_layers(x)
        
        # Transpose back for LSTM [batch_size, sequence_length, features]
        x = x.transpose(1, 2)
        
        # Apply LSTM layers
        lstm_out, (h_n, c_n) = self.lstm_layers[0](x)
        
        # Get the final hidden state from both directions
        h_n = h_n.view(2, 2, batch_size, self.lstm_hidden_size)  # [num_layers, num_directions, batch_size, hidden_size]
        final_hidden = torch.cat((h_n[-1, 0, :, :], h_n[-1, 1, :, :]), dim=1)
        
        # Classification
        x = self.classifier(final_hidden)
        
        return x

class PTBXLPipeline:
    def __init__(self, 
                 data_dir, 
                 metadata_csv,
                 output_dir="./results",
                 sampling_rate=100,
                 sequence_length=1000,  # 10 seconds at 500Hz
                 n_folds=1,
                 batch_size=32,
                 num_workers=0,
                 device=None):
        """
        Initialize the PTB-XL ECG processing pipeline.
        
        Args:
            data_dir (str): Directory containing the PTB-XL dataset files
            metadata_csv (str): Path to CSV file with filenames and labels
            output_dir (str): Directory to save results
            sampling_rate (int): ECG sampling rate (default: 500Hz)
            sequence_length (int): Number of time steps (default: 5000 for 10s)
            n_folds (int): Number of cross-validation folds
            batch_size (int): Batch size for training
            num_workers (int): Number of workers for data loading
            device (str): Device to use for training (default: cuda if available, else cpu)
        """
        self.data_dir = data_dir
        self.metadata_csv = metadata_csv
        self.output_dir = output_dir
        self.sampling_rate = sampling_rate
        self.sequence_length = sequence_length
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and prepare metadata
        self.load_metadata()

    def calculate_f1_score(self, y_true, y_pred):
        """
        Calculate binary F1 score.
        
        Args:
            y_true (torch.Tensor): True labels
            y_pred (torch.Tensor): Predicted labels (thresholded)
            
        Returns:
            float: F1 score
        """
        # Convert tensors to numpy arrays
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        
        # Calculate F1 score
        return f1_score(y_true, y_pred)

    def undersample_training_data(self, X_data, y_labels):
        """
        Undersample the majority class (NORM) to balance with minority class (MI)
        in the training set only.
        
        Args:
            X_data (numpy.ndarray): Input data
            y_labels (numpy.ndarray): Target labels
            
        Returns:
            tuple: (X_balanced, y_balanced) balanced dataset
        """
        # Count samples in each class
        mi_indices = np.where(y_labels == 1)[0]
        norm_indices = np.where(y_labels == 0)[0]
        
        n_mi = len(mi_indices)
        n_norm = len(norm_indices)
        
        print(f"Before undersampling - NORM: {n_norm}, MI: {n_mi}")
        
        # If NORM class is larger, undersample it
        if n_norm > n_mi:
            # Randomly select n_mi samples from NORM class
            np.random.seed(SEED)  # For reproducibility
            selected_norm_indices = np.random.choice(norm_indices, n_mi, replace=False)
            
            # Combine MI indices with selected NORM indices
            balanced_indices = np.concatenate([mi_indices, selected_norm_indices])
            
            # Create balanced dataset
            X_balanced = X_data[balanced_indices]
            y_balanced = y_labels[balanced_indices]
            
            print(f"After undersampling - NORM: {np.sum(y_balanced == 0)}, MI: {np.sum(y_balanced == 1)}")
            
            return X_balanced, y_balanced
        else:
            # No need for undersampling
            print("No undersampling needed, MI class is not minority")
            return X_data, y_labels
        
    def load_metadata(self):
        """Load and prepare the metadata from CSV file."""
        self.metadata = pd.read_csv(self.metadata_csv)
        print(f"Loaded metadata with {len(self.metadata)} records")
        
        # Check if the required columns exist
        required_columns = ['filename', 'label', 'split']
        for col in required_columns:
            if col not in self.metadata.columns:
                raise ValueError(f"Required column '{col}' not found in the metadata CSV")
        
        # Print some statistics
        print(f"Label distribution:\n{self.metadata['label'].value_counts()}")
        print(f"Split distribution:\n{self.metadata['split'].value_counts()}")

    def load_wfdb_record(self, filename):
        """
        Load a WFDB record from the PTB-XL dataset.
        
        Args:
            filename (str): Record filename without extension
            
        Returns:
            numpy.ndarray: ECG signal data with shape (sequence_length, 12)
        """
        try:
            record_path = os.path.join(self.data_dir, filename)
            record = wfdb.rdrecord(record_path)
            signals = record.p_signal
            
            # Ensure the signal has the expected length
            if signals.shape[0] < self.sequence_length:
                # Pad if shorter
                pad_length = self.sequence_length - signals.shape[0]
                signals = np.pad(signals, ((0, pad_length), (0, 0)), 'constant')
            elif signals.shape[0] > self.sequence_length:
                # Truncate if longer
                signals = signals[:self.sequence_length, :]
                
            return signals
        except Exception as e:
            print(f"Error loading record {filename}: {e}")
            # Return zeros if file can't be loaded
            return np.zeros((self.sequence_length, 12))

    def normalize_signal(self, signal):
        """
        Normalize ECG signal lead-wise.
        
        Args:
            signal (numpy.ndarray): ECG signal with shape (sequence_length, 12)
            
        Returns:
            numpy.ndarray: Normalized ECG signal
        """
        normalized = np.zeros_like(signal)
        
        for i in range(signal.shape[1]):  # For each lead
            lead_data = signal[:, i]
            mean = np.mean(lead_data)
            std = np.std(lead_data)
            # Avoid division by zero
            normalized[:, i] = (lead_data - mean) / (std if std > 0 else 1)
            
        return normalized

    def preprocess_data(self, files_subset=None):
        """
        Preprocess the ECG data by loading, normalizing, and preparing labels.
        
        Args:
            files_subset (list, optional): Subset of filenames to process
            
        Returns:
            tuple: (X_data, y_labels) preprocessed data and labels
        """
        if files_subset is None:
            files_subset = self.metadata
        
        X_data = []
        y_labels = []
        
        print(f"Preprocessing {len(files_subset)} records...")
        
        for idx, row in tqdm(files_subset.iterrows(), total=len(files_subset)):
            # Load record
            signal = self.load_wfdb_record(row['filename'])
            
            # Normalize signal
            signal = self.normalize_signal(signal)
            
            X_data.append(signal)
            y_labels.append(row['label'])
        
        X_data = np.array(X_data)
        y_labels = np.array(y_labels)
        
        print(f"Preprocessed data shape: {X_data.shape}")
        print(f"Labels shape: {y_labels.shape}")
        
        return X_data, y_labels

    def train_epoch(self, model, dataloader, criterion, optimizer, scheduler=None):
        """
        Train the model for one epoch.
        
        Args:
            model (torch.nn.Module): Model to train
            dataloader (torch.utils.data.DataLoader): Training dataloader
            criterion (torch.nn.Module): Loss function
            optimizer (torch.optim.Optimizer): Optimizer
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
            
        Returns:
            dict: Dictionary with training metrics
        """
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        for X, y in tqdm(dataloader, desc="Training", leave=False):
            X, y = X.to(self.device), y.to(self.device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X).view(-1)
            loss = criterion(outputs, y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Compute metrics
            running_loss += loss.item() * X.size(0)
            pred = (outputs > 0.5).float()
            
            # Store predictions and targets for F1 calculation
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
        
        # Apply scheduler if provided
        if scheduler is not None:
            scheduler.step()
        
        # Calculate metrics
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_f1 = f1_score(np.array(all_targets), np.array(all_preds))
        
        return {
            'loss': epoch_loss,
            'f1': epoch_f1
        }

    def validate(self, model, dataloader, criterion):
        """
        Validate the model.
        
        Args:
            model (torch.nn.Module): Model to validate
            dataloader (torch.utils.data.DataLoader): Validation dataloader
            criterion (torch.nn.Module): Loss function
            
        Returns:
            dict: Dictionary with validation metrics and predictions
        """
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for X, y in tqdm(dataloader, desc="Validation", leave=False):
                X, y = X.to(self.device), y.to(self.device)
                
                # Forward pass
                outputs = model(X).view(-1)
                loss = criterion(outputs, y)
                
                # Compute metrics
                running_loss += loss.item() * X.size(0)
                pred = (outputs > 0.5).float()
                
                # Save predictions and targets
                all_preds.extend(pred.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # Calculate metrics
        val_loss = running_loss / len(dataloader.dataset)
        val_f1 = f1_score(np.array(all_targets), np.array(all_preds))
        
        all_preds = np.array(all_preds)
        all_outputs = np.array(all_outputs)
        all_targets = np.array(all_targets)
        
        return {
            'loss': val_loss,
            'f1': val_f1,
            'predictions': all_preds,
            'outputs': all_outputs,
            'targets': all_targets
        }

    def calculate_metrics(self, val_results):
        """
        Calculate detailed metrics from validation results.
        
        Args:
            val_results (dict): Validation results
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        y_true = val_results['targets']
        y_pred = val_results['predictions']
        y_score = val_results['outputs']
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_score),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        return metrics

    def create_dataloaders(self, X_train, y_train, X_val, y_val):
        """
        Create training and validation dataloaders.
        
        Args:
            X_train (numpy.ndarray): Training data
            y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray): Validation data
            y_val (numpy.ndarray): Validation labels
            
        Returns:
            tuple: (train_dataloader, val_dataloader)
        """
        # Create datasets
        train_dataset = ECGDataset(X_train, y_train)
        val_dataset = ECGDataset(X_val, y_val)
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_dataloader, val_dataloader

    def train_and_evaluate_model(self, model_class, model_name, X_data, y_labels, epochs=50):
        """
        Train and evaluate a model using cross-validation.
        
        Args:
            model_class (class): PyTorch model class
            model_name (str): Name of the model
            X_data (numpy.ndarray): Input data
            y_labels (numpy.ndarray): Target labels
            epochs (int): Number of training epochs
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        # Prepare cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=SEED)
        
        # Metrics storage
        all_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': [],
            'confusion_matrices': [],
            'training_history': []
        }
        
        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_labels)):
            print(f"\n{'='*50}")
            print(f"Training {model_name} - Fold {fold+1}/{self.n_folds}")
            print(f"{'='*50}")
            
            # Split data
            X_train, X_val = X_data[train_idx], X_data[val_idx]
            y_train, y_val = y_labels[train_idx], y_labels[val_idx]
            
            # Apply undersampling to the training set only
            X_train, y_train = self.undersample_training_data(X_train, y_train)
            
            # Create dataloaders
            train_dataloader, val_dataloader = self.create_dataloaders(
                X_train, y_train, X_val, y_val
            )
            
            # (rest of the method remains the same, but update printing to show F1 instead of accuracy)
            
            # Initialize model
            model = model_class()
            model = model.to(self.device)
            
            # Print model summary
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model total parameters: {total_params:,}")
            print(f"Model trainable parameters: {trainable_params:,}")
            
            # Define loss function and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5, 
                verbose=True
            )
            
            # Early stopping setup
            early_stopping_patience = 10
            best_val_loss = float('inf')
            early_stopping_counter = 0
            best_model_path = os.path.join(self.output_dir, f"{model_name}_fold{fold+1}.pt")
            
            # Training history
            history = {
                'train_loss': [],
                'train_f1': [],  # Changed from train_acc
                'val_loss': [],
                'val_f1': []     # Changed from val_acc
            }
            
            # Training loop
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}/{epochs}")
                
                # Train model
                train_metrics = self.train_epoch(model, train_dataloader, criterion, optimizer)
                
                # Validate model
                val_metrics = self.validate(model, val_dataloader, criterion)
                
                # Update scheduler with validation loss
                scheduler.step(val_metrics['loss'])
                
                # Save training history
                history['train_loss'].append(train_metrics['loss'])
                history['train_f1'].append(train_metrics['f1'])  # Changed from accuracy
                history['val_loss'].append(val_metrics['loss'])
                history['val_f1'].append(val_metrics['f1'])      # Changed from accuracy
                
                # Print epoch results
                print(f"Train Loss: {train_metrics['loss']:.4f}, Train F1: {train_metrics['f1']:.4f}")  # Changed
                print(f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f}")         # Changed
                
                # Check for early stopping
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    early_stopping_counter = 0
                    
                    # Save best model
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Model saved to {best_model_path}")
                else:
                    early_stopping_counter += 1
                    print(f"EarlyStopping counter: {early_stopping_counter}/{early_stopping_patience}")
                    
                    if early_stopping_counter >= early_stopping_patience:
                        print("Early stopping triggered")
                        break
            
            # Load best model for final evaluation
            model.load_state_dict(torch.load(best_model_path))
            
            # Evaluate best model
            final_val_metrics = self.validate(model, val_dataloader, criterion)
            final_metrics = self.calculate_metrics(final_val_metrics)
            
            # Store metrics
            for key in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                all_metrics[key].append(final_metrics[key])
            all_metrics['confusion_matrices'].append(final_metrics['confusion_matrix'])
            all_metrics['training_history'].append(history)
            
            # Print final metrics
            print(f"\nFold {fold+1} Final Results:")
            print(f"Accuracy: {final_metrics['accuracy']:.4f}")
            print(f"Precision: {final_metrics['precision']:.4f}")
            print(f"Recall: {final_metrics['recall']:.4f}")
            print(f"F1 Score: {final_metrics['f1']:.4f}")
            print(f"AUC: {final_metrics['auc']:.4f}")
            print(f"Confusion Matrix:\n{final_metrics['confusion_matrix']}")
            
            # Plot learning curves
            self.plot_learning_curves(history, model_name, fold)
            
            # Clean up
            del model, optimizer, scheduler, train_dataloader, val_dataloader
            torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        
        # Calculate and print average metrics
        print(f"\n{'='*50}")
        print(f"{model_name} - Cross-Validation Results")
        print(f"{'='*50}")
        
        for key in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            avg_value = np.mean(all_metrics[key])
            std_value = np.std(all_metrics[key])
            print(f"Average {key}: {avg_value:.4f} ± {std_value:.4f}")
        
        return all_metrics
    
    def plot_learning_curves(self, history, model_name, fold):
        """
        Plot learning curves from training history.
        
        Args:
            history (dict): Training history
            model_name (str): Name of the model
            fold (int): Current fold number
        """
        plt.figure(figsize=(12, 5))
        
        # Loss curve
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} - Fold {fold+1} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # F1 score curve (changed from accuracy)
        plt.subplot(1, 2, 2)
        plt.plot(history['train_f1'], label='Training F1 Score')
        plt.plot(history['val_f1'], label='Validation F1 Score')
        plt.title(f'{model_name} - Fold {fold+1} F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(
            self.output_dir, 
            f"{model_name}_fold{fold+1}_learning_curves.png"
        ))
        plt.close()
    
    def run_pipeline(self):
        """
        Run the full pipeline: preprocess data, perform cross-validation 
        on train+val data, and evaluate on held-out test set.
        """
        print("Starting PTB-XL ECG NORM vs MI detection pipeline...")
        
        # Preprocess all data
        X_data, y_labels = self.preprocess_data()
        
        print(f"Class balance in dataset:")
        print(f"  - NORM (label 0): {np.sum(y_labels == 0)} records ({np.mean(y_labels == 0) * 100:.1f}%)")
        print(f"  - MI (label 1): {np.sum(y_labels == 1)} records ({np.mean(y_labels == 1) * 100:.1f}%)")
        
        # Split data according to predefined split in the CSV
        train_val_mask = self.metadata['split'].isin(['train', 'val']).values
        test_mask = self.metadata['split'] == 'test'
        
        # Extract train+val and test sets
        X_train_val = X_data[train_val_mask]
        y_train_val = y_labels[train_val_mask]
        X_test = X_data[test_mask]
        y_test = y_labels[test_mask]
        
        print(f"\nDataset splits:")
        print(f"  - Train+Val set: {len(X_train_val)} records")
        print(f"  - Test set: {len(X_test)} records")
        
        print("\nClass balance in Train+Val set:")
        print(f"  - NORM (label 0): {np.sum(y_train_val == 0)} records ({np.mean(y_train_val == 0) * 100:.1f}%)")
        print(f"  - MI (label 1): {np.sum(y_train_val == 1)} records ({np.mean(y_train_val == 1) * 100:.1f}%)")
        
        print("\nClass balance in Test set:")
        print(f"  - NORM (label 0): {np.sum(y_test == 0)} records ({np.mean(y_test == 0) * 100:.1f}%)")
        print(f"  - MI (label 1): {np.sum(y_test == 1)} records ({np.mean(y_test == 1) * 100:.1f}%)")
        
        # Train and evaluate Transformer model with cross-validation on train+val set
        print("\n" + "="*80)
        print("PHASE 1: Cross-validation on Train+Val set")
        print("="*80)
        
        # transformer_metrics = self.train_and_evaluate_model(
        #     TransformerModel,
        #     "Transformer",
        #     X_train_val,
        #     y_train_val,
        #     epochs=50
        # )
        
        # # Train and evaluate CNN-LSTM model with cross-validation on train+val set
        # cnn_lstm_metrics = self.train_and_evaluate_model(
        #     CNNLSTMModel,
        #     "CNN_LSTM",
        #     X_train_val,
        #     y_train_val,
        #     epochs=50
        # )
        
        # # Compare models from cross-validation
        # self.compare_models({
        #     "Transformer": transformer_metrics,
        #     "CNN_LSTM": cnn_lstm_metrics
        # })
        
        # # Train final models on entire train+val set and evaluate on test set
        # print("\n" + "="*80)
        # print("PHASE 2: Final evaluation on held-out Test set")
        # print("="*80)
        
        # Train final Transformer model
        final_transformer = TransformerModel().to(self.device)
        final_transformer_metrics = self.train_final_model(
            final_transformer, 
            "Final_Transformer", 
            X_train_val, 
            y_train_val, 
            X_test, 
            y_test,
            epochs=50
        )
        
        # Train final CNN-LSTM model
        final_cnn_lstm = CNNLSTMModel().to(self.device)
        final_cnn_lstm_metrics = self.train_final_model(
            final_cnn_lstm, 
            "Final_CNN_LSTM", 
            X_train_val, 
            y_train_val, 
            X_test, 
            y_test,
            epochs=50
        )
        
        # Compare final models on test set
        print("\nFinal Model Performance on Test Set:")
        print("-" * 80)
        print(f"{'Metric':<12} | {'Transformer':<25} | {'CNN-LSTM':<25}")
        print("-" * 80)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            transformer_val = final_transformer_metrics[metric]
            cnn_lstm_val = final_cnn_lstm_metrics[metric]
            print(f"{metric:<12} | {transformer_val:.4f} | {cnn_lstm_val:.4f}")
        
        print("\nPipeline completed successfully!")
    
    def train_final_model(self, model, model_name, X_train, y_train, X_test, y_test, epochs=50):
        """
        Train a final model on the entire training set and evaluate on the test set.
        
        Args:
            model (torch.nn.Module): PyTorch model to train
            model_name (str): Name of the model
            X_train (numpy.ndarray): Training data
            y_train (numpy.ndarray): Training labels
            X_test (numpy.ndarray): Test data
            y_test (numpy.ndarray): Test labels
            epochs (int): Maximum number of training epochs
            
        Returns:
            dict: Dictionary with test evaluation metrics
        """
        print(f"\nTraining final {model_name} model on entire training set...")
        
        # Apply undersampling to the training set
        X_train, y_train = self.undersample_training_data(X_train, y_train)
        
        # Create datasets and dataloaders
        train_dataset = ECGDataset(X_train, y_train)
        test_dataset = ECGDataset(X_test, y_test)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training setup
        best_val_loss = float('inf')
        early_stopping_counter = 0
        early_stopping_patience = 10
        best_model_path = os.path.join(self.output_dir, f"{model_name}_final.pt")
        
        # History tracking
        history = {
            'train_loss': [],
            'train_acc': []
        }
        
        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Train for one epoch
            train_metrics = self.train_epoch(model, train_dataloader, criterion, optimizer)
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['f1'])
            
            # Simple validation on a small subset to track progress and for early stopping
            # Create a small validation set from the training data
            val_size = min(len(X_train) // 10, 1000)  # 10% or max 1000 samples
            val_indices = np.random.choice(len(X_train), val_size, replace=False)
            X_val_sample = X_train[val_indices]
            y_val_sample = y_train[val_indices]
            
            val_dataset = ECGDataset(X_val_sample, y_val_sample)
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=self.num_workers
            )
            
            # Validate
            val_metrics = self.validate(model, val_dataloader, criterion)
            
            # Update learning rate based on validation loss
            scheduler.step(val_metrics['loss'])
            
            # Print progress
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train F1: {train_metrics['f1']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
            # Check for early stopping and model saving
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                early_stopping_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"Model saved to {best_model_path}")
            else:
                early_stopping_counter += 1
                print(f"EarlyStopping counter: {early_stopping_counter}/{early_stopping_patience}")
                
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    break
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(best_model_path))
        
        # Evaluate on test set
        print(f"\nEvaluating {model_name} on test set...")
        test_results = self.validate(model, test_dataloader, criterion)
        final_metrics = self.calculate_metrics(test_results)
        
        # Print test metrics
        print(f"Test Results for {model_name}:")
        print(f"Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Precision: {final_metrics['precision']:.4f}")
        print(f"Recall: {final_metrics['recall']:.4f}")
        print(f"F1 Score: {final_metrics['f1']:.4f}")
        print(f"AUC: {final_metrics['auc']:.4f}")
        print(f"Confusion Matrix:\n{final_metrics['confusion_matrix']}")
        
        # Plot and save learning curve
        plt.figure(figsize=(10, 4))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.title(f'{model_name} - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_final_training_curve.png"))
        plt.close()
        
        return final_metrics


# Example usage
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = PTBXLPipeline(
        data_dir=r"e:\Rusab\New folder\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        metadata_csv=r"e:\Rusab\New folder\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\norm_mi_binary_simple_100hz.csv",
        output_dir="./ptb_xl_results",
        n_folds=5,
        batch_size=32
    )
    
    # Run the pipeline
    pipeline.run_pipeline()