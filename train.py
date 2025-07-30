import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import logging
from typing import Tuple, List
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import TextGenerationModel
from utils import Tokenizer, create_sequences, prepare_data, get_device, save_model
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def load_data(filepath: str) -> str:
    """Load text data from file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        logger.info(f"Loaded text data from {filepath}")
        return text
    except FileNotFoundError:
        logger.error(f"Data file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_data_loaders(X: torch.Tensor, y: torch.Tensor, batch_size: int, 
                       validation_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    # Split data into train and validation
    dataset_size = len(X)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    # Shuffle indices
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create datasets
    train_dataset = TensorDataset(X[train_indices], y[train_indices])
    val_dataset = TensorDataset(X[val_indices], y[val_indices])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Created data loaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    return train_loader, val_loader

def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, 
                  device: torch.device) -> float:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler,
                device: torch.device, epochs: int, patience: int = 5) -> List[Tuple[float, float]]:
    """Train the model with early stopping and learning rate scheduling"""
    
    early_stopping = EarlyStopping(patience=patience)
    history = []
    
    logger.info("Starting training...")
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log progress
        logger.info(f'Epoch {epoch+1}/{epochs}:')
        logger.info(f'  Training Loss: {train_loss:.4f}')
        logger.info(f'  Validation Loss: {val_loss:.4f}')
        logger.info(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        history.append((train_loss, val_loss))
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    logger.info("Training completed!")
    return history

def main():
    """Main training function"""
    try:
        # Get device
        device = get_device()
        
        # Load data
        text = load_data(config.paths.data_file)
        logger.info(f"Text length: {len(text)} characters")
        
        # Initialize tokenizer
        tokenizer = Tokenizer(
            max_words=config.model.max_words,
            max_sequence_len=config.model.max_sequence_len
        )
        
        # Fit tokenizer on text
        tokenizer.fit_on_texts([text])
        total_words = len(tokenizer.word_index) + 1
        logger.info(f"Vocabulary size: {total_words}")
        
        # Create sequences
        input_sequences = create_sequences(text, tokenizer)
        logger.info(f"Created {len(input_sequences)} input sequences")
        
        if not input_sequences:
            logger.error("No valid sequences generated. Check input text or tokenizer.")
            return
        
        # Prepare data
        X, y = prepare_data(input_sequences, config.model.max_sequence_len)
        logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            X, y, 
            batch_size=config.training.batch_size,
            validation_split=config.training.validation_split
        )
        
        # Initialize model
        model = TextGenerationModel(
            vocab_size=total_words,
            embedding_dim=config.model.embedding_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        ).to(device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=config.training.epochs,
            patience=config.training.patience
        )
        
        # Save model and tokenizer
        save_model(model, config.paths.model_file)
        tokenizer.save(config.paths.tokenizer_file)
        
        # Save max sequence length
        with open(config.paths.max_sequence_len_file, 'wb') as f:
            pickle.dump(config.model.max_sequence_len, f)
        
        logger.info("Model and tokenizer saved successfully!")
        
        # Test generation
        test_text = "Sherlock Holmes was"
        generated = model.generate(
            tokenizer, 
            test_text, 
            max_length=10,
            temperature=0.8,
            top_k=5
        )
        logger.info(f"Test generation: {generated}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()