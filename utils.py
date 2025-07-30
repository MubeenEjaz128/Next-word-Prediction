import re
import pickle
import torch
from collections import defaultdict
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Tokenizer:
    """Custom tokenizer for text preprocessing and tokenization"""
    
    def __init__(self, max_words=10000, max_sequence_len=50):
        self.word_index = {}
        self.index_word = {}
        self.word_counts = defaultdict(int)
        self.max_words = max_words
        self.max_sequence_len = max_sequence_len
        
    def clean_text(self, text: str) -> str:
        """Clean text by removing special characters and converting to lowercase"""
        # Remove special characters but keep spaces and basic punctuation
        cleaned = re.sub(r'[^\w\s\.\,\!\?]', '', text.lower())
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def fit_on_texts(self, texts: List[str]) -> None:
        """Fit the tokenizer on a list of texts"""
        all_words = []
        
        for text in texts:
            cleaned_text = self.clean_text(text)
            words = cleaned_text.split()
            all_words.extend(words)
        
        # Count word frequencies
        for word in all_words:
            self.word_counts[word] += 1
        
        # Sort by frequency and take top max_words
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:self.max_words]
        
        # Create word to index mapping
        for i, (word, _) in enumerate(sorted_words):
            self.word_index[word] = i + 1
            self.index_word[i + 1] = word
        
        logger.info(f"Tokenizer fitted on {len(self.word_index)} unique words")
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert texts to sequences of token indices"""
        sequences = []
        
        for text in texts:
            cleaned_text = self.clean_text(text)
            words = cleaned_text.split()
            sequence = [self.word_index.get(word, 0) for word in words]
            sequences.append(sequence)
        
        return sequences
    
    def sequences_to_texts(self, sequences: List[List[int]]) -> List[str]:
        """Convert sequences back to texts"""
        texts = []
        
        for sequence in sequences:
            words = [self.index_word.get(idx, 'unknown') for idx in sequence if idx != 0]
            text = ' '.join(words)
            texts.append(text)
        
        return texts
    
    def save(self, filepath: str) -> None:
        """Save tokenizer to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Tokenizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Tokenizer':
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            tokenizer = pickle.load(f)
        logger.info(f"Tokenizer loaded from {filepath}")
        return tokenizer

def create_sequences(text: str, tokenizer: Tokenizer, min_length: int = 2) -> tuple:
    """Create input sequences for training"""
    input_sequences = []
    
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        token_list = tokenizer.texts_to_sequences([sentence])[0]
        
        # Create n-gram sequences
        for i in range(min_length, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    
    return input_sequences

def prepare_data(sequences: List[List[int]], max_sequence_len: int) -> tuple:
    """Prepare data for training"""
    # Pad sequences
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_sequence_len:
            padded_seq = [0] * (max_sequence_len - len(seq)) + seq
        else:
            padded_seq = seq[-max_sequence_len:]
        padded_sequences.append(padded_seq)
    
    # Convert to numpy array first, then to tensor
    import numpy as np
    padded_sequences = np.array(padded_sequences)
    
    # Split into X and y
    X = torch.tensor(padded_sequences[:, :-1], dtype=torch.long)
    y = torch.tensor(padded_sequences[:, -1], dtype=torch.long)
    
    return X, y

def load_model(model_path: str, vocab_size: int, device: str = 'cpu') -> torch.nn.Module:
    """Load a trained model"""
    from models import TextGenerationModel
    
    model = TextGenerationModel(vocab_size=vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")
    return model

def save_model(model: torch.nn.Module, model_path: str) -> None:
    """Save a trained model"""
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    return device 