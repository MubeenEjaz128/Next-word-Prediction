import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for the text generation model"""
    embedding_dim: int = 100
    hidden_dim: int = 150
    num_layers: int = 1
    dropout: float = 0.2
    max_words: int = 10000
    max_sequence_len: int = 50

@dataclass
class TrainingConfig:
    """Configuration for training"""
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 50
    patience: int = 5  # Early stopping patience
    validation_split: float = 0.2
    min_loss_improvement: float = 0.001

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_length: int = 50
    temperature: float = 1.0
    top_k: int = 10
    top_p: float = 0.9

@dataclass
class AppConfig:
    """Configuration for the Flask app"""
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    auto_open_browser: bool = True

@dataclass
class PathsConfig:
    """Configuration for file paths"""
    data_file: str = "sherlock-holm.es_stories_plain-text_advs.txt"
    model_file: str = "sherlock_holmes_model.pth"
    tokenizer_file: str = "tokenizer.pkl"
    max_sequence_len_file: str = "max_sequence_len.pkl"

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.generation = GenerationConfig()
        self.app = AppConfig()
        self.paths = PathsConfig()
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        config = cls()
        
        # Model config
        config.model.embedding_dim = int(os.getenv('EMBEDDING_DIM', 100))
        config.model.hidden_dim = int(os.getenv('HIDDEN_DIM', 150))
        config.model.num_layers = int(os.getenv('NUM_LAYERS', 1))
        config.model.dropout = float(os.getenv('DROPOUT', 0.2))
        config.model.max_words = int(os.getenv('MAX_WORDS', 10000))
        config.model.max_sequence_len = int(os.getenv('MAX_SEQUENCE_LEN', 50))
        
        # Training config
        config.training.batch_size = int(os.getenv('BATCH_SIZE', 32))
        config.training.learning_rate = float(os.getenv('LEARNING_RATE', 0.001))
        config.training.epochs = int(os.getenv('EPOCHS', 50))
        config.training.patience = int(os.getenv('PATIENCE', 5))
        config.training.validation_split = float(os.getenv('VALIDATION_SPLIT', 0.2))
        config.training.min_loss_improvement = float(os.getenv('MIN_LOSS_IMPROVEMENT', 0.001))
        
        # Generation config
        config.generation.max_length = int(os.getenv('MAX_LENGTH', 50))
        config.generation.temperature = float(os.getenv('TEMPERATURE', 1.0))
        config.generation.top_k = int(os.getenv('TOP_K', 10))
        config.generation.top_p = float(os.getenv('TOP_P', 0.9))
        
        # App config
        config.app.host = os.getenv('HOST', '127.0.0.1')
        config.app.port = int(os.getenv('PORT', 5000))
        config.app.debug = os.getenv('DEBUG', 'False').lower() == 'true'
        config.app.auto_open_browser = os.getenv('AUTO_OPEN_BROWSER', 'True').lower() == 'true'
        
        # Paths config
        config.paths.data_file = os.getenv('DATA_FILE', 'sherlock-holm.es_stories_plain-text_advs.txt')
        config.paths.model_file = os.getenv('MODEL_FILE', 'sherlock_holmes_model.pth')
        config.paths.tokenizer_file = os.getenv('TOKENIZER_FILE', 'tokenizer.pkl')
        config.paths.max_sequence_len_file = os.getenv('MAX_SEQUENCE_LEN_FILE', 'max_sequence_len.pkl')
        
        return config

# Global config instance
config = Config.from_env() 