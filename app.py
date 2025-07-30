import webbrowser
from threading import Timer
from flask import Flask, render_template, request, jsonify
import torch
import pickle
import logging
import sys
import os
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import TextGenerationModel
from utils import Tokenizer, get_device
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    global model, tokenizer, device
    
    try:
        # Get device
        device = get_device()
        
        # Load tokenizer
        tokenizer = Tokenizer.load(config.paths.tokenizer_file)
        logger.info("Tokenizer loaded successfully")
        
        # Load max sequence length
        with open(config.paths.max_sequence_len_file, 'rb') as f:
            max_sequence_len = pickle.load(f)
        tokenizer.max_sequence_len = max_sequence_len
        
        # Initialize model
        total_words = len(tokenizer.word_index) + 1
        model = TextGenerationModel(
            vocab_size=total_words,
            embedding_dim=config.model.embedding_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        ).to(device)
        
        # Load trained weights
        model.load_state_dict(torch.load(config.paths.model_file, map_location=device))
        model.eval()
        
        logger.info("Model loaded successfully")
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        logger.error("Please run train.py first to train the model")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def generate_text(seed_text: str, next_words: int, temperature: float = 1.0, top_k: int = 10, top_p: float = 0.9) -> str:
    """Generate text using the trained model"""
    try:
        if model is None or tokenizer is None:
            return "Error: Model not loaded. Please ensure the model has been trained."
        
        # Generate text using the model's generate method
        generated_text = model.generate(
            tokenizer=tokenizer,
            seed_text=seed_text,
            max_length=next_words,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        return f"Error generating text: {str(e)}"

@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text API endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        seed_text = data.get('seed_text', '').strip()
        next_words = int(data.get('next_words', 10))
        temperature = float(data.get('temperature', 1.0))
        top_k = int(data.get('top_k', 10))
        top_p = float(data.get('top_p', 0.9))
        
        # Validate inputs
        if not seed_text:
            return jsonify({'error': 'Seed text cannot be empty'}), 400
        
        if next_words <= 0 or next_words > 100:
            return jsonify({'error': 'Number of words must be between 1 and 100'}), 400
        
        if temperature <= 0 or temperature > 2.0:
            return jsonify({'error': 'Temperature must be between 0.1 and 2.0'}), 400
        
        if top_k <= 0 or top_k > 50:
            return jsonify({'error': 'Top-k must be between 1 and 50'}), 400
        
        if top_p <= 0 or top_p > 1.0:
            return jsonify({'error': 'Top-p must be between 0.1 and 1.0'}), 400
        
        # Generate text
        generated_text = generate_text(seed_text, next_words, temperature, top_k, top_p)
        
        return jsonify({
            'generated_text': generated_text,
            'seed_text': seed_text,
            'next_words': next_words,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if model is None or tokenizer is None:
            return jsonify({
                'status': 'unhealthy',
                'message': 'Model not loaded'
            }), 503
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None,
            'tokenizer_loaded': tokenizer is not None,
            'device': str(device) if device else 'unknown'
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        info = {
            'vocabulary_size': len(tokenizer.word_index) + 1 if tokenizer else 0,
            'embedding_dim': config.model.embedding_dim,
            'hidden_dim': config.model.hidden_dim,
            'num_layers': config.model.num_layers,
            'dropout': config.model.dropout,
            'max_sequence_len': tokenizer.max_sequence_len if tokenizer else 0,
            'device': str(device) if device else 'unknown'
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def open_browser():
    """Open browser when app starts"""
    if config.app.auto_open_browser:
        webbrowser.open_new(f"http://{config.app.host}:{config.app.port}/")

if __name__ == "__main__":
    # Load model and tokenizer
    if not load_model_and_tokenizer():
        logger.error("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Start browser
    if config.app.auto_open_browser:
        Timer(1, open_browser).start()
    
    # Run app
    logger.info(f"Starting Flask app on {config.app.host}:{config.app.port}")
    app.run(
        host=config.app.host,
        port=config.app.port,
        debug=config.app.debug
    )