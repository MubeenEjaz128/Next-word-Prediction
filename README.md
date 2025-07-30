# Sherlock Holmes Text Generator

A sophisticated text generation system trained on Sherlock Holmes stories using LSTM neural networks. This project provides both a training pipeline and a web interface for generating Sherlock Holmes-style text.

---

## 🚀 Features

---

### Core Features
- **LSTM-based Text Generation**: Advanced neural network architecture for natural text generation
- **Custom Tokenizer**: Optimized text preprocessing and tokenization
- **Web Interface**: Modern, responsive web UI built with Flask and Tailwind CSS
- **Advanced Generation Controls**: Temperature and Top-K sampling for creative control
- **Real-time Generation**: Instant text generation with loading states and error handling

---

### Technical Features
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Configuration Management**: Environment-based configuration system
- **Early Stopping**: Prevents overfitting during training
- **Learning Rate Scheduling**: Adaptive learning rate for better convergence
- **Validation Split**: Proper train/validation data separation
- **Logging**: Comprehensive logging for debugging and monitoring
- **Health Checks**: API endpoints for monitoring system status

---

## 📁 Project Structure

```
sherlock-holmes-text-generator/
├── app.py                 # Flask web application
├── train.py              # Training script
├── models.py             # Neural network model definition
├── utils.py              # Utility functions and tokenizer
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── templates/
│   └── index.html       # Modern web interface
├── sherlock-holm.es_stories_plain-text_advs.txt  # Training data
├── sherlock_holmes_model.pth    # Trained model (generated)
├── tokenizer.pkl        # Tokenizer (generated)
├── max_sequence_len.pkl # Sequence length (generated)
└── training.log         # Training logs (generated)
```

---

## 🛠️ Installation

---

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

---

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sherlock-holmes-text-generator
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download training data**
   - Ensure `sherlock-holm.es_stories_plain-text_advs.txt` is in the project directory
   - This file contains the Sherlock Holmes stories used for training

---

## 🎯 Usage

### Training the Model

1. **Start training**
   ```bash
   python train.py
   ```

   The training script will:
   - Load and preprocess the Sherlock Holmes text data
   - Create a custom tokenizer
   - Train an LSTM model with early stopping
   - Save the trained model and tokenizer
   - Display training progress and final test generation

2. **Training configuration**
   You can customize training parameters by setting environment variables:
   ```bash
   export EPOCHS=100
   export BATCH_SIZE=64
   export LEARNING_RATE=0.001
   export EMBEDDING_DIM=200
   export HIDDEN_DIM=300
   python train.py
   ```

### Running the Web Application

1. **Start the Flask app**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your browser to `http://127.0.0.1:5000`
   - The app will automatically open in your default browser

3. **Generate text**
   - Enter seed text in the text area
   - Adjust generation parameters:
     - **Number of Words**: How many words to generate (1-100)
     - **Temperature**: Controls creativity (0.1-2.0)
     - **Top-K**: Controls diversity (1-50)
   - Click "Generate Text" to create Sherlock Holmes-style text

### API Endpoints

The application provides several API endpoints:

- `GET /` - Web interface
- `POST /generate` - Generate text
- `GET /health` - Health check
- `GET /model_info` - Model information

#### Generate Text API
```bash
curl -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "seed_text": "Sherlock Holmes was",
    "next_words": 20,
    "temperature": 1.0,
    "top_k": 10
  }'
```

## ⚙️ Configuration

### Environment Variables

You can customize the application behavior using environment variables:

#### Model Configuration
- `EMBEDDING_DIM`: Embedding dimension (default: 100)
- `HIDDEN_DIM`: Hidden layer dimension (default: 150)
- `NUM_LAYERS`: Number of LSTM layers (default: 1)
- `DROPOUT`: Dropout rate (default: 0.2)
- `MAX_WORDS`: Maximum vocabulary size (default: 10000)
- `MAX_SEQUENCE_LEN`: Maximum sequence length (default: 50)

#### Training Configuration
- `BATCH_SIZE`: Training batch size (default: 32)
- `LEARNING_RATE`: Learning rate (default: 0.001)
- `EPOCHS`: Number of training epochs (default: 50)
- `PATIENCE`: Early stopping patience (default: 5)
- `VALIDATION_SPLIT`: Validation data fraction (default: 0.2)

#### Generation Configuration
- `MAX_LENGTH`: Maximum generation length (default: 50)
- `TEMPERATURE`: Default temperature (default: 1.0)
- `TOP_K`: Default top-k value (default: 10)

#### Application Configuration
- `HOST`: Flask host (default: 127.0.0.1)
- `PORT`: Flask port (default: 5000)
- `DEBUG`: Debug mode (default: False)
- `AUTO_OPEN_BROWSER`: Auto-open browser (default: True)

## 🧠 Model Architecture

The text generation model uses a sophisticated LSTM architecture:

- **Embedding Layer**: Converts token indices to dense vectors
- **LSTM Layers**: Processes sequential data with optional dropout
- **Dropout Layer**: Prevents overfitting
- **Output Layer**: Projects to vocabulary size for next-word prediction

### Key Features:
- **Weight Initialization**: Xavier uniform initialization for better training
- **Padding Handling**: Proper handling of variable-length sequences
- **Temperature Sampling**: Controls randomness in generation
- **Top-K Sampling**: Limits vocabulary choices for more focused output

## 📊 Training Process

The training pipeline includes several advanced features:

1. **Data Preprocessing**
   - Text cleaning and normalization
   - Custom tokenizer with frequency-based vocabulary
   - Sequence creation with n-gram approach

2. **Model Training**
   - Batch processing with DataLoader
   - Train/validation split
   - Early stopping to prevent overfitting
   - Learning rate scheduling
   - Comprehensive logging

3. **Evaluation**
   - Validation loss monitoring
   - Test generation for quality assessment
   - Model and tokenizer persistence

## 🎨 Web Interface

The modern web interface provides:

- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Controls**: Interactive sliders for generation parameters
- **Loading States**: Visual feedback during generation
- **Error Handling**: User-friendly error messages
- **Model Information**: Display of model statistics
- **Modern UI**: Clean, professional design with Tailwind CSS

## 🔧 Development

### Adding New Features

1. **Model Improvements**
   - Modify `models.py` to add new architectures
   - Update `config.py` for new parameters
   - Test with `train.py`

2. **Web Interface**
   - Edit `templates/index.html` for UI changes
   - Update `app.py` for new API endpoints

3. **Training Enhancements**
   - Modify `train.py` for new training features
   - Update `utils.py` for new preprocessing steps

### Testing

```bash
# Test model loading
python -c "from app import load_model_and_tokenizer; print(load_model_and_tokenizer())"

# Test generation
python -c "from models import TextGenerationModel; print('Model imported successfully')"
```

## 📈 Performance

### Training Performance
- **Training Time**: ~10-30 minutes on CPU, ~5-15 minutes on GPU
- **Memory Usage**: ~2-4GB RAM depending on batch size
- **Model Size**: ~50-100MB depending on vocabulary size

### Generation Performance
- **Inference Time**: ~0.1-1 second per word
- **Memory Usage**: ~500MB-1GB during inference
- **Concurrent Users**: Supports multiple simultaneous users

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   export BATCH_SIZE=16  # Reduce batch size
   ```

2. **Model Files Not Found**
   ```bash
   python train.py  # Train the model first
   ```

3. **Port Already in Use**
   ```bash
   export PORT=5001  # Use different port
   ```

4. **Slow Generation**
   - Ensure model is loaded on GPU if available
   - Reduce `next_words` parameter
   - Lower `top_k` value

### Logs

Check the following log files for debugging:
- `training.log`: Training progress and errors
- Flask console output: Web application logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Arthur Conan Doyle**: For the original Sherlock Holmes stories
- **PyTorch**: For the deep learning framework
- **Flask**: For the web framework
- **Tailwind CSS**: For the modern UI components

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the logs for error messages
3. Open an issue on the repository
4. Contact the maintainers

---

**Happy generating! 🕵️‍♂️** #   N e x t - w o r d - P r e d i c t i o n 
 
 
