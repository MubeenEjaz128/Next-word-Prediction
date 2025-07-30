import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=150, num_layers=1, dropout=0.2):
        super(TextGenerationModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Take the last output
        output = self.fc(lstm_out[:, -1, :])
        # output shape: (batch_size, vocab_size)
        
        return output
    
    def generate(self, tokenizer, seed_text, max_length=50, temperature=1.0, top_k=10, top_p=0.9):
        """Generate text using the trained model"""
        self.eval()
        generated_text = seed_text
        
        with torch.no_grad():
            for _ in range(max_length):
                # Tokenize the current text
                token_list = tokenizer.texts_to_sequences([generated_text])[0]
                if not token_list:
                    token_list = [0]
                
                # Pad or truncate to match training sequence length
                seq_len = len(token_list)
                if seq_len < tokenizer.max_sequence_len - 1:
                    token_list = [0] * (tokenizer.max_sequence_len - 1 - seq_len) + token_list
                else:
                    token_list = token_list[-(tokenizer.max_sequence_len - 1):]
                
                # Convert to tensor
                token_tensor = torch.tensor([token_list], dtype=torch.long).to(next(self.parameters()).device)
                
                # Get model prediction
                output = self(token_tensor)
                
                # Apply temperature and top-k sampling
                if temperature != 1.0:
                    output = output / temperature
                
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(output, top_k, dim=-1)
                    output = torch.full_like(output, float('-inf'))
                    output.scatter_(-1, top_k_indices, top_k_logits)
                
                # Sample from the distribution
                probs = F.softmax(output, dim=-1)
                predicted = torch.multinomial(probs, 1).item()
                
                # Convert to word
                output_word = tokenizer.index_word.get(predicted, 'unknown')
                generated_text += " " + output_word
                
                # Stop if we generate too many unknown words
                if generated_text.count('unknown') > 5:
                    break
        
        return generated_text 