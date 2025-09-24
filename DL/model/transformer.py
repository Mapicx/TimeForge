import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoderModel(nn.Module):
    """
    Transformer Encoder model updated to use an embedding layer for categorical features.
    """
    def __init__(self, num_continuous_features, num_satellites, embedding_dim, d_model, n_heads, n_layers, d_ff, dropout, output_seq_len, input_seq_len):
        super(TransformerEncoderModel, self).__init__()
        self.d_model = d_model
        
        # Embedding layer for the satellite ID (PRN)
        self.embedding = nn.Embedding(num_embeddings=num_satellites, embedding_dim=embedding_dim)

        # A single linear layer to project the combined features to d_model
        # The input size is the number of continuous features + the embedding dimension
        self.encoder = nn.Linear(num_continuous_features + embedding_dim, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        
        # Final linear layer to produce the forecast
        self.decoder = nn.Linear(d_model * input_seq_len, output_seq_len)

    def forward(self, continuous_src, prn_src):
        """
        Forward pass for the model.
        Args:
            continuous_src (Tensor): Continuous features, shape [batch_size, seq_len, num_continuous_features]
            prn_src (Tensor): PRN indices, shape [batch_size, 1]
        """
        # Get the embedding vector for the satellite, shape: [batch_size, 1, embedding_dim]
        prn_embedding = self.embedding(prn_src)
        
        # Repeat the embedding for each time step in the sequence
        # New shape: [batch_size, seq_len, embedding_dim]
        prn_embedding = prn_embedding.repeat(1, continuous_src.size(1), 1)
        
        # Concatenate the continuous features and the repeated embedding vector
        # New shape: [batch_size, seq_len, num_continuous_features + embedding_dim]
        src = torch.cat([continuous_src, prn_embedding], dim=2)
        
        # --- Standard Transformer forward pass ---
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.reshape(output.size(0), -1) # Flatten the sequence
        output = self.decoder(output)
        
        return output