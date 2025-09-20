import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongAttention(nn.Module):
    """
    Implements Luong's 'general' style attention.
    """
    def __init__(self, hidden_size):
        """
        Initializes the Attention module.

        Args:
            hidden_size (int): The hidden size of the encoder and decoder.
                               Must be the same.
        """
        super(LuongAttention, self).__init__()
        # Luong's 'general' score requires a linear layer to align dimensions
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Calculates attention weights and context vector.

        Args:
            decoder_hidden (torch.Tensor): The previous hidden state from the decoder.
                                           Shape: (num_layers, batch_size, hidden_size)
            encoder_outputs (torch.Tensor): The output hidden states from the encoder.
                                            Shape: (batch_size, sequence_length, hidden_size)

        Returns:
            tuple: A tuple containing:
                - context_vector (torch.Tensor): The context vector, a weighted sum of encoder outputs.
                                                 Shape: (batch_size, 1, hidden_size)
                - attention_weights (torch.Tensor): The attention weights.
                                                   Shape: (batch_size, 1, sequence_length)
        """
        # Note: We only use the top layer's hidden state from the decoder
        # Squeeze to remove the num_layers dimension if it's 1, or select the last layer
        last_layer_hidden = decoder_hidden[-1].unsqueeze(1) # Shape: (batch_size, 1, hidden_size)

        # Calculate alignment scores (general style)
        # 1. Project encoder outputs through the linear layer
        projected_encoder_outputs = self.attn(encoder_outputs) # Shape: (batch_size, seq_len, hidden_size)

        # 2. Calculate the dot product between decoder hidden and projected encoder outputs
        #    bmm is batch matrix multiplication
        attn_scores = torch.bmm(last_layer_hidden, projected_encoder_outputs.transpose(1, 2))
        # Shape: (batch_size, 1, sequence_length)

        # Apply softmax to get the attention weights (probabilities)
        attention_weights = F.softmax(attn_scores, dim=2)
        # Shape: (batch_size, 1, sequence_length)

        # Calculate the context vector by multiplying weights with encoder outputs
        context_vector = torch.bmm(attention_weights, encoder_outputs)
        # Shape: (batch_size, 1, hidden_size)

        return context_vector, attention_weights