import torch
import torch.nn as nn
import torch.nn.functional as F

from .loung_attention import LuongAttention

class GRUDecoder(nn.Module):
    """
    A GRU-based Decoder with an Attention mechanism.
    """
    def __init__(self, output_size, hidden_size, num_layers=1, dropout=0.1):
        """
        Initializes the Decoder.

        Args:
            output_size (int): The number of features in the output prediction.
                               For your problem, this will likely be 1.
            hidden_size (int): The number of features in the hidden state h.
                               Must match the encoder's hidden_size.
            num_layers (int, optional): The number of recurrent layers. Defaults to 1.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(GRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # The input to the GRU at each step will be the concatenated
        # context vector and the previous decoder input.
        # So, the input size is hidden_size (from context) + output_size (from previous prediction)
        self.gru = nn.GRU(
            input_size=hidden_size + output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # The attention mechanism
        self.attention = LuongAttention(hidden_size)

        # A final fully-connected layer to map the GRU output to the desired prediction size
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, decoder_input, decoder_hidden, encoder_outputs):
        """
        Performs one decoding step.

        Args:
            decoder_input (torch.Tensor): The input for the current time step.
                                          Typically the previous prediction or a "start" token.
                                          Shape: (batch_size, 1, output_size)
            decoder_hidden (torch.Tensor): The previous hidden state of the decoder.
                                           Shape: (num_layers, batch_size, hidden_size)
            encoder_outputs (torch.Tensor): All hidden states from the encoder.
                                            Shape: (batch_size, sequence_length, hidden_size)

        Returns:
            tuple: A tuple containing:
                - prediction (torch.Tensor): The output prediction for this time step.
                                             Shape: (batch_size, 1, output_size)
                - decoder_hidden (torch.Tensor): The new hidden state of the decoder.
                                                 Shape: (num_layers, batch_size, hidden_size)
                - attention_weights (torch.Tensor): The attention weights for this step (for visualization).
                                                    Shape: (batch_size, 1, sequence_length)
        """
        # 1. Calculate the context vector using the attention mechanism
        context_vector, attention_weights = self.attention(decoder_hidden, encoder_outputs)
        # context_vector shape: (batch_size, 1, hidden_size)

        # 2. Combine the context vector and the decoder input
        #    This creates the rich input for the GRU cell
        gru_input = torch.cat((decoder_input, context_vector), dim=2)
        # gru_input shape: (batch_size, 1, hidden_size + output_size)

        # 3. Pass the combined input and previous hidden state through the GRU
        gru_output, decoder_hidden = self.gru(gru_input, decoder_hidden)
        # gru_output shape: (batch_size, 1, hidden_size)
        # decoder_hidden shape: (num_layers, batch_size, hidden_size)

        # 4. Pass the GRU output through the final linear layer to get the prediction
        prediction = self.out(gru_output)
        # prediction shape: (batch_size, 1, output_size)

        return prediction, decoder_hidden, attention_weights