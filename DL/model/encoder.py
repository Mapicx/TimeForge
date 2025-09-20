import torch
import torch.nn as nn

class GRUEncoder(nn.Module):
    """
    A GRU-based Encoder for a sequence-to-sequence model.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        """
        Initializes the Encoder.

        Args:
            input_size (int): The number of features in the input data.
                              For example, if you are using just the clock and ephemeris errors, this would be 2.
            hidden_size (int): The number of features in the hidden state h.
                               This is a hyperparameter you can tune.
            num_layers (int, optional): The number of recurrent layers. Defaults to 1.
            dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs
                                       of each GRU layer except the last layer. Defaults to 0.1.
        """
        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Makes tensor shapes more intuitive: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, input_sequence):
        """
        Processes the input sequence through the GRU.

        Args:
            input_sequence (torch.Tensor): The input data for the encoder.
                                           Shape: (batch_size, sequence_length, input_size)

        Returns:
            tuple: A tuple containing:
                - encoder_outputs (torch.Tensor): The output hidden state for each time step.
                                                  Shape: (batch_size, sequence_length, hidden_size)
                                                  This is what the attention mechanism will use.
                - hidden_state (torch.Tensor): The final hidden state of the GRU.
                                               Shape: (num_layers, batch_size, hidden_size)
                                               This will be used to initialize the decoder's hidden state.
        """
        # The GRU layer returns the full output sequence and the final hidden state
        encoder_outputs, hidden_state = self.gru(input_sequence)
        return encoder_outputs, hidden_state

# --- Example Usage ---
if __name__ == '__main__':
    # Define some hyperparameters for the example
    batch_size = 32
    sequence_length = 672  # 7 days * 24 hours * 4 intervals/hour
    input_features = 5     # Example: using 5 features from your dataset
    hidden_features = 256  # Example hidden size
    num_gru_layers = 2     # Using a stacked GRU

    # 1. Create a dummy input tensor
    # This simulates a batch of 32 sequences, each 672 time steps long, with 5 features per step.
    dummy_input = torch.randn(batch_size, sequence_length, input_features)

    # 2. Instantiate the encoder
    encoder = GRUEncoder(
        input_size=input_features,
        hidden_size=hidden_features,
        num_layers=num_gru_layers
    )

    # 3. Pass the dummy input through the encoder
    encoder_outputs, final_hidden_state = encoder(dummy_input)

    # 4. Print the shapes of the outputs to verify
    print("--- Encoder Test ---")
    print(f"Input shape:          {dummy_input.shape}")
    print(f"Encoder outputs shape:  {encoder_outputs.shape}")
    print(f"Final hidden state shape: {final_hidden_state.shape}")
    print("\nNote: The 'Encoder outputs' tensor is the one you will feed to your attention mechanism.")