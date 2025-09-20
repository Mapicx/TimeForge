import torch
import torch.nn as nn
import random
import torch.nn.functional as F

from .decoder import GRUDecoder
from .encoder import GRUEncoder


class Seq2Seq(nn.Module):
    """
    The main model that combines the Encoder and Decoder with Attention.
    """
    def __init__(self, encoder, decoder, device):
        """
        Initializes the Seq2Seq model.

        Args:
            encoder (GRUEncoder): The encoder module.
            decoder (GRUDecoder): The decoder module.
            device (torch.device): The device (e.g., 'cuda' or 'cpu') to run the model on.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source_sequence, target_sequence=None, teacher_forcing_ratio=0.5):
        """
        Performs the forward pass for the entire model.

        Args:
            source_sequence (torch.Tensor): The input sequence for the encoder.
                                            Shape: (batch_size, source_len, input_features)
            target_sequence (torch.Tensor, optional): The ground truth sequence for teacher forcing.
                                                      Shape: (batch_size, target_len, output_features)
            teacher_forcing_ratio (float, optional): The probability of using teacher forcing.
                                                     Defaults to 0.5.

        Returns:
            torch.Tensor: The model's predictions.
                          Shape: (batch_size, target_len, output_features)
        """
        batch_size = source_sequence.shape[0]
        # Determine target length from target_sequence if provided, else from a fixed config
        target_len = target_sequence.shape[1] if target_sequence is not None else 96 # (24 hours * 4)
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, self.decoder.output_size).to(self.device)

        # 1. Pass the source sequence through the encoder
        encoder_outputs, hidden = self.encoder(source_sequence)

        # 2. The first input to the decoder is the last known value from the source sequence
        #    (or you could use a special <SOS> token if you prefer)
        decoder_input = source_sequence[:, -1, :self.decoder.output_size].unsqueeze(1)

        # 3. Loop for each time step in the target sequence
        for t in range(target_len):
            # Pass the input, hidden state, and encoder outputs to the decoder
            decoder_output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)

            # Place the prediction in the outputs tensor
            outputs[:, t] = decoder_output.squeeze(1)
            
            # Decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            if teacher_force and target_sequence is not None:
                # Use the actual ground truth as the next input
                decoder_input = target_sequence[:, t, :].unsqueeze(1)
            else:
                # Use the model's own prediction as the next input
                decoder_input = decoder_output

        return outputs

# --- Example Usage ---
if __name__ == '__main__':
    # Define hyperparameters
    INPUT_DIM = 5      # Number of input features
    OUTPUT_DIM = 1     # Predicting a single value (e.g., clock error)
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.1
    
    SOURCE_SEQ_LEN = 672 # 7 days
    TARGET_SEQ_LEN = 96  # 24 hours (8th day)
    BATCH_SIZE = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Instantiate the components
    encoder = GRUEncoder(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = GRUDecoder(OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)

    # 2. Instantiate the main model
    model = Seq2Seq(encoder, decoder, device).to(device)

    # 3. Create dummy input and target tensors
    dummy_source = torch.randn(BATCH_SIZE, SOURCE_SEQ_LEN, INPUT_DIM).to(device)
    dummy_target = torch.randn(BATCH_SIZE, TARGET_SEQ_LEN, OUTPUT_DIM).to(device) # For teacher forcing

    # 4. Perform a forward pass
    predictions = model(dummy_source, dummy_target, teacher_forcing_ratio=0.5)

    # 5. Print shapes to verify everything is working
    print("--- Seq2Seq Model Test ---")
    print(f"Device: {device}")
    print(f"Source input shape:  {dummy_source.shape}")
    print(f"Target (truth) shape: {dummy_target.shape}")
    print(f"Model output shape:  {predictions.shape}")
    print("\nModel is ready for training!")