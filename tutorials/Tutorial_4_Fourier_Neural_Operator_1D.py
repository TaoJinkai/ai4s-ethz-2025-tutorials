import torch
import torch.nn as nn

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        """
        A 1D Fourier layer that performs:
        1. FFT -> Transform to frequency domain
        2. Linear transform on low-frequency modes -> Weighting low-frequency modes
        3. iFFT -> Transform back to the physical space
        """
        self.in_channels = in_channels      # C_in
        self.out_channels = out_channels    # C_out
        self.modes1 = modes1                # Number of frequency modes to keep

        # Initialize the complex weights (in_channels, out_channels, modes1)
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        # Complex multiplication (B, C_in, K) * (C_in, C_out, K) -> (B, C_out, K)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        """
        x: Input tensor of shape (B, C_in, S) where:
        - B: batch size
        - C_in: input channels
        - S: number of spatial points (grid size)
        
        Returns: Tensor of shape (B, C_out, S)
        """
        B, C_in, S = x.shape  # Extract dimensions from input

        # Step 1: Apply FFT along the spatial dimension (last dimension)
        x_ft = torch.fft.rfft(x, dim=-1)  # Fourier Transform along last dimension (S)
        
        # Step 2: Create an empty tensor for the output in frequency space
        S_freq = x_ft.size(-1)  # This is S // 2 + 1 due to the nature of rfft
        out_ft = torch.zeros(B, self.out_channels, S_freq, device=x.device, dtype=torch.cfloat)

        # Step 3: Multiply the relevant Fourier modes by the learned weights
        x_ft_low = x_ft[:, :, :self.modes1]  # Select the first 'modes1' Fourier modes
        out_ft_low = self.compl_mul1d(x_ft_low, self.weights1)

        # Step 4: Insert the transformed low-frequency modes into the output tensor
        out_ft[:, :, :self.modes1] = out_ft_low

        # Step 5: Apply inverse FFT to transform back to the physical space
        x_out = torch.fft.irfft(out_ft, n=S, dim=-1)  # Inverse FFT to return to physical space

        return x_out


class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains:
        1. Lift the input to the desired channel dimension by self.fc0.
        2. 3 layers of the Fourier layers (u' = (W + K)(u)) where W is weights and K is conv.
        3. Project from the channel space to the output space by self.fc1 and self.fc2.
        """
        self.modes1 = modes  # Number of Fourier modes to use
        self.width = width   # Number of channels (hidden size)
        self.padding = 1     # Padding for non-periodic boundaries

        # 1. Input projection: (u0(x), x) -> higher dimension (width)
        self.linear_p = nn.Linear(2, self.width)

        # 2. Fourier convolution layers
        self.spect1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect3 = SpectralConv1d(self.width, self.width, self.modes1)

        # 3. Point-wise convolutions (1x1 conv)
        self.lin0 = nn.Conv1d(self.width, self.width, 1)
        self.lin1 = nn.Conv1d(self.width, self.width, 1)
        self.lin2 = nn.Conv1d(self.width, self.width, 1)

        # 4. Linear layers to project to the output space
        self.linear_q = nn.Linear(self.width, 32)
        self.output_layer = nn.Linear(32, 1)

        # Activation function
        self.activation = torch.nn.Tanh()

    def fourier_layer(self, x, spectral_layer, conv_layer):
        """
        x: (B, C, S)
        spectral_layer(x): (B, C, S)
        conv_layer(x):     (B, C, S)
        Output: (B, C, S)
        """
        x1 = spectral_layer(x)  # Apply the Fourier layer
        x2 = conv_layer(x)      # Apply the point-wise convolution
        x_out = x1 + x2         # Residual connection
        x_out = self.activation(x_out)
        return x_out

    def linear_layer(self, x, linear_transformation):
        """
        x: (B, S, C_in)
        linear_transformation: nn.Linear(C_in, C_out)
        Output: (B, S, C_out)
        """
        return self.activation(linear_transformation(x))

    def forward(self, x):
        """
        Forward pass of the network
        x: (B, S, 2)
        Returns: (B, S, 1)
        """
        # 1) Project input from (2) channels to (width) channels
        x = self.linear_p(x)  # (B, S, width)

        # 2) Permute to (B, width, S) for Fourier convolution
        x = x.permute(0, 2, 1)

        # 3) Apply the Fourier layers with residual connections
        x = self.fourier_layer(x, self.spect1, self.lin0)  # (B, width, S)
        x = self.fourier_layer(x, self.spect2, self.lin1)  # (B, width, S)
        x = self.fourier_layer(x, self.spect3, self.lin2)  # (B, width, S)

        # 4) Permute back to (B, S, width) for linear layers
        x = x.permute(0, 2, 1)

        # 5) Apply the final linear transformation to reduce to a single output channel
        x = self.linear_layer(x, self.linear_q)  # (B, S, 32)
        x = self.output_layer(x)  # (B, S, 1)

        return x


def demo():
    B = 2  # Batch size
    S = 8  # Spatial grid size (number of points)
    modes = 4  # Number of Fourier modes to use
    width = 16  # Number of channels

    # 1) Construct input (B, S, 2): [u0(x), x]
    u0 = torch.randn(B, S, 1)  # (B, S, 1) random initial condition
    x_coord = torch.linspace(-1, 1, S).unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1)  # (B, S, 1)

    # Concatenate to form input: (B, S, 2)
    x_input = torch.cat([u0, x_coord], dim=-1)

    print("=== Input x_input ===")
    print("x_input shape:", x_input.shape)  # (B, S, 2)

    # 2) Create the model
    model = FNO1d(modes=modes, width=width)

    # 3) Pass the input through the model
    output = model(x_input)

    print("=== Model Output ===")
    print("Output shape:", output.shape)  # (B, S, 1)

# Call the demo function
demo()
# === Input x_input ===
# x_input shape: torch.Size([2, 8, 2])
# === Model Output ===
# Output shape: torch.Size([2, 8, 1])
