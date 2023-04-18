import math

import torch
from torch import nn
from torch.nn import MultiheadAttention


class SelfAttentionLayer(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.self_attention = MultiheadAttention(embed_dim=out_ch, num_heads=4)

    def forward(self, h):
        # Apply self-attention
        h = h.permute(0, 2, 3, 1)  # Change shape to (batch_size, sequence_length, channels)
        h = h.flatten(1, 2)  # Flatten the spatial dimensions
        h, _ = self.self_attention(h, h, h)
        return h


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, activation, use_self_attention, use_skip_connection, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.use_self_attention = use_self_attention
        self.use_skip_connection = use_skip_connection
        self.activation = activation

        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)

        # Add self-attention layer
        if self.use_self_attention:
            print("Self-attention Invoked")
            self.self_attention_layer = SelfAttentionLayer(out_ch)
        
        # Skip connection
        if self.use_skip_connection:
            # Modify the skip connection to handle the doubled number of input channels when up is True
            skip_in_channels = 2 * in_ch if up else in_ch
            self.skip = nn.Sequential(nn.Conv2d(skip_in_channels, out_ch, 1), nn.BatchNorm2d(out_ch))

    def forward(self, x, t,):
        # First Conv
        h = self.bnorm1(self.activation(self.conv1(x)))
        # Time embedding
        time_emb = self.activation(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        # Apply self-attention if flagged
        if self.use_self_attention:
            h = self.self_attention_layer(h)
            h = h.view(x.shape[0], self.conv2.out_channels, x.shape[2], x.shape[3])
        # Second Conv
        h = self.bnorm2(self.activation(self.conv2(h)))
        # Add residual connection
        if self.use_skip_connection:
            skip_x = self.skip(x)
            h = h + skip_x
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self, activation_input, use_self_attention, use_skip_connection):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1
        time_emb_dim = 32
        
        # Activation fuction
        self.activation_input = activation_input
        if self.activation_input == "relu":
            activation = nn.ReLU()
        elif self.activation_input == "silu":
            activation = nn.SiLU()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            activation,
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList(
            [
                Block(down_channels[i], down_channels[i + 1], time_emb_dim, activation, use_self_attention, use_skip_connection)
                for i in range(len(down_channels) - 1)
            ]
        )
        # Upsample
        self.ups = nn.ModuleList(
            [
                Block(
                    up_channels[i],
                    up_channels[i + 1],
                    time_emb_dim,
                    activation,
                    use_self_attention,
                    use_skip_connection,
                    up=True,
                )
                for i in range(len(up_channels) - 1)
            ]
        )

        self.output = nn.Conv2d(up_channels[-1], 3, out_dim)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)
