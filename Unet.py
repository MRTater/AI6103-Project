import math

import torch
from torch import nn
from torch.nn import MultiheadAttention


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, activation, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
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
        # self.relu  = nn.ReLU()
        # Add self-attention layer
        self.self_attention = MultiheadAttention(embed_dim=out_ch, num_heads=1)
        # Skip connection
        self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch))

    def forward(
        self,
        x,
        t,
    ):
        # First Conv
        # h = self.bnorm1(self.relu(self.conv1(x)))
        h = self.bnorm1(self.activation(self.conv1(x)))
        # Time embedding
        # time_emb = self.relu(self.time_mlp(t))
        time_emb = self.activation(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        # Apply self-attention
        h = h.permute(
            0, 2, 3, 1
        )  # Change shape to (batch_size, sequence_length, channels)
        h = h.flatten(1, 2)  # Flatten the spatial dimensions
        h, _ = self.self_attention(h, h, h)
        h = h.view(
            x.shape[0], self.conv2.out_channels, x.shape[2], x.shape[3]
        )  # Reshape back to (batch_size, channels, height, width)
        # Second Conv
        # h = self.bnorm2(self.relu(self.conv2(h)))
        h = self.bnorm2(self.activation(self.conv2(h)))
        # Add residual connection
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

    def __init__(self, activation):
        super().__init__()
        image_channels = 3
        # down_channels = (64, 128, 256, 512, 1024)
        # up_channels = (1024, 512, 256, 128, 64)
        down_channels = (64, 128, 256, 512, 1024, 2048)  # deeper U-net
        up_channels = (2048, 1024, 512, 256, 128, 64)
        out_dim = 1
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            # nn.ReLU()
            activation,
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList(
            [
                Block(down_channels[i], down_channels[i + 1], time_emb_dim, activation)
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
