import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import FullMask, LengthMask, TriangularCausalMask

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, memory, x_mask=None, x_length_mask=None,
                memory_mask=None, memory_length_mask=None):
        # Normalize the masks
        B = x.shape[0]
        L = x.shape[1]
        L_prime = memory.shape[1]
        x_mask = x_mask or TriangularCausalMask(L, device=x.device)
        x_length_mask = x_length_mask  or \
            LengthMask(x.new_full((B,), L, dtype=torch.int64))
        memory_mask = memory_mask or FullMask(L, L_prime, device=x.device)
        memory_length_mask = memory_length_mask or \
            LengthMask(x.new_full((B,), L_prime, dtype=torch.int64))

        # First apply the self attention and add it to the input
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            query_lengths=x_length_mask,
            key_lengths=x_length_mask
        ))
        x = self.norm1(x)

        # Secondly apply the cross attention and add it to the previous output
        x = x + self.dropout(self.cross_attention(
            x, memory, memory,
            attn_mask=memory_mask,
            query_lengths=x_length_mask,
            key_lengths=memory_length_mask
        ))

        # Finally run the fully connected part of the layer
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, memory, x_mask=None, x_length_mask=None,
                memory_mask=None, memory_length_mask=None):
        # Apply all the transformer decoders
        for layer in self.layers:
            x = layer(x, memory, x_mask=x_mask, x_length_mask=x_length_mask,
                      memory_mask=memory_mask,
                      memory_length_mask=memory_length_mask)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x