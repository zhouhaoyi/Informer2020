import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", DCSP=False):
        super().__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=d_model // 2, out_channels=d_model // 2, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model // 2) if DCSP else nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.csp=DCSP

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        if self.csp:
            split_x = torch.split(x, x.shape[2] // 2, dim = 2)
            csp_x = split_x[1].clone()
            norm_x = split_x[0].clone()
            norm_x = self.conv3(norm_x.permute(0, 2, 1))
            norm_x = norm_x.transpose(1, 2)
            new_x = self.dropout(self.self_attention(
                csp_x, csp_x, csp_x,
                attn_mask = x_mask
            )[0])
            csp_x = csp_x + self.dropout(new_x)
            csp_x = self.norm1(csp_x)
            x = torch.cat((csp_x, norm_x), 2)
        else:
            x = x + self.dropout(self.self_attention(
                x, x, x,
                attn_mask=x_mask
            )[0])
            x = self.norm1(x)


        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
