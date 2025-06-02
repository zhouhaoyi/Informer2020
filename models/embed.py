import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * 
                    -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular'
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1))
        return x.transpose(1, 2)

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() *
                    -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed   = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed    = Embed(day_size, d_model)
        self.month_embed  = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        hour_x    = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x     = self.day_embed(x[:, :, 1])
        month_x   = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)

class TUPESelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=5000, dropout=0.1):
        super(TUPESelfAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.pos_emb_q = nn.Embedding(max_len, d_model)
        self.pos_emb_k = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos_emb_q.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb_k.weight, mean=0.0, std=0.02)

        self.max_len = max_len

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        H = self.n_heads
        d_k = self.d_k

        Q = self.W_q(x).view(B, T, H, d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, d_k).transpose(1, 2)

        content_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        device = x.device
        pos_ids = torch.arange(T, dtype=torch.long, device=device)
        pos_q_full = self.pos_emb_q(pos_ids)
        pos_k_full = self.pos_emb_k(pos_ids)
        pos_q = pos_q_full.view(T, H, d_k)
        pos_k = pos_k_full.view(T, H, d_k)
        pos_q_t = pos_q.permute(1, 0, 2)
        pos_k_t = pos_k.permute(1, 0, 2)
        pos_scores = torch.matmul(pos_q_t, pos_k_t.transpose(-2, -1))
        pos_scores = pos_scores.unsqueeze(0).expand(B, -1, -1, -1)

        total_scores = content_scores + pos_scores

        if attn_mask is not None:
            total_scores = total_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(total_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_out(attn_output)
        return out, attn_weights

class TUPETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super(TUPETransformerEncoderLayer, self).__init__()
        self.self_attn = TUPESelfAttention(d_model, n_heads, max_len=5000, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = F.relu

    def forward(self, src, src_mask=None):
        attn_out, attn_weights = self.self_attn(src, attn_mask=src_mask)
        src2 = self.dropout1(attn_out)
        src = src + src2
        src = self.norm1(src)

        ff = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + ff
        src = self.norm2(src)
        return src, attn_weights

class TUPETransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super(TUPETransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TUPETransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        out = x
        attn_weights_all = []
        for layer in self.layers:
            out, w = layer(out, src_mask=mask)
            attn_weights_all.append(w)
        return out, attn_weights_all

class BaselineModel(nn.Module):
    def __init__(self, c_in, d_model, n_heads, num_layers, dim_feedforward, dropout, embed_type, freq, c_out):
        super(BaselineModel, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.pos_embedding   = PositionalEmbedding(d_model=d_model)
        if embed_type != 'timeF':
            self.temp_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temp_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x, x_mark, src_mask=None):
        v = self.value_embedding(x)
        p = self.pos_embedding(x)
        t = self.temp_embedding(x_mark)
        emb = self.dropout(v + p + t)
        emb = emb.transpose(0, 1)
        out = self.encoder(emb)
        out = out.transpose(0, 1)
        y_hat = self.projection(out)
        return y_hat

class TUPEModel(nn.Module):
    def __init__(self, c_in, d_model, n_heads, num_layers, dim_feedforward, dropout, embed_type, freq, c_out):
        super(TUPEModel, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.pos_embedding   = PositionalEmbedding(d_model=d_model)
        if embed_type != 'timeF':
            self.temp_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temp_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(dropout)

        self.encoder = TUPETransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x, x_mark, src_mask=None):
        v = self.value_embedding(x)
        p = self.pos_embedding(x)
        t = self.temp_embedding(x_mark)
        emb = self.dropout(v + p + t)
        out, attn_weights_all = self.encoder(emb, mask=src_mask)
        y_hat = self.projection(out)
        return y_hat, attn_weights_all

class DummyTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, length=200, seq_len=24, c_in=1, c_out=1):
        super().__init__()
        self.length = length
        self.seq_len = seq_len
        self.c_in = c_in
        self.c_out = c_out

        self.data = torch.randn(length, seq_len, c_in)

        months   = torch.randint(0, 13, (length, seq_len, 1))
        days     = torch.randint(0, 32, (length, seq_len, 1))
        weekdays = torch.randint(0, 7,  (length, seq_len, 1))
        hours    = torch.randint(0, 24, (length, seq_len, 1))
        self.mark = torch.cat([months, days, weekdays, hours], dim=2).float()

        self.targets = torch.randn(length, seq_len, c_out)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            'x':      self.data[idx],
            'x_mark': self.mark[idx],
            'y':      self.targets[idx]
        }

def train_one_epoch(baseline_model, tupe_model, loader, device, optim_baseline, optim_tupe, criterion):
    baseline_model.train()
    tupe_model.train()

    mse_baseline_epoch = 0.0
    mse_tupe_epoch     = 0.0
    num_batches = 0

    for batch in loader:
        x      = batch['x'].to(device)
        x_mark = batch['x_mark'].to(device)
        y_true = batch['y'].to(device)

        optim_baseline.zero_grad()
        y_pred_baseline = baseline_model(x, x_mark)
        loss_baseline   = criterion(y_pred_baseline, y_true)
        loss_baseline.backward()
        optim_baseline.step()
        mse_baseline_epoch += loss_baseline.item()

        optim_tupe.zero_grad()
        y_pred_tupe, _ = tupe_model(x, x_mark)
        loss_tupe      = criterion(y_pred_tupe, y_true)
        loss_tupe.backward()
        optim_tupe.step()
        mse_tupe_epoch += loss_tupe.item()

        num_batches += 1

    avg_mse_baseline = mse_baseline_epoch / num_batches
    avg_mse_tupe     = mse_tupe_epoch / num_batches
    return avg_mse_baseline, avg_mse_tupe

if __name__ == "__main__":
    c_in       = 1
    d_model    = 64
    n_heads    = 4
    num_layers = 2
    dim_ff     = 128
    dropout    = 0.1
    embed_type = 'fixed'
    freq       = 'h'
    c_out      = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    baseline_model = BaselineModel(
        c_in=c_in,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=dropout,
        embed_type=embed_type,
        freq=freq,
        c_out=c_out
    ).to(device)

    tupe_model = TUPEModel(
        c_in=c_in,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=dropout,
        embed_type=embed_type,
        freq=freq,
        c_out=c_out
    ).to(device)

    dataset = DummyTimeSeriesDataset(length=200, seq_len=24, c_in=c_in, c_out=c_out)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    optim_baseline = optim.Adam(baseline_model.parameters(), lr=1e-4)
    optim_tupe     = optim.Adam(tupe_model.parameters(), lr=5e-4)

    criterion = nn.MSELoss()

    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        mse_base, mse_tupe = train_one_epoch(
            baseline_model,
            tupe_model,
            loader,
            device,
            optim_baseline,
            optim_tupe,
            criterion
        )
        print(f"Epoch {epoch}: Baseline MSE = {mse_base:.6f}, TUPE MSE = {mse_tupe:.6f}")