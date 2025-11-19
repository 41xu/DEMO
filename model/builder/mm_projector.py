# build multimodal_project layer
import re
import torch
import torch.nn as nn


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
    

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        # x: [B,C,L+extra]
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    """
    TODO: bachnorm part: current save script didn't save the state_dict of batchnorm. need to be fixed.
    therefore when load models, i set strict=False, and the model will skip batchnorm part.
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, dilation=dilation, padding=padding)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size, stride=stride, dilation=dilation, padding=padding)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.dropout = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channel, out_channel, 1) if in_channel != out_channel else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        # 残差连接
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(out + res)
        return out
    

class TemporalConvNet(nn.Module):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2):
        # num_channels: [263, 512, 512]
        super().__init__()
        layers = []
        num_levels = len(num_channels) - 1
        for i in range(num_levels):
            in_channels = num_channels[i]
            out_channels = num_channels[i+1]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, L]
        return self.network(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=2048):
        super().__init__()
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # 1,L,D
    
    def forward(self, x):
        # X: [B, L, D]
        return x + self.pe[:, :x.size(1)]

class MotionTransformer(nn.Module):
    def __init__(self, motion_dim=263, hidden_dim=1024, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(motion_dim, hidden_dim)
        self.temporal_pos = nn.Parameter(torch.randn(1, 1500, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(hidden_dim)
        # self.pos_emb = PositionalEncoding(hidden_dim, max_length=1500) # here max_length should be the max_lenght of motion, which is max 70s*20fps 

    def forward(self, motion):
        # 1, L, 263 
        B, L, D = motion.shape
        x = self.input_proj(motion) # B,L,hidden_dim
        x = x + self.temporal_pos[:, :L, :] # B,L,hidden_dim
        x = self.encoder(x.transpose(0,1)) # B,L,hidden_dim
        x = self.ln(x) # L,B,hidden_dim
        x = x.transpose(0,1) # B,L,hidden_dim
        return x


class PatchTransformer(nn.Module):
    def __init__(self, joint_dim, W=16, hidden_dim=1024, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.W = W
        self.joint_dim = joint_dim # 24*3 or 22*3 or 263

        self.input_proj = nn.Linear(self.joint_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pos_emb = PositionalEncoding(hidden_dim, max_length=W)

    def forward(self, patch_flatten):
        # B,L,joint_dim*W
        B, L, C = patch_flatten.shape
        x = patch_flatten.view(B, L, self.W, self.joint_dim)
        x = x.view(B * L, self.W, self.joint_dim)
        
        x = self.input_proj(x) # B*L,W,D
        x = self.pos_emb(x) # + temporal encoding within patch window_size
        x = self.encoder(x) # B*L,W,D
        x = self.ln(x)

        x = x.transpose(1, 2) # B*L,D,W
        x = self.pool(x).squeeze(-1) # B*L,D
        x = x.view(B, L, -1) # B,L,D

        return x

def build_vision_projector(config):
    projector_type = getattr(config, "mm_projector_type", "linear")
    pretrain_projector = getattr(config, "pretrain_mm", None)
    
    if projector_type == "linear":
        projector = nn.Linear(config.mm_hidden_size, config.hidden_size) # 1024,5120
        if pretrain_projector is not None:
            print("-----------------")
            mm_pretrain = torch.load(pretrain_projector)
            mm_pretrain = {(k[11:] if k.startswith('base_model.') else k): v for k, v in mm_pretrain.items()}
            if any(k.startswith('model.model.') for k in mm_pretrain):
                mm_pretrain = {(k[6:] if k.startswith('model.') else k): v for k, v in mm_pretrain.items()}
            projector.load_state_dict(mm_pretrain, strict=False)
            print(f"Load motion_encoder_projector from {pretrain_projector} success!!")
        else:
            print("no pretrain projector")
        return projector
    
    # TODO: this part is not implemented and used in current M2T
    # but i just copy this from llava
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_vision_tower(config):
    vision_tower = getattr(config, 'mm_vision_tower', getattr(config, 'vision_tower', None))
    pretrain_vision_tower = getattr(config, 'pretrain_mm', None)
    if vision_tower == 'mlp':
        vision_tower_module = nn.Sequential(*[
            nn.Linear(config.motion_dim, config.mlp_hidden),
            nn.ReLU(),
            nn.Linear(config.mlp_hidden, config.mm_hidden_size),
            nn.Dropout(0.0),
        ])
    elif vision_tower == 'tcn':
        num_channels = [config.motion_dim, config.mlp_hidden//2, config.mlp_hidden, config.mm_hidden_size]
        vision_tower_module = TemporalConvNet(num_channels, kernel_size=3, dropout=0.2)
    elif vision_tower == "attn":
        vision_tower_module = PatchTransformer(
            config.motion_dim,
        )
    elif vision_tower == "mattn":
        vision_tower_module = MotionTransformer()
    elif vision_tower == 'vqvae':
        vision_tower_module = None

    if pretrain_vision_tower is not None:
        print("-----------------")
        mm_pretrain = torch.load(pretrain_vision_tower)
        mm_pretrain = {(k[11:] if k.startswith('base_model.') else k): v for k, v in mm_pretrain.items()}
        if any(k.startswith('model.model.') for k in mm_pretrain):
            mm_pretrain = {(k[6:] if k.startswith('model.') else k): v for k, v in mm_pretrain.items()}
        vision_tower_module.load_state_dict(mm_pretrain, strict=False)
        print(f"Load motion_encoder from {pretrain_vision_tower} success!!")
    else:
        print("no pretrain vision tower")
    
    return vision_tower_module
        