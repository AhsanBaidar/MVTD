import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = x.permute(1,0,2)
        B, N, C = x.shape
        x = x.transpose(1,2).view(B,C,int(N**0.5),int(N**0.5)).contiguous()
        x = self.dwconv(x).flatten(2).transpose(1, 2)#B,N,C
        x = x.permute(1,0,2)
        return x
class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Extractor(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1,drop_path=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.query_norm = norm_layer(d_model)
        self.feat_norm = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        #convffn
        self.ffn = ConvFFN(in_features=d_model, hidden_features=int(d_model * 0.25), drop=0.)
        self.ffn_norm = norm_layer(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, feat):

        def _inner_forward(query, feat):
            # query:l,b,d;feat:l,b,d
            attn = self.attn(self.query_norm(query),
                             self.feat_norm(feat), self.feat_norm(feat))[0]
            query = query + attn

            query = query + self.drop_path(self.ffn(self.ffn_norm(query)))
            return query

        query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    def __init__(self, d_model, n_heads=8,norm_layer=partial(nn.LayerNorm, eps=1e-6),  dropout=0.1,
                 init_values=0.):
        super().__init__()
        self.query_norm = norm_layer(d_model)
        self.feat_norm = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,dropout=dropout)
        self.gamma = nn.Parameter(init_values * torch.ones((d_model)), requires_grad=True)

    def forward(self, query,feat):
            #query:l,b,d;feat:l,b,d
        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query),
                             self.feat_norm(feat),self.feat_norm(feat))[0]
            return query + self.gamma * attn
        query = _inner_forward(query, feat)
        return query


class InteractionBlock(nn.Module):
    def __init__(self, d_model, extra_extractor, grad_ckpt):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        self.injector = Injector(d_model=d_model)
        self.extractor = Extractor(d_model=d_model)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(d_model=d_model)
                for _ in range(2)])
        else:
            self.extra_extractors = None

    def forward(self,x,xs,blocks):
        x = self.injector(x.permute(1,0,2),xs.permute(1,0,2)).permute(1,0,2)
        for idx,blk in enumerate(blocks):
            x = checkpoint.checkpoint(blk, x, None,use_reentrant=False) if self.grad_ckpt else blk(x,None)
        xs = checkpoint.checkpoint(self.extractor, xs.permute(1,0,2),x.permute(1,0,2),use_reentrant=False).permute(1,0,2) \
            if self.grad_ckpt else self.extractor(xs.permute(1, 0, 2), x.permute(1, 0, 2)).permute(1, 0, 2)  # b,n,c
        # xs = self.extractor(xs.permute(1,0,2),x.permute(1,0,2)).permute(1,0,2)#b,n,c
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                xs = checkpoint.checkpoint(extractor, xs.permute(1, 0, 2), x.permute(1, 0, 2), use_reentrant=False).permute(1, 0, 2) \
                    if self.grad_ckpt else extractor(xs.permute(1, 0, 2), x.permute(1, 0, 2)).permute(1, 0,2)  # b,n,c
                # xs = extractor(xs.permute(1,0,2),x.permute(1,0,2)).permute(1,0,2)
        return x,xs



class Mamba_Neck(nn.Module):
    def __init__(self, in_channel=512,d_model=512,d_inner=1024,bias=False,n_layers=4,dt_rank=32,d_state=16,d_conv=3,dt_min=0.001,
                 dt_max=0.1,dt_init='random',dt_scale=1.0,conv_bias=True,dt_init_floor=0.0001,grad_ckpt=False):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.bias = bias
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.dt_scale = dt_scale
        self.num_channels = self.d_model
        self.layers = nn.ModuleList(
            [ResidualBlock(dt_scale,d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor,grad_ckpt)
             for _ in range(n_layers)])
        self.interactions = nn.ModuleList([
            InteractionBlock(d_model=d_model,extra_extractor=(True if i == n_layers - 1 else False),grad_ckpt=grad_ckpt)
            for i in range(n_layers)
        ])
        # self.norm_f = RMSNorm(config.d_model)

    def forward(self, x,xs,h,blocks,interaction_indexes):
        #  x : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        #  y : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)
        for i,index in enumerate(interaction_indexes):
            xs, h[i] = self.layers[i](xs, h[i])
            x,xs = self.interactions[i](x,xs,blocks[index[0]:index[1]])

        return x, xs, h


class ResidualBlock(nn.Module):
    def __init__(self,dt_scale, d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor,grad_ckpt):
        super().__init__()

        self.grad_ckpt = grad_ckpt
        self.mixer = MambaBlock(dt_scale,d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor)
        self.norm = RMSNorm(d_model)

    def forward(self, x, h):
        #  x : (B, L, D)
        # h : (B, L, ED, N)
        #  output : (B,L, D)

        x = self.norm(x)
        output, h = checkpoint.checkpoint(self.mixer,x,h,use_reentrant=False) if self.grad_ckpt else self.mixer(x, h)
        output = output + x
        return output, h



class MambaBlock(nn.Module):
    def __init__(self,dt_scale, d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor):
        super().__init__()
        #  projects block input from D to 2*ED (two branches)
        self.dt_scale = dt_scale
        self.d_model = d_model
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)

        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner,
                                kernel_size=d_conv, bias=conv_bias,
                                groups=self.d_inner,
                                padding=(d_conv - 1)//2)

        #  projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)

        #  projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        #  dt initialization
        #  dt weights
        dt_init_std = self.dt_rank ** -0.5 * self.dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))  #  inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        #  todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(self.d_inner))

        #  projects block output from ED back to D
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
    def forward(self, x, h):
        #  x : (B,L, D)
        # h : (B,L, ED, N)

        #  y : (B, L, D)


        xz = self.in_proj(x)  # (B, L,2*ED)
        x, z = xz.chunk(2, dim=-1)  #  (B,L, ED), (B,L, ED)
        x_cache = x.permute(0,2,1)#(B, ED,L)

        #  x branch
        x = self.conv1d( x_cache).permute(0,2,1) #  (B,L , ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)
        #y->B,L,ED;h->B,L,ED,N

        #  z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  #  (B, L, D)

        return output, h

    def ssm_step(self, x, h):
        #  x : (B, L, ED)
        #  h : (B, L, ED, N)

        A = -torch.exp(
            self.A_log.float())  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()
        #  TODO remove .float()

        deltaBC = self.x_proj(x)  #  (B, L, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state],
                                  dim=-1)  #  (B, L,dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B,L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B,L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L,ED, N)

        if h is None:
            h = torch.zeros(x.size(0), x.size(1), self.d_inner, self.d_state, device=deltaA.device)  #  (B, L, ED, N)

        h = deltaA * h + BX  #  (B, L, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x#B,L,ED

        #  todo : pq h.squeeze(1) ??
        return y, h


#  taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


def build_neck(cfg,encoder):
    in_channel = encoder.num_channels
    d_model = cfg.MODEL.NECK.D_MODEL
    n_layers = cfg.MODEL.NECK.N_LAYERS
    d_state = cfg.MODEL.NECK.D_STATE
    grad_ckpt = cfg.MODEL.ENCODER.GRAD_CKPT
    neck = Mamba_Neck(in_channel=in_channel,d_model=d_model,d_inner=2*d_model,n_layers=n_layers,dt_rank=d_model//16,d_state=d_state,grad_ckpt=grad_ckpt)
    return neck