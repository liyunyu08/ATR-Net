import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

class LayerNorm(nn.Module):  
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim,  1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# PreNorm for Transformer
class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# MLP for Transformer
class MLP(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0., layer_scale_init=-1):
        super().__init__()
        self.layer_scale_init = layer_scale_init

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        if self.layer_scale_init > 0:
            self.layer_scale = nn.Parameter(torch.ones(1, 1, dim) * self.layer_scale_init)
        else:
            self.layer_scale = None

    def forward(self, x):

        return self.net(x) + x if self.layer_scale is None else self.net(x) * self.layer_scale + x


# SelfAttention Block for Transformer with einops
class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0., temperature=1., layer_scale_init=-1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.temperature = temperature
        self.layer_scale_init = layer_scale_init

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        if self.layer_scale_init > 0:
            self.layer_scale = nn.Parameter(torch.ones(1, 1, dim) * self.layer_scale_init)
        else:
            self.layer_scale = None

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)/self.temperature

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out) + x if self.layer_scale is None else self.to_out(out).mul_(self.layer_scale) + x


# Transformer Block for Transformer
class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, mlp_expansion=4, dropout=0., temperature=1., layer_scale_init=-1):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = dim * mlp_expansion
        self.dropout = dropout
        self.temperature = temperature
        self.layer_scale_init = layer_scale_init

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout, temperature=temperature,
                                       layer_scale_init=layer_scale_init)),
                PreNorm(dim, MLP(dim, self.mlp_dim, dropout=dropout, layer_scale_init=layer_scale_init))
            ]))

    def forward(self, x):
        for attn, mlp in self.layers:
            x = attn(x)
            x = mlp(x)
        return x


class TaskDescriptorGenerator(nn.Module):

    def __init__(self, dim, depth, heads, num_descriptor, mlp_expansion=4, dropout=0., temperature=1.,
                 layer_scale_init=-1):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = dim * mlp_expansion
        self.dropout = dropout
        self.temperature = temperature
        self.layer_scale_init = layer_scale_init
        self.num_descriptor = num_descriptor

        self.task_descriptor = nn.Parameter(torch.randn(1, num_descriptor, dim))


        self.transformer = Transformer(dim, depth, heads, mlp_expansion, dropout, temperature, layer_scale_init)

    def forward(self, support_feature):
        n,c,h,w = support_feature.shape
        support_feature = rearrange(support_feature, 'n c h w -> 1 (n h w) c')
        x = torch.cat([self.task_descriptor, support_feature], dim=1)
        x = self.transformer(x)
        task_descriptor = x[:, :self.num_descriptor, :]


        return task_descriptor 


class TaskSpecificRegionSelector(nn.Module):

    def forward(self, feature, key_channels):
        "feature: (B, C, H, W)"
        "key_channels: (1, M, C)"
        b, c, h, w = feature.shape
        m = key_channels.shape[1]
        # B, M, C, H, W
        feature = torch.reshape(feature, (b, 1, c, h, w))
        key_channels = torch.reshape(key_channels, (1, m, c, 1, 1))

        # B N H W
        dot = feature * key_channels
        dot = torch.mean(dot, dim=2)
        dot = torch.sigmoid(dot)

        return dot

class MAP(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(MAP, self).__init__()


    def forward(self, task,feat):
        b, c, h, w = feat.shape
        m = task.shape[1]

        feature = torch.reshape(feat, (b, 1, c, h, w))
        task = torch.reshape(task, (b, m, 1, h, w))
        dot = feature*task+feature
        dot = torch.mean(dot, dim=1)
        return dot


# Task Enhanced Feature Generator for Step. 9-10
class RFM(nn.Module):

    def __init__(self, in_dim,num_token):
        super(RFM, self).__init__()
        self.in_dim = in_dim
        self.num_token = num_token

        self.selected_func = nn.Sequential(
            LayerNorm(self.in_dim),
            nn.Conv2d(self.in_dim, self.num_token, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.num_token, self.num_token, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.num_token, self.num_token, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.num_token, self.num_token, 3, 1, 1, bias=False),
            Rearrange('b n h w -> b n (h w)'),
            nn.Sigmoid() )

    def forward(self, feature, task_specific_region):

        task = self.selected_func(task_specific_region)
        task = task[:, :, :, None]
        feat = rearrange(feature, 'b c h w -> b (h w) c')
        feat = feat[:, None, :, :]
        dot = feat * task
        dot = torch.mean(dot, dim=-2)
        dot = dot.permute(0, 2, 1)

        return dot



# Integrat the above modules to the TE-Module(Task-Enhanced Module) for Step. 1-10
class TACF(nn.Module):

    def __init__(self,args, dim, descriptor_depth, heads, num_descriptor,num_token, mlp_expansion=4, dropout=0.,
                 temperature=1., layer_scale_init=-1):

        super().__init__()
        self.dim = dim
        self.descriptor_depth = descriptor_depth
        self.heads = heads
        self.num_descriptor = num_descriptor
        self.mlp_expansion = mlp_expansion
        self.dropout = dropout
        self.temperature = temperature
        self.layer_scale_init = layer_scale_init
        self.token = num_token
        self.args = args


        self.task_descriptor_generator = TaskDescriptorGenerator(dim, descriptor_depth,
                                                                 heads, num_descriptor,
                                                                 mlp_expansion, dropout,
                                                                 temperature, layer_scale_init)
        self.task_specific_region_selector = TaskSpecificRegionSelector()

        self.rfm = RFM(dim,self.token)
        self.map = MAP(num_descriptor, dim)



    def forward(self, support_feature,query_feature):
        task_descriptors = self.task_descriptor_generator(support_feature)

        task_specific_query_region = self.task_specific_region_selector(query_feature, task_descriptors)
        task_specific_support_region = self.task_specific_region_selector(support_feature, task_descriptors)

        task_att_qry = self.map(task_specific_query_region,query_feature)
        task_att_spt = self.map(task_specific_support_region,support_feature)

        task_specific_query_feature = self.rfm(task_att_qry,task_att_qry)
        task_specific_support_feature = self.rfm(task_att_spt,task_att_spt)

        return task_specific_support_feature, task_specific_query_feature


