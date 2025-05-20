import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.utils.checkpoint as checkpoint
from timm.models.vision_transformer import DropPath, Mlp, trunc_normal_
from timm.models.layers import to_2tuple


class Attention(nn.Module):
    def __init__(self, input_size, dim, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., rpe=True):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * input_size - 1) * (2 * input_size - 1), num_heads)
        ) if rpe else None
        # if rpe:
        #     trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpe_index=None, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if rpe_index is not None:
            S = int(math.sqrt(rpe_index.size(-1)))
            relative_position_bias = self.relative_position_bias_table[rpe_index].view(-1, S, S, self.num_heads)
            relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()
            assert N == S
            attn = attn + relative_position_bias
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.float().clamp(min=torch.finfo(torch.float32).min, max=torch.finfo(torch.float32).max)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockWithRPE(nn.Module):
    def __init__(self, input_size, dim, num_heads=0., mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., rpe=True,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        with_attn = num_heads > 0.

        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = Attention(
            input_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, rpe=rpe,
        ) if with_attn else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, rpe_index=None, mask=None):
        if self.attn is not None:
            x = x + self.drop_path(self.attn(self.norm1(x), rpe_index, mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        x = self.proj(x).view(
            B, -1,
            patches_resolution[0], self.inner_patches,
            patches_resolution[1], self.inner_patches,
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x):
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class HiViT(nn.Module):
    def __init__(self, search_size=256, template_size=128, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=512, depths=[4, 4, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, ape=True, rpe=True, patch_norm=True, use_checkpoint=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.ape = ape
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.num_main_blocks = depths[-1]
        img_size = search_size
        self.num_patches_search = (search_size // patch_size) * (search_size // patch_size)
        self.num_patches_template = (template_size // patch_size) * (template_size // patch_size)

        embed_dim = embed_dim // 2 ** (self.num_layers - 1)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        Hp, Wp = self.patch_embed.patches_resolution
        assert Hp == Wp

        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches_search + self.num_patches_template, self.num_features)
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)
        if rpe:
            coords_h = torch.arange(Hp)
            coords_w = torch.arange(Wp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += Hp - 1
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = iter(x.item() for x in
                   torch.linspace(0, drop_path_rate, sum(depths) + sum(depths[:-1])))  # stochastic depth decay rule

        # build blocks
        self.blocks = nn.ModuleList()
        for stage_depth in depths:
            is_main_stage = embed_dim == self.num_features
            nhead = num_heads if is_main_stage else 0
            ratio = mlp_ratio if is_main_stage else stem_mlp_ratio
            # every block not in main stage include two mlp blocks
            stage_depth = stage_depth if is_main_stage else stage_depth * 2
            for i in range(stage_depth):
                self.blocks.append(
                    BlockWithRPE(
                        Hp, embed_dim, nhead, ratio, qkv_bias, qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                        rpe=rpe, norm_layer=norm_layer,
                    )
                )
            if not is_main_stage:
                self.blocks.append(
                    PatchMerge(embed_dim, norm_layer)
                )
                embed_dim *= 2

        self.norm = norm_layer(self.num_features)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, ids_keep=None, mask=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        if ids_keep is not None:
            x = torch.gather(
                x, dim=1, index=ids_keep[:, :, None, None, None].expand(-1, -1, *x.shape[2:])
            )

        for blk in self.blocks[:-self.num_main_blocks]:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        x = x[..., 0, 0, :]
        if self.ape:
            pos_embed = self.absolute_pos_embed
            if ids_keep is not None:
                pos_embed = torch.gather(
                    pos_embed.expand(B, -1, -1),
                    dim=1,
                    index=ids_keep[:, :, None].expand(-1, -1, pos_embed.shape[2]),
                )
            x += pos_embed
        x = self.pos_drop(x)

        rpe_index = None
        if self.rpe:
            if ids_keep is not None:
                B, L = ids_keep.shape
                rpe_index = self.relative_position_index
                rpe_index = torch.gather(
                    rpe_index[ids_keep, :], dim=-1, index=ids_keep[:, None, :].expand(-1, L, -1)
                ).reshape(B, -1)
            else:
                rpe_index = self.relative_position_index.view(-1)

        for blk in self.blocks[-self.num_main_blocks:]:
            x = checkpoint.checkpoint(blk, x, rpe_index, mask) if self.use_checkpoint else blk(x, rpe_index, mask)
        return x

    def forward_features_cross(self, z, x):
        B = x.shape[0]
        if isinstance(z, list) and len(z) == 2:
            x = self.patch_embed(x)
            z0 = self.patch_embed(z[0])
            z1 = self.patch_embed(z[1])
            zx = torch.cat([z0, z1, x], dim=1)
        elif isinstance(z, list) and len(z) == 1:
            x = self.patch_embed(x)
            z = self.patch_embed(z[0])
            zx = torch.cat([z, x], dim=1)
        else:
            x = self.patch_embed(x)
            z = self.patch_embed(z)
            zx = torch.cat([z, x], dim=1)

        for blk in self.blocks[:-self.num_main_blocks]:
            zx = blk(zx)
        zx = zx[..., 0, 0, :]
        if self.ape:
            pos_embed = self.absolute_pos_embed
            pos_embed_interpolate = F.interpolate(pos_embed.unsqueeze(0), size=(zx.shape[1], zx.shape[2]),
                                                  mode='bilinear')
            pos_embed_interpolate = pos_embed_interpolate.squeeze(0)
            zx += pos_embed_interpolate
        zx = self.pos_drop(zx)
        rpe_index = None
        if self.rpe:
            rpe_index = self.relative_position_index.view(-1)

        for blk in self.blocks[-self.num_main_blocks:]:
            zx = blk(zx, rpe_index, None)
        z = zx[:, :self.num_patches_template, :]
        x = zx[:, -self.num_patches_search:, :]
        return z, x

    def forward_ori(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.fc_norm(x)
        x = self.head(x)
        return x

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None, return_last_attn=None):
        z, x = self.forward_features_cross(z, x)
        x = self.norm(x)
        aux_dict = {}
        return x, aux_dict

    def finetune_track(self, cfg, patch_start_index):
        return None


def hivit_base(pth=None, **kwargs):
    model = HiViT(
        embed_dim=512, depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4.,
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)  # [4, 4, 20]

    if pth is not None:
        if len(pth) > 1:
            load_pretrained(model, pth)
    return model


def hivit_small(pth=None, **kwargs):
    model = HiViT(
        embed_dim=512, depths=[2, 2, 10], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4.,
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)  # [4, 4, 20]

    if pth is not None:
        if len(pth) > 1:
            load_pretrained(model, pth)
    return model


def hivit_light(pth=None, **kwargs):
    model = HiViT(
        embed_dim=512, depths=[2, 2, 6], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4.,
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)  # [4, 4, 20]

    if pth is not None:
        if len(pth) > 1:
            load_pretrained(model, pth)
    return model



def load_pretrained(model, pth='default', strict=False):
    print("Pretrained model: " + str(pth))

    state_dict = torch.load(pth, map_location='cpu')

    # adjust position encoding
    pe = state_dict['absolute_pos_embed']
    b_pe, hw_pe, c_pe = pe.shape
    side_pe = int(math.sqrt(hw_pe))
    side_num_patches_search = int(math.sqrt(model.num_patches_search))
    side_num_patches_template = int(math.sqrt(model.num_patches_template))
    pe_2D = pe.reshape([b_pe, side_pe, side_pe, c_pe]).permute([0, 3, 1, 2])  # b,c,h,w
    if side_pe != side_num_patches_search:
        pe_s_2D = nn.functional.interpolate(pe_2D, [side_num_patches_search, side_num_patches_search],
                                            align_corners=True, mode='bicubic')
        pe_s = torch.flatten(pe_s_2D.permute([0, 2, 3, 1]), 1, 2)
    else:
        pe_s = pe
    if side_pe != side_num_patches_template:
        pe_t_2D = nn.functional.interpolate(pe_2D, [side_num_patches_template, side_num_patches_template],
                                            align_corners=True, mode='bicubic')
        pe_t = torch.flatten(pe_t_2D.permute([0, 2, 3, 1]), 1, 2)
    else:
        pe_t = pe
    pe_zx = torch.cat((pe_t, pe_s), dim=1)
    state_dict['absolute_pos_embed'] = pe_zx
    # del state_dict['cls_token']
    norm_weight = state_dict['norm.weight']
    norm_bias = state_dict['norm.bias']
    a = model.norm.normalized_shape[0]
    state_dict['norm.weight'] = norm_weight[:a]
    state_dict['norm.bias'] = norm_bias[:a]

    miss_k, unexpect_k = model.load_state_dict(state_dict, strict=strict)
    print('missing_keys: ' + str(miss_k))
    print('unexpected_keys: ' + str(unexpect_k))


if __name__ == '__main__':
    print('Fixed inference speed evaluation and model scailing evaluation')
    import torch.backends.cudnn
    import torch.distributed as dist
    import random
    import numpy as np

    seed = 1001
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('set torch.backends.cudnn.deterministic/torch.backends.cudnn.benchmark/random.seed')

    pth = '/home/xyren/data/fei/save_pth/mae_hivit_base_1600ep.pth'

    """build model"""
    s_model =hivit_base() # hivit_base(pth)
    print('build model done')

    """load checkpoint"""
    # b3: [3, 4, 18, 3],
    # param = torch.load(pth)
    # missing_keys, unexpected_keys = s_model.load_statue_dict(param, strict=False)
    print('No state dict of checkpoint is loaded ')

    """analyze model construction """
    n_parameters1 = sum(p.numel() for n, p in s_model.named_parameters() if 'block1' in n)
    n_parameters2 = sum(p.numel() for n, p in s_model.named_parameters() if 'block2' in n)
    n_parameters3 = sum(p.numel() for n, p in s_model.named_parameters() if 'block3' in n)
    n_parameters4 = sum(p.numel() for n, p in s_model.named_parameters() if 'block' in n)
    n_parameters = sum(p.numel() for n, p in s_model.named_parameters())
    print('total params is :' + '%.2f' % (n_parameters / 1e6))
    print('stage 1 params is :' + '%.2f' % (n_parameters1 / 1e6) + ', percentage is :' + '%.4f' % (
            n_parameters1 / n_parameters))
    print('stage 2 params is :' + '%.2f' % (n_parameters2 / 1e6) + ', percentage is :' + '%.4f' % (
            n_parameters2 / n_parameters))
    print('stage 3 params is :' + '%.2f' % (n_parameters3 / 1e6) + ', percentage is :' + '%.4f' % (
            n_parameters3 / n_parameters))
    print('construction unit params is :' + '%.2f' % (n_parameters4 / 1e6) + ', percentage is :' + '%.4f' % (
            n_parameters4 / n_parameters))

    """test settings"""
    fps_add = 0
    num_video = 5
    Lz = 256  # 112 #128
    Lx = 256  # 224 #256
    num_frame = 100
    inputz_test_fixed = torch.randn([1, 3, Lz, Lz]).cuda()
    inputx_test_fixed = torch.randn([1, 3, Lx, Lx]).cuda()
    inputz_test = torch.randn([num_video, 1, 3, Lz, Lz]).cuda()
    inputx_test = torch.randn([num_video, num_frame, 1, 3, Lx, Lx]).cuda()
    print('length of z is ' + str(Lz))
    print('length of x is ' + str(Lx))
    print('number of video is ' + str(num_video))
    print('number of frame in each video is ' + str(num_frame))
    s_model.eval().cuda()
    print('set model to eval mode and put it into cuda')

    """evaluation for model parameter and flops"""
    from thop import profile

    flops_tools, params = profile(s_model, inputs=([inputz_test_fixed, inputx_test_fixed]), custom_ops=None,
                                  verbose=False)
    print('flops is :' + '%.2f' % (flops_tools / 1e9))
    print('params is :' + '%.2f' % (params / 1e6))

    """inference speed"""
    import cv2
    import time

    print('torch.no_grad')
    with torch.no_grad():
        for video_index in range(num_video):
            start = time.time()
            # tic = cv2.getTickCount()
            for frame_index in range(num_frame):
                ouput = s_model(inputz_test[0,], inputx_test[0, frame_index,])  # inputz_test[video_index, ])
            # toc = cv2.getTickCount()

            torch.cuda.synchronize()
            end = time.time()
            avg_lat = (end - start) / num_frame
            fps = 1. / avg_lat
            print('For Video ' + str(video_index) + ", FPS using time tool: %.2f fps" % (fps))
            # cpu_frq = cv2.getTickFrequency()
            # time_duration = (toc - tic) / cpu_frq
            # fps = num_frame / time_duration
            # print('FPS using cv2 tool: ' + '%.2f' % (fps))
            fps_add = fps + fps_add

    print('fps average is : ' + '%.2f' % (fps_add / num_video))

    a = 1
