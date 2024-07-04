from functools import partial
import torch.nn as nn
import torch
from torch.nn import functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

    def drop_path(self, x, drop_prob=0., training=False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=torch.float32)
        random_tensor = torch.floor(random_tensor)
        output = x.divide(keep_prob) * random_tensor
        return output


class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            x = x.permute([0, 3, 1, 2])
        x = self.conv(x)
        x = x.permute([0, 2, 3, 1])  # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x


class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.scale


class SquaredReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return torch.square(self.relu(x))


class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.relu = nn.ReLU()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


def to_2tuple(x):
    if isinstance(x, (int, float)):
        x = [x, x]
    return x


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerNormGeneral(nn.Module):
    def __init__(self, affine_shape=None, normalized_dim=(-1,), scale=True,
                 bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        if isinstance(affine_shape, int):
            affine_shape = [affine_shape]
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


def resize_complex_weight(origin_weight, new_h, new_w):
    h, w, num_heads = origin_weight.shape[0:3]  # size, w, c, 2
    origin_weight = origin_weight.reshape((1, h, w, num_heads * 2)).permute([0, 3, 1, 2])
    new_weight = F.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    ).permute([0, 2, 3, 1]).reshape((new_h, new_w, num_heads, 2))
    return new_weight


class DynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=True,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.local_conv = nn.Sequential(
            nn.Conv2d(self.med_channels, self.med_channels, 3, 1, 1, groups=self.med_channels),
            nn.BatchNorm2d(self.med_channels),
            act1_layer()
        )
        self.complex_weights = nn.Parameter(torch.rand(self.size, self.filter_size, num_filters, 2))
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        B, H, W, _ = x.shape
        routeing = self.reweight(x.mean(dim=(1, 2))).reshape((B, self.num_filters, -1))
        routeing = F.softmax(routeing, dim=1)
        x = self.pwconv1(x)
        x = self.act1(x)

        local_x = x.permute((0, 3, 1, 2))
        local_x = self.local_conv(local_x)
        local_x = local_x.permute((0, 2, 3, 1))

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1], x.shape[2])
        else:
            complex_weights = self.complex_weights

        weight = torch.einsum('bfc,hwfl->bhwcl', routeing, complex_weights)

        weight = torch.view_as_complex(weight)

        if self.weight_resize:
            weight = weight.reshape((-1, x.shape[1], x.shape[2], self.med_channels))
        else:
            weight = weight.reshape((-1, self.size, self.filter_size, self.med_channels))
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = x + local_x
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    """

    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=SquaredReLU,
                 norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """

    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None,
                 size=14,
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop, size=size)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x


DOWNSAMPLE_LAYERS_FOUR_STAGES = [partial(Downsampling,
                                         kernel_size=7, stride=4, padding=2,
                                         post_norm=partial(LayerNormGeneral, bias=False,
                                                           eps=1e-6)
                                         )] + \
                                [partial(Downsampling,
                                         kernel_size=3, stride=2, padding=1,
                                         pre_norm=partial(LayerNormGeneral, bias=False,
                                                          eps=1e-6), pre_permute=True
                                         )] * 3


class MetaFormer(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[2, 2, 6, 2],
                 dims=[64, 128, 320, 512],
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 token_mixers=nn.Identity,
                 mlps=Mlp,
                 norm_layers=partial(LayerNormGeneral, eps=1e-6, bias=False),
                 drop_path_rate=0.,
                 head_dropout=0.0,
                 layer_scale_init_values=None,
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 output_norm=partial(nn.LayerNorm, eps=1e-6),
                 head_fn=nn.Linear,
                 input_size=(3, 224, 224),
                 **kwargs,
                 ):
        super().__init__()

        self.num_classes = num_classes

        if not isinstance(depths, (list, tuple)):
            depths = [depths]  # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i + 1]) for i in
             range(num_stage)]
        )

        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage

        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = nn.ModuleList()  # each stage consists of multiple metaformer blocks
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[MetaFormerBlock(dim=dims[i],
                                  token_mixer=token_mixers[i],
                                  mlp=mlps[i],
                                  norm_layer=norm_layers[i],
                                  drop_path=dp_rates[cur + j],
                                  layer_scale_init_value=layer_scale_init_values[i],
                                  res_scale_init_value=res_scale_init_values[i],
                                  size=(input_size[1] // (2 ** (i + 2)),
                                        input_size[2] // (2 ** (i + 2))),
                                  ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = output_norm(dims[-1])

        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)

    def forward_features(self, x):
        outs = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x.permute(0, 3, 1, 2))
        return outs  # (B, H, W, C) -> (B, C)

    def forward(self, x):
        outs = self.forward_features(x)
        # x = self.head(x)
        return outs

    def initialize(self):
        pass


def LightPCT_T(**kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[16, 32, 80, 128],
        token_mixers=DynamicFilter,
        head_fn=MlpHead,
        input_size=(3, 224, 224),
        **kwargs)
    model.load_state_dict(torch.load("FEDER/lib/LightPCT_T.pth"), strict=True)
    return model


def LightPCT_M(**kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[24, 48, 128, 160],
        token_mixers=DynamicFilter,
        head_fn=MlpHead,
        input_size=(3, 224, 224),
        **kwargs)
    model.load_state_dict(torch.load("FEDER/lib/LightPCT_M.pth"), strict=True)
    return model


def LightPCT_S(**kwargs):
    model = MetaFormer(
        depths=[3, 3, 18, 3],
        dims=[32, 64, 160, 256],
        token_mixers=DynamicFilter,
        head_fn=MlpHead,
        input_size=(3, 224, 224),
        **kwargs)
    return model


def LightPCT_L(**kwargs):
    model = MetaFormer(
        depths=[3, 3, 15, 3],
        dims=[48, 96, 192, 384],
        token_mixers=DynamicFilter,
        head_fn=MlpHead,
        input_size=(3, 224, 224),
        **kwargs)
    return model


if __name__ == '__main__':
    torch.set_printoptions(precision=8)
    net = LightPCT_T(class_num=1000, drop_path_rate=0.1)
    x = torch.ones(1, 3, 224, 224)
    outs = net(x)
    print([i.shape for i in outs])
    total_params = sum(p.numel() for p in net.parameters())
    print('total params : ', total_params)
    # x = torch.ones(1, 3, 224, 224)
    # wgt = torch.load(r'F:\worksinphd\CODFFT\backbone\LightPCT_T.pth')
    # net.load_state_dict(wgt)
    # net.eval()
    # print(net(x)[0][0:10])
    # paddle.summary(net, (1, 3, 224, 224))
    # # x = paddle.rand((1, 3, 224, 224))
    # # net = LightPCT_L(class_num=1000, drop_path_rate=0.1)
    # # net.eval()
    # paddle.flops(net, [1, 3, 224, 224])
    # # for _ in range(3):
    # #     net(x)
    # # times = time.time()
    # # for _ in range(10):
    # #     net(x)
    # # times_end = time.time()
    # # print(times_end - times)
