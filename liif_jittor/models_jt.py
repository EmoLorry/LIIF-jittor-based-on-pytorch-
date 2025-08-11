import copy
import math
from argparse import Namespace
import jittor as jt
from jittor import init
from jittor import nn
from jittor import nn as F  # 添加functional导入
from utils_jittor import make_coord  # 添加make_coord导入
jt.flags.use_cuda = 1
models = {}
#注册模型   model = models_jt.make(config['model'])
# model:
#   name: metasr
#   args:
#     encoder_spec:
#       name: edsr-baseline
#       args:
#         no_upsampling: true
def register(name):

    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

def make(model_spec, args=None, load_sd=False):
    if (args is not None):
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model
# ============================================================================
# EDSR Components (from edsr.py)
# ============================================================================

#快速卷积模块，自动计算pading,实现same效果 size(input)=size(output)
#当卷积核大小是奇数（如 3、5、7）时，这个填充策略可以让输出的高宽与输入一致（即 "same" 卷积效果）。
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

#均值偏移层（图像归一化 / 反归一化）
#这是固定的参数，只是用来图像归一化，输入形状 (B, 3, H, W)不变
class MeanShift(nn.Conv):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.404), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super().__init__(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)

        std = jt.array(rgb_std, dtype=jt.float32)
        eye = jt.init.eye(3).reshape(3, 3, 1, 1)
        self.weight.update(eye / std.reshape(3, 1, 1, 1))

        bias = (sign * rgb_range) * jt.array(rgb_mean, dtype=jt.float32) / std
        self.bias.update(bias)

        # 冻结参数，这是固定的参数
        self.weight.stop_grad()
        self.bias.stop_grad()

class ResBlock(nn.Module):
#n_feats：特征通道数，残差块输入输出通道相同。
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm(n_feats))
            if (i == 0):
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def execute(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


#上采样，图片放大
class Upsampler(nn.Sequential):

    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if ((scale & (scale - 1)) == 0):
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, (4 * n_feats), 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm(n_feats))
                if (act == 'relu'):
                    m.append(nn.ReLU())
                elif (act == 'prelu'):
                    m.append(nn.PReLU(num_parameters=n_feats))
        elif (scale == 3):
            m.append(conv(n_feats, (9 * n_feats), 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm(n_feats))
            if (act == 'relu'):
                m.append(nn.ReLU())
            elif (act == 'prelu'):
                m.append(nn.PReLU(num_parameters=n_feats))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)



url = {'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
       'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt', 
       'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt', 
       'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt', 
       'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
       'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'}


class EDSR(nn.Module):

    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU()
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if (url_name in url):
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        m_body = [ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale) for _ in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            m_tail = [Upsampler(conv, scale, n_feats, act=False), conv(n_feats, args.n_colors, kernel_size)]
            self.tail = nn.Sequential(*m_tail)

    def execute(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for (name, param) in state_dict.items():
            if (name in own_state):
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if (name.find('tail') == (- 1)):
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].shape, param.shape))
            elif strict:
                if (name.find('tail') == (- 1)):
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))

@register('edsr-baseline')
def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1, scale=2, no_upsampling=False, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.rgb_range = rgb_range
    args.n_colors = 3
    return EDSR(args)

    
@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def execute(self, x):
        shape = x.shape[:(- 1)]
        x = self.layers(x.view(((- 1), x.shape[(- 1)])))
        return x.view((*shape, (- 1)))

@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.encoder = make(encoder_spec)
        if (imnet_spec is not None):
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def execute(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = jt.contrib.concat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = jt.contrib.concat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)
                #取相对x和y坐标相乘计算面积，并加上1e-9防止除0
                area = jt.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = jt.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def execute(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)



##############################
#RDN
##############################


class RDB_Conv(nn.Module):

    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[nn.Conv(Cin, G, kSize, padding=((kSize - 1) // 2), stride=1), nn.ReLU()])

    def execute(self, x):
        out = self.conv(x)
        return jt.contrib.concat((x, out), dim=1)

class RDB(nn.Module):

    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers
        convs = []
        for c in range(C):
            convs.append(RDB_Conv((G0 + (c * G)), G))
        self.convs = nn.Sequential(*convs)
        self.LFF = nn.Conv((G0 + (C * G)), G0, 1, padding=0, stride=1)

    def execute(self, x):
        return (self.LFF(self.convs(x)) + x)

class RDN(nn.Module):

    def __init__(self, args):
        super(RDN, self).__init__()
        self.args = args
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        (self.D, C, G) = {'A': (20, 6, 32), 'B': (16, 8, 64)}[args.RDNconfig]
        self.SFENet1 = nn.Conv(args.n_colors, G0, kSize, padding=((kSize - 1) // 2), stride=1)
        self.SFENet2 = nn.Conv(G0, G0, kSize, padding=((kSize - 1) // 2), stride=1)
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))
        self.GFF = nn.Sequential(*[nn.Conv((self.D * G0), G0, 1, padding=0, stride=1), nn.Conv(G0, G0, kSize, padding=((kSize - 1) // 2), stride=1)])
        if args.no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = args.n_colors
            if ((r == 2) or (r == 3)):
                self.UPNet = nn.Sequential(*[nn.Conv(G0, ((G * r) * r), kSize, padding=((kSize - 1) // 2), stride=1), nn.PixelShuffle(r), nn.Conv(G, args.n_colors, kSize, padding=((kSize - 1) // 2), stride=1)])
            elif (r == 4):
                self.UPNet = nn.Sequential(*[nn.Conv(G0, (G * 4), kSize, padding=((kSize - 1) // 2), stride=1), nn.PixelShuffle(2), nn.Conv(G, (G * 4), kSize, padding=((kSize - 1) // 2), stride=1), nn.PixelShuffle(2), nn.Conv(G, args.n_colors, kSize, padding=((kSize - 1) // 2), stride=1)])
            else:
                raise ValueError('scale must be 2 or 3 or 4.')

    def execute(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(jt.contrib.concat(RDBs_out, dim=1))
        x += f__1
        if self.args.no_upsampling:
            return x
        else:
            return self.UPNet(x)

@register('rdn')
def make_rdn(G0=64, RDNkSize=3, RDNconfig='B', scale=2, no_upsampling=False):
    args = Namespace()
    args.G0 = G0
    args.RDNkSize = RDNkSize
    args.RDNconfig = RDNconfig
    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.n_colors = 3
    return RDN(args)





# ============================================================================
# MetaSR Model (from misc.py) 
# ============================================================================

@register('metasr')
class MetaSR(nn.Module):

    def __init__(self, encoder_spec):
        super().__init__()

        self.encoder = make(encoder_spec)
        imnet_spec = {
            'name': 'mlp',
            'args': {
                'in_dim': 3,
                'out_dim': self.encoder.out_dim * 9 * 3,
                'hidden_list': [256]
            }
        }
        self.imnet = make(imnet_spec)

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        feat = F.unfold(feat, 3, padding=1).view(
            feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        feat_coord = make_coord(feat.shape[-2:], flatten=False)
        feat_coord[:, :, 0] -= (2 / feat.shape[-2]) / 2
        feat_coord[:, :, 1] -= (2 / feat.shape[-1]) / 2
        feat_coord = feat_coord.permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        coord_ = coord.clone()
        coord_[:, :, 0] -= cell[:, :, 0] / 2
        coord_[:, :, 1] -= cell[:, :, 1] / 2
        coord_q = (coord_ + 1e-6).clamp(-1 + 1e-6, 1 - 1e-6)
        q_feat = F.grid_sample(
            feat, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord = F.grid_sample(
            feat_coord, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        rel_coord = coord_ - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2] / 2
        rel_coord[:, :, 1] *= feat.shape[-1] / 2

        r_rev = cell[:, :, 0] * (feat.shape[-2] / 2)
        inp = jt.contrib.concat([rel_coord, r_rev.unsqueeze(-1)], dim=-1)

        bs, q = coord.shape[:2]
        pred = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], 3)
        pred = jt.bmm(q_feat.contiguous().view(bs * q, 1, -1), pred)
        pred = pred.view(bs, q, 3)
        return pred

    def execute(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
