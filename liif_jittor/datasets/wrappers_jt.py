import random
import math
from PIL import Image

import numpy as np
import jittor as jt
from jittor.dataset import Dataset
jt.flags.use_cuda = 1
from .datasets_jt import register
from utils_jittor import to_pixel_samples


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        super().__init__()
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q
# 必须告诉 jittor 总长度，否则会报错
        self.set_attrs(total_len = len(self.dataset))


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        #sr-implicit-paired和paired-image-folders配对使用，paired-image-folders中规定好了idx一次取两个
#   dataset:
#     name: paired-image-folders
#     args:
#       root_path_1: ./load/div2k/DIV2K_valid_LR_bicubic/X2
#       root_path_2: ./load/div2k/DIV2K_valid_HR
#   wrapper:
#     name: sr-implicit-paired

#      s缩放比例
        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
       #是否裁剪，如果inp_size为None，则不裁剪
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            #裁剪后随机取，保证两者像素位置绝对对应且大小缩放为s
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        # hr_coord/hr_rgb 是 jt.Var
        if self.sample_q is not None:
            N = hr_coord.shape[0]
            idx = jt.randperm(N)[:self.sample_q]   # jt.randperm 在 Jittor 中的 API 名称请按当前版本确认
            hr_coord = hr_coord[idx]
            hr_rgb = hr_rgb[idx]
        # if self.sample_q is not None:
        #     sample_lst = np.random.choice(
        #         len(hr_coord), self.sample_q, replace=False)
        #     hr_coord = hr_coord[sample_lst]
        #     hr_rgb = hr_rgb[sample_lst]

        cell = jt.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


# def resize_fn(img, size):
#     # Accept size as int or (H, W)
#     if isinstance(size, int):
#         target_h, target_w = size, size
#     else:
#         target_h, target_w = int(size[0]), int(size[1])

#     # Convert input to numpy HWC uint8 for PIL
#     if isinstance(img, jt.Var):
#         arr = img.numpy()  # CHW, float in [0,1]
#     else:
#         arr = np.asarray(img)

#     if arr.ndim == 3 and arr.shape[0] in (1, 3):
#         arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC

#     if arr.dtype != np.uint8:
#         arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

#     pil_img = Image.fromarray(arr)
#     pil_resized = pil_img.resize((target_w, target_h), resample=Image.BICUBIC)

#     out = np.asarray(pil_resized).astype('float32') / 255.0  # HWC in [0,1]
#     out = np.transpose(out, (2, 0, 1))  # HWC -> CHW
#     return jt.array(out)
#这个jittor的实现底层效果不同，影响性能对齐
# def resize_fn(img, size):
#     # Accept size as int or (H, W)
#     if isinstance(size, int):
#         target_h, target_w = size, size
#     else:
#         target_h, target_w = int(size[0]), int(size[1])

#     # 确保 img 是 4D 张量 [N, C, H, W]
#     if img.ndim == 3:
#         img = img.unsqueeze(0)  # [1, C, H, W]
    
#     # 使用正确的参数格式 (H, W)
#     out = jt.nn.resize(img, (target_h, target_w), mode='bicubic', align_corners=False)
#     return out.squeeze(0)  # 如果只处理单张图，去掉 batch 维
def resize_fn(img, size):
    """对齐与 torchvision.transforms 相同的 resize 效果，否则性能差异：复现踩坑点"""
    # 将 Jittor 张量转换为 PIL 图像
    img_np = img.numpy().transpose(1, 2, 0)  # CHW -> HWC
    img_np = (img_np * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    
    if isinstance(size, int):
        size = (size, size)
    resized_img = pil_img.resize(size, Image.BICUBIC)
    
    resized_np = np.array(resized_img, dtype=np.float32) / 255.0
    resized_np = resized_np.transpose(2, 0, 1)  # HWC -> CHW
    return jt.array(resized_np)


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        super().__init__()
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

# 必须告诉 jittor 总长度，否则会报错
        self.set_attrs(total_len = len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, (w_lr, w_lr))

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
        #print(crop_hr.shape,type(crop_hr))
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        # hr_coord/hr_rgb 是 jt.Var
        if self.sample_q is not None:
            N = hr_coord.shape[0]
            idx = jt.randperm(N)[:self.sample_q]   # jt.randperm 在 Jittor 中的 API 名称请按当前版本确认
            hr_coord = hr_coord[idx]
            hr_rgb = hr_rgb[idx]

        cell = jt.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

#inp:图，coord:像素坐标，cell:像素大小，gt:真实值

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

