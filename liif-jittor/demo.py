import argparse
import os
from PIL import Image

import jittor as jt
import numpy as np

import models_jt
from utils_jittor import make_coord
from test import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # Jittor不需要这个

    # 将PIL图像转换为Jittor张量
    img_pil = Image.open(args.input).convert('RGB')
    img_arr = np.array(img_pil).astype('float32') / 255.0
    img_arr = img_arr.transpose(2, 0, 1)  # HWC -> CHW
    img = jt.array(img_arr)

    model = models_jt.make(jt.load(args.model)['model'], load_sd=True)

    h, w = list(map(int, args.resolution.split(',')))
    coord = make_coord((h, w))
    cell = jt.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, ((img - 0.5) / 0.5).unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).reshape(h, w, 3).permute(2, 0, 1)
    
    # 将Jittor张量转换为PIL图像并保存
    pred_np = pred.numpy()
    pred_np = (pred_np * 255).astype('uint8')
    pred_np = pred_np.transpose(1, 2, 0)  # CHW -> HWC
    pred_img = Image.fromarray(pred_np)
    pred_img.save(args.output)