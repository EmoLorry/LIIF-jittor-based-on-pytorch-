import argparse
import os
import math
from functools import partial

import yaml
from tqdm import tqdm
import jittor as jt
import datasets
import utils_jittor
import models_jt
jt.flags.use_cuda = 1

def batched_predict(model, inp, coord, cell, bsize):
    with jt.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = jt.contrib.concat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = jt.array(t['sub'], dtype=jt.float32).reshape(1, -1, 1, 1)
    inp_div = jt.array(t['div'], dtype=jt.float32).reshape(1, -1, 1, 1)
    t = data_norm['gt']
    gt_sub  = jt.array(t['sub'], dtype=jt.float32).reshape(1, 1, -1)
    gt_div  = jt.array(t['div'], dtype=jt.float32).reshape(1, 1, -1)
    
    if eval_type is None:
        metric_fn = utils_jittor.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils_jittor.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils_jittor.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils_jittor.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with jt.no_grad():
                pred = model(inp, batch['coord'], batch['cell'])
        else:
            pred = batched_predict(model, inp,
                batch['coord'], batch['cell'], eval_bsize)
        pred = pred * gt_div + gt_sub
        pred = pred.clamp(0, 1)

        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.reshape(*shape).permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].reshape(*shape).permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    # Jittor不需要DataLoader，直接使用dataset
    dataset.set_attrs(
        batch_size=spec['batch_size'],
        shuffle=False,  # 测试时不需要打乱
        num_workers=8
    )
    loader = dataset

    model_spec = jt.load(args.model)['model']
    model = models_jt.make(model_spec, load_sd=True)

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)
    print('result: {:.4f}'.format(res))
