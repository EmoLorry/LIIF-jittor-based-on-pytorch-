import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import jittor as jt
# Jittor 内存管理配置
os.environ['JT_SAVE_MEM'] = '1'        # 启用内存节省模式
os.environ['cpu_mem_limit'] = '-1'     # CPU内存无限制
os.environ['device_mem_limit'] = '-1'  # GPU内存无限制

import argparse
import os
from tqdm import tqdm
import yaml

import jittor as jt
import jittor.nn as nn
from jittor.lr_scheduler import MultiStepLR
jt.flags.use_cuda = 1
jt.flags.enable_tuner = 1

import datasets
import models_jt
import utils_jittor
from test import eval_psnr
##dataset和warpper的时候用到torch需要重构为jittor
def make_data_loader(spec, tag=''):
    if spec is None:
        return None
# train_dataset:
#   dataset:
#     name: image-folder
#     args:
#       root_path: ./load/div2k/DIV2K_train_HR
#       repeat: 20
#       cache: in_memory
#   wrapper:
#     name: sr-implicit-downsampled
#     args:
#       inp_size: 48
#       scale_max: 4
#       augment: true
#       sample_q: 2304
#   batch_size: 64
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    #打印数据类型
    print(type(dataset[0]))
    # 打印第一个字段的值的类型（判断是否为Jittor张量）
    # 打印第一个字段的值的类型（判断是否为Jittor张量）
    first_key = next(iter(dataset[0].keys()))
    print(type(dataset[0][first_key]))  # 若输出<class 'jittor_core.Var'>，则为Jittor张量
    #DataLoader(dataset, batch_size=spec['batch_size'])
    #这里torch和jittor不一致，
    # Jittor无需dataloader封装，dataset即可
    dataset.set_attrs(
        batch_size=spec['batch_size'],  # 设置批大小
        shuffle=(tag == 'train'),       # 设置是否打乱
        num_workers=8                   # 设置工作进程数
    )
    # 直接基于设置好属性的数据集创建

    return dataset



def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():

    model = models_jt.make(config['model'])
    #实例化优化器
    optimizer = utils_jittor.make_optimizer(
        model.parameters(), config['optimizer'])
    epoch_start = 1

    ## 多步学习率衰减 参数 optimizer示例 milestones=[] gamma=0.5
    if config.get('multi_step_lr') is None:
        lr_scheduler = None
    else:
        lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils_jittor.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler



def train(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils_jittor.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = jt.array(t['sub'], dtype=jt.float32).reshape(1, -1, 1, 1).cuda()
    inp_div = jt.array(t['div'], dtype=jt.float32).reshape(1, -1, 1, 1).cuda()
    ##(1,1,1,1)可广播成（B,C,H,W）
    #for batch in tqdm(train_loader ）
    #inp = (batch['inp'] - inp_sub) / inp_div
    #gt = (batch['gt'] - gt_sub) / gt_div
    t = data_norm['gt']
    gt_sub  = jt.array(t['sub'], dtype=jt.float32).reshape(1, 1, -1).cuda()
    gt_div  = jt.array(t['div'], dtype=jt.float32).reshape(1, 1, -1).cuda()
    #(1,1,1)可广播成（H*W,C）
    #for batch in tqdm(train_loader ）
    #gt = (batch['gt'] - gt_sub) / gt_div

    total_batches = len(train_loader) // train_loader.batch_size
    for batch in tqdm(train_loader, total=total_batches, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()
        #批次张量：inp, coord, cell, gt

        # 等价于: inp = (batch['inp'] - 0.5) / 0.5
        # 结果: 将 [0, 1] 范围转换为 [-1, 1] 范围
        inp = (batch['inp'] - inp_sub) / inp_div
        pred = model(inp, batch['coord'], batch['cell'])
        gt = (batch['gt'] - gt_sub) / gt_div
        loss = loss_fn(pred, gt)

        #模型功能：由inp,coord,cell预测gt?
        #模型学习的是 fθ(coord, cell | Encoder(inp)) 的连续隐式函数。
        #coord/cell 决定“在哪、以多大尺度”重建；inp 的特征提供内容条件。
        #模型学的是一个“以输入图像特征为条件、以坐标为自变量的连续函数”。   
        train_loss.add(loss.item())
        # jittor 风格：传入 loss，内部会反向并更新
        optimizer.step(loss)  

        pred = None; loss = None

    return train_loss.item()



def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils_jittor.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)


    train_loader, val_loader = make_data_loaders()
    
    # 如果config中没有data_norm，则设置归一化为恒等变换
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    # print("运行到这里，按回车继续...")
    # input()
    # print("继续执行后面的代码")

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils_jittor.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        # Jittor 优化器结构可能与 PyTorch 不同，需要适配

        # Jittor 风格：直接获取学习率
        if hasattr(optimizer, 'lr'):
            lr = optimizer.lr
        elif hasattr(optimizer, 'learning_rate'):
            lr = optimizer.learning_rate
        else:
            # 从配置中获取学习率
            lr = config['optimizer']['args']['lr']
        
        # 调试信息：打印优化器属性
        if epoch == 1:
            print(f"Optimizer type: {type(optimizer)}")
            print(f"Optimizer attributes: {dir(optimizer)}")
            print(f"Learning rate: {lr}")
        
        writer.add_scalar('lr', lr, epoch)

        train_loss = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        log_info.append('lr={:.6f}'.format(lr))
        writer.add_scalars('loss', {'train': train_loss}, epoch)
        
        model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        jt.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            jt.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            model_ = model
            val_res = eval_psnr(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                jt.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils_jittor.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils_jittor.time_text(t), utils_jittor.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save_model_jittor', save_name)

    main(config, save_path)
