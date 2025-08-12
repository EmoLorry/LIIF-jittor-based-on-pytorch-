# LIIF项目 PyTorch 到 Jittor 迁移文档

## 项目概述

本项目实现了LIIF (Local Implicit Image Function) 超分辨率算法，包含两个版本：
- `liif_torch/`: 基于PyTorch的原始实现
- `liif_jittor/`: 基于Jittor的迁移实现

## 1. 项目框架（文件结构）

### 1.1 整体目录结构对比

```
LIIF/
├── liif_torch/                    # PyTorch版本
│   ├── datasets/                  # 数据集相关
│   │   ├── __init__.py
│   │   ├── datasets.py           # 数据集注册
│   │   ├── image_folder.py       # 图像文件夹数据集
│   │   └── wrappers.py           # 数据包装器
│   ├── logs/                     # 训练日志
│   ├── save_model/               # 模型保存
│   ├── train-div2k/              # 训练数据
│   ├── models.py                 # 模型定义
│   ├── train.py                  # 训练脚本
│   ├── test.py                   # 测试脚本
│   ├── utils.py                  # 工具函数
│   └── *.yaml                    # 配置文件
│
└── liif_jittor/                  # Jittor版本
    ├── datasets/                 # 数据集相关
    │   ├── __init__.py
    │   ├── datasets_jt.py        # 数据集注册
    │   ├── image_folder_jt.py    # 图像文件夹数据集
    │   └── wrappers_jt.py        # 数据包装器
    ├── logs/                     # 训练日志
    ├── save_model_jittor/        # 模型保存
    ├── train-div2k/              # 训练数据
    ├── models_jt.py              # 模型定义
    ├── models_torch.py           # 原始PyTorch模型（参考）
    ├── train.py                  # 训练脚本
    ├── test.py                   # 测试脚本
    ├── utils_jittor.py           # 工具函数
    ├── cuda.py                   # CUDA相关优化
    └── *.yaml                    # 配置文件
```

### 1.2 核心文件对应关系

| PyTorch版本 | Jittor版本 | 功能描述 |
|------------|-----------|----------|
| `models.py` | `models_jt.py` | 核心模型定义 |
| `train.py` | `train.py` | 训练脚本 |
| `test.py` | `test.py` | 测试脚本 |
| `utils.py` | `utils_jittor.py` | 工具函数 |
| `datasets/wrappers.py` | `datasets/wrappers_jt.py` | 数据包装器 |
| `datasets/image_folder.py` | `datasets/image_folder_jt.py` | 图像数据集 |

## 2. 文件之间的关系

### 2.1 训练流程关系图

```
配置文件(.yaml) 
    ↓
train.py (训练入口)
    ↓
models_jt.py (模型定义)
    ↓
datasets/ (数据加载)
    ↓
utils_jittor.py (工具函数)
    ↓
test.py (验证评估)
```
### 2.2 模型组件关系

```
LIIF模型
├── EDSR编码器 (models_jt.py)
├── MLP网络 (models_jt.py)
└── 坐标查询机制 (models_jt.py)
```

## 3. 各文件不同的地方要点总结

### 3.1 模型定义文件 (`models.py` vs `models_jt.py`)

#### 3.1.1 导入差异
```python
# PyTorch版本
import torch
import torch.nn as nn
import torch.nn.functional as F

# Jittor版本
import jittor as jt
from jittor import nn
from jittor import nn as F
jt.flags.use_cuda = 1
```

#### 3.1.2 前向传播方法名
```python
# PyTorch版本
def forward(self, x):
    # 前向传播逻辑

# Jittor版本
def execute(self, x):
    # 前向传播逻辑
```

#### 3.1.3 张量操作差异
```python
# PyTorch版本
torch.cat([tensor1, tensor2], dim=1)
torch.stack(tensors, dim=0)
torch.eye(3)
torch.Tensor(data)

# Jittor版本
jt.contrib.concat([tensor1, tensor2], dim=1)
jt.stack(tensors, dim=0)
jt.init.eye(3)
jt.array(data, dtype=jt.float32)
```

#### 3.1.4 参数初始化差异
```python
# PyTorch版本
self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std

# Jittor版本
eye = jt.init.eye(3).reshape(3, 3, 1, 1)
self.weight.update(eye / std.reshape(3, 1, 1, 1))
bias = (sign * rgb_range) * jt.array(rgb_mean, dtype=jt.float32) / std
self.bias.update(bias)
```

### 3.2 训练脚本 (`train.py`)

#### 3.2.1 数据加载器差异
```python
# PyTorch版本
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=spec['batch_size'],
    shuffle=(tag == 'train'), num_workers=8, pin_memory=True)

# Jittor版本
# Jittor无需DataLoader封装，直接设置数据集属性
dataset.set_attrs(
    batch_size=spec['batch_size'],
    shuffle=(tag == 'train'),
    num_workers=8
)
return dataset
```

#### 3.2.2 张量创建差异
```python
# PyTorch版本
inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()

# Jittor版本
inp_sub = jt.array(t['sub'], dtype=jt.float32).reshape(1, -1, 1, 1)
inp_div = jt.array(t['div'], dtype=jt.float32).reshape(1, -1, 1, 1)
```

#### 3.2.3 模型保存差异
```python
# PyTorch版本
torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

# Jittor版本
jt.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))
```

### 3.3 工具函数 (`utils.py` vs `utils_jittor.py`)

#### 3.3.1 优化器导入差异
```python
# PyTorch版本
from torch.optim import SGD, Adam

# Jittor版本
from jittor.optim import SGD, Adam
```

#### 3.3.2 坐标生成函数差异
```python
# PyTorch版本
seq = v0 + r + (2 * r) * torch.arange(n).float()
ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)

# Jittor版本
seq = v0 + r + (2 * r) * jt.arange(n).float()
ret = jt.stack(jt.meshgrid(*coord_seqs), dim=-1)
```

#### 3.3.3 PSNR计算差异
```python
# PyTorch版本
return -10 * torch.log10(mse)

# Jittor版本
return -10 * jt.log(mse) / jt.log(jt.array(10.0))
```

### 3.4 数据集文件

#### 3.4.1 图像文件夹数据集 (`image_folder.py` vs `image_folder_jt.py`)

**图像转换差异：**
```python
# PyTorch版本
return transforms.ToTensor()(Image.open(x).convert('RGB'))

# Jittor版本
def pil_to_jt_tensor(img_pil):
    arr = np.array(img_pil)
    arr = arr.astype('float32') / 255.0
    arr = arr.transpose(2, 0, 1).copy()
    return jt.array(arr)
```

### 3.5 测试脚本 (`test.py`)


#### 3.5.1 张量连接差异
```python
# PyTorch版本
pred = torch.cat(preds, dim=1)

# Jittor版本
pred = jt.contrib.concat(preds, dim=1)
```

