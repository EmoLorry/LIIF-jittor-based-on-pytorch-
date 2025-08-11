# LIIF项目 PyTorch 到 Jittor API 映射参考

## 1. 基础导入对比

### 1.1 核心库导入
```python
# PyTorch版本
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

# Jittor版本
import jittor as jt
from jittor import nn
from jittor import nn as F
from jittor.dataset import Dataset
from jittor.optim import SGD, Adam
from jittor.lr_scheduler import MultiStepLR
jt.flags.use_cuda = 1
```

## 2. 张量操作API映射

### 2.1 张量创建
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `torch.Tensor(data)` | `jt.array(data, dtype=jt.float32)` | 创建张量 |
| `torch.FloatTensor(data)` | `jt.array(data, dtype=jt.float32)` | 创建浮点张量 |
| `torch.zeros(shape)` | `jt.zeros(shape)` | 创建零张量 |
| `torch.ones(shape)` | `jt.ones(shape)` | 创建一张量 |
| `torch.randn(shape)` | `jt.randn(shape)` | 创建随机正态分布张量 |
| `torch.eye(n)` | `jt.init.eye(n)` | 创建单位矩阵 |

### 2.2 张量形状操作
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `tensor.view(shape)` | `tensor.reshape(shape)` | 改变张量形状 |
| `tensor.contiguous()` | `tensor.contiguous()` | 确保内存连续 |
| `tensor.permute(dims)` | `tensor.permute(dims)` | 维度重排 |
| `tensor.transpose(dim1, dim2)` | `tensor.transpose(dim1, dim2)` | 转置 |

### 2.3 张量连接和堆叠
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `torch.cat(tensors, dim)` | `jt.contrib.concat(tensors, dim)` | 张量连接 |
| `torch.stack(tensors, dim)` | `jt.stack(tensors, dim)` | 张量堆叠 |
| `torch.bmm(tensor1, tensor2)` | `jt.bmm(tensor1, tensor2)` | 批量矩阵乘法 |

### 2.4 数学运算
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `torch.abs(tensor)` | `jt.abs(tensor)` | 绝对值 |
| `torch.log(tensor)` | `jt.log(tensor)` | 自然对数 |
| `torch.log10(tensor)` | `jt.log(tensor) / jt.log(jt.array(10.0))` | 常用对数 |
| `torch.exp(tensor)` | `jt.exp(tensor)` | 指数函数 |
| `torch.sqrt(tensor)` | `jt.sqrt(tensor)` | 平方根 |
| `torch.pow(tensor, exp)` | `jt.pow(tensor, exp)` | 幂运算 |

### 2.5 序列生成
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `torch.arange(start, end, step)` | `jt.arange(start, end, step)` | 等差数列 |
| `torch.meshgrid(*tensors)` | `jt.meshgrid(*tensors)` | 网格生成 |
| `torch.randperm(n)` | `jt.randperm(n)` | 随机排列 |

## 3. 神经网络模块API映射

### 3.1 基础层
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `nn.Conv2d(in, out, kernel)` | `nn.Conv(in, out, kernel)` | 2D卷积 |
| `nn.BatchNorm2d(channels)` | `nn.BatchNorm(channels)` | 批归一化 |
| `nn.ReLU()` | `nn.ReLU()` | ReLU激活 |
| `nn.PReLU(channels)` | `nn.PReLU(num_parameters=channels)` | PReLU激活 |
| `nn.PixelShuffle(upscale_factor)` | `nn.PixelShuffle(upscale_factor)` | 像素重排 |

### 3.2 前向传播方法
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `def forward(self, x):` | `def execute(self, x):` | 前向传播方法 |

### 3.3 参数操作
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `self.weight.data = value` | `self.weight.update(value)` | 更新权重 |
| `self.bias.data = value` | `self.bias.update(value)` | 更新偏置 |
| `p.requires_grad = False` | `p.stop_grad()` | 停止梯度计算 |

## 4. 数据集和加载器API映射

### 4.1 数据集基类
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `class Dataset:` | `class Dataset:` | 数据集基类 |
| `def __init__(self):` | `def __init__(self): super().__init__()` | 初始化 |
| `def __len__(self):` | `def __len__(self):` | 数据集长度 |
| `def __getitem__(self, idx):` | `def __getitem__(self, idx):` | 获取数据项 |

### 4.2 数据加载器
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `DataLoader(dataset, batch_size, shuffle, num_workers)` | `dataset.set_attrs(batch_size, shuffle, num_workers)` | 数据加载设置 |
| `pin_memory=True` | 自动处理 | 内存固定 |

### 4.3 数据集特殊设置
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| 无需设置 | `self.set_attrs(total_len=len)` | 设置数据集总长度 |

## 5. 优化器和学习率调度器

### 5.1 优化器
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `torch.optim.SGD(params, lr)` | `jittor.optim.SGD(params, lr)` | SGD优化器 |
| `torch.optim.Adam(params, lr)` | `jittor.optim.Adam(params, lr)` | Adam优化器 |

### 5.2 学习率调度器
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `MultiStepLR(optimizer, milestones, gamma)` | `MultiStepLR(optimizer, milestones, gamma)` | 多步学习率衰减 |

## 6. 模型保存和加载

### 6.1 模型保存
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `torch.save(state_dict, path)` | `jt.save(state_dict, path)` | 保存模型 |
| `model.load_state_dict(state_dict)` | `model.load_state_dict(state_dict)` | 加载模型参数 |

### 6.2 模型加载
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `torch.load(path)` | `jt.load(path)` | 加载模型文件 |

## 7. 梯度计算控制

### 7.1 无梯度计算
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `with torch.no_grad():` | `with jt.no_grad():` | 禁用梯度计算 |

### 7.2 梯度操作
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `optimizer.zero_grad()` | `optimizer.zero_grad()` | 清零梯度 |
| `loss.backward()` | `loss.backward()` | 反向传播 |
| `optimizer.step()` | `optimizer.step()` | 更新参数 |

## 8. 设备管理

### 8.1 GPU设置
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `tensor.cuda()` | `tensor.cuda()` | 移动到GPU |
| `model.cuda()` | 自动处理 | 模型移动到GPU |
| `os.environ['CUDA_VISIBLE_DEVICES']` | `os.environ['CUDA_VISIBLE_DEVICES']` | 设置可见GPU |

### 8.2 Jittor特有设置
| Jittor特有 | 说明 |
|-----------|------|
| `jt.flags.use_cuda = 1` | 启用CUDA |
| `jt.flags.enable_tuner = 1` | 启用调优器 |

## 9. 图像处理相关

### 9.1 图像转换
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `transforms.ToTensor()(img)` | `pil_to_jt_tensor(img)` | PIL图像转张量 |
| `transforms.ToPILImage()(tensor)` | 自定义函数 | 张量转PIL图像 |

### 9.2 内存优化
| Jittor特有 | 说明 |
|-----------|------|
| `jt.reuse_np_array(numpy_array)` | 复用numpy数组内存 |

## 10. 常见错误和解决方案

### 10.1 数据类型错误
```python
# 错误：未指定数据类型
jt.array([1, 2, 3])

# 正确：明确指定数据类型
jt.array([1, 2, 3], dtype=jt.float32)
```

### 10.2 维度操作错误
```python
# 错误：使用view()可能失败
tensor.view(-1, 3)

# 正确：使用reshape()更安全
tensor.reshape(-1, 3)
```

### 10.3 数据集长度错误
```python
# 错误：未设置total_len
class MyDataset(Dataset):
    def __init__(self):
        super().__init__()

# 正确：设置total_len
class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=1000)
```

### 10.4 随机采样错误
```python
# 错误：使用numpy随机采样
sample_lst = np.random.choice(N, sample_q)

# 正确：使用Jittor随机采样
idx = jt.randperm(N)[:sample_q]
```

## 11. 性能优化建议

### 11.1 内存管理
1. 使用 `jt.reuse_np_array()` 避免重复内存分配
2. 及时释放不需要的张量引用
3. 在CPU上进行图像预处理

### 11.2 数据类型优化
1. 明确指定 `dtype=jt.float32`
2. 避免不必要的类型转换
3. 使用合适的数据类型

### 11.3 CUDA优化
1. 启用 `jt.flags.use_cuda = 1`
2. 使用 `jt.flags.enable_tuner = 1` 进行性能调优
3. 合理设置批大小

## 12. 迁移检查清单

- [ ] 所有 `torch` 导入替换为 `jt`
- [ ] 所有 `forward()` 方法替换为 `execute()`
- [ ] 所有 `torch.cat()` 替换为 `jt.contrib.concat()`
- [ ] 所有 `torch.stack()` 替换为 `jt.stack()`
- [ ] 所有 `torch.eye()` 替换为 `jt.init.eye()`
- [ ] 所有 `torch.arange()` 替换为 `jt.arange()`
- [ ] 所有 `torch.meshgrid()` 替换为 `jt.meshgrid()`
- [ ] 所有 `torch.log10()` 替换为对数换底公式
- [ ] 所有 `torch.no_grad()` 替换为 `jt.no_grad()`
- [ ] 所有 `torch.save()` 替换为 `jt.save()`
- [ ] 所有 `torch.load()` 替换为 `jt.load()`
- [ ] 移除所有 `DataLoader` 封装
- [ ] 添加 `jt.flags.use_cuda = 1`
- [ ] 设置数据集 `total_len`
- [ ] 使用 `jt.randperm()` 替代 `np.random.choice()`
- [ ] 使用 `reshape()` 替代 `view()`
- [ ] 使用 `jt.reuse_np_array()` 优化内存
- [ ] 明确指定张量数据类型

