# PyTorch 到 Jittor API 映射参考

## 核心API映射表

| 功能类别 | PyTorch | Jittor | 说明 |
|---------|---------|--------|------|
| **导入** | `import torch` | `import jittor as jt` | 核心库 |
| **导入** | `import torch.nn as nn` | `from jittor import nn` | 神经网络 |
| **导入** | `from torch.utils.data import Dataset` | `from jittor.dataset import Dataset` | 数据集 |
| **导入** | `from torch.optim import Adam` | `from jittor.optim import Adam` | 优化器 |
| **方法名** | `def forward(self, x):` | `def execute(self, x):` | 前向传播 |
| **张量创建** | `torch.Tensor(data)` | `jt.array(data, dtype=jt.float32)` | 创建张量 |
| **张量连接** | `torch.cat(tensors, dim)` | `jt.contrib.concat(tensors, dim)` | 张量连接 |
| **张量堆叠** | `torch.stack(tensors, dim)` | `jt.stack(tensors, dim)` | 张量堆叠 |
| **单位矩阵** | `torch.eye(n)` | `jt.init.eye(n)` | 单位矩阵 |
| **等差数列** | `torch.arange(n)` | `jt.arange(n)` | 等差数列 |
| **网格生成** | `torch.meshgrid(*tensors)` | `jt.meshgrid(*tensors)` | 网格生成 |
| **随机排列** | `np.random.choice(N, k)` | `jt.randperm(N)[:k]` | 随机采样 |
| **对数运算** | `torch.log10(x)` | `jt.log(x) / jt.log(jt.array(10.0))` | 常用对数 |
| **无梯度** | `with torch.no_grad():` | `with jt.no_grad():` | 禁用梯度 |
| **模型保存** | `torch.save(model, path)` | `jt.save(model, path)` | 保存模型 |
| **模型加载** | `torch.load(path)` | `jt.load(path)` | 加载模型 |
| **形状改变** | `tensor.view(shape)` | `tensor.reshape(shape)` | 改变形状 |
| **数据加载** | `DataLoader(dataset, ...)` | `dataset.set_attrs(...)` | 数据加载 |
| **CUDA设置** | 自动处理 | `jt.flags.use_cuda = 1` | GPU设置 |

## 关键差异总结

1. **方法名**: `forward()` → `execute()`
2. **张量库**: `torch` → `jt`
3. **数据加载**: 移除 `DataLoader`，直接设置数据集属性
4. **内存优化**: 使用 `jt.reuse_np_array()` 优化内存
5. **数据类型**: 明确指定 `dtype=jt.float32`
6. **数据集长度**: 必须设置 `total_len`
7. **随机采样**: 使用 `jt.randperm()` 替代 `np.random.choice()`

