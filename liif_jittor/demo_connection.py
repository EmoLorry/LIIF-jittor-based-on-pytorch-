#!/usr/bin/env python3
"""
演示 model.train() 和 loss.backward() 如何连接的机制
"""

import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 3)
        self.layer2 = nn.Linear(3, 1)
    
    def forward(self, x):
        print(f"  Forward pass - training mode: {self.training}")
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

def demonstrate_connection():
    print("=" * 60)
    print("演示 model.train() 和 loss.backward() 的连接机制")
    print("=" * 60)
    
    # 1. 创建模型实例
    model = SimpleModel()
    print(f"1. 模型对象ID: {id(model)}")
    print(f"   模型在内存地址: {hex(id(model))}")
    
    # 2. 查看模型参数
    print("\n2. 模型参数:")
    for name, param in model.named_parameters():
        print(f"   {name}: id={id(param)}, shape={param.shape}")
        print(f"   参数地址: {hex(id(param))}")
        print(f"   requires_grad: {param.requires_grad}")
    
    # 3. model.train() 设置状态
    print(f"\n3. 调用 model.train() 前: training={model.training}")
    model.train()
    print(f"   调用 model.train() 后: training={model.training}")
    
    # 4. 前向传播 - 构建计算图
    print(f"\n4. 前向传播:")
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    print(f"   输入 x: id={id(x)}")
    
    pred = model(x)  # 这里会打印training状态
    print(f"   输出 pred: id={id(pred)}")
    print(f"   pred.grad_fn: {pred.grad_fn}")  # 显示计算图信息
    
    # 5. 计算损失
    target = torch.tensor([[3.0]])
    loss = nn.MSELoss()(pred, target)
    print(f"\n5. 损失计算:")
    print(f"   loss: {loss.item():.4f}, id={id(loss)}")
    print(f"   loss.grad_fn: {loss.grad_fn}")
    
    # 6. 查看计算图连接
    print(f"\n6. 计算图连接:")
    print(f"   loss -> {loss.grad_fn}")
    print(f"   pred -> {pred.grad_fn}")
    
    # 7. 反向传播前检查梯度
    print(f"\n7. 反向传播前的梯度:")
    for name, param in model.named_parameters():
        print(f"   {name}.grad: {param.grad}")
    
    # 8. 关键：loss.backward() 如何找到模型参数
    print(f"\n8. 执行 loss.backward():")
    print(f"   loss对象知道整个计算图")
    print(f"   autograd会沿着计算图回溯")
    loss.backward()
    
    # 9. 反向传播后检查梯度
    print(f"\n9. 反向传播后的梯度:")
    for name, param in model.named_parameters():
        print(f"   {name}.grad: {param.grad}")
        print(f"   梯度地址: {hex(id(param.grad)) if param.grad is not None else 'None'}")

def demonstrate_tensor_tracking():
    print("\n" + "=" * 60)
    print("演示PyTorch如何追踪tensor之间的连接")
    print("=" * 60)
    
    model = SimpleModel()
    model.train()
    
    # 创建输入
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    print(f"输入 x 的ID: {id(x)}")
    
    # 逐步前向传播，观察tensor的创建和连接
    print(f"\n逐步前向传播:")
    
    # Step 1: 第一层
    h1 = model.layer1(x)
    print(f"h1 = layer1(x), id={id(h1)}")
    print(f"h1.grad_fn = {h1.grad_fn}")
    
    # Step 2: ReLU
    h2 = torch.relu(h1)  
    print(f"h2 = relu(h1), id={id(h2)}")
    print(f"h2.grad_fn = {h2.grad_fn}")
    
    # Step 3: 第二层
    pred = model.layer2(h2)
    print(f"pred = layer2(h2), id={id(pred)}")
    print(f"pred.grad_fn = {pred.grad_fn}")
    
    # Step 4: 损失
    target = torch.tensor([[3.0]])
    loss = nn.MSELoss()(pred, target)
    print(f"loss = mse(pred, target), id={id(loss)}")
    print(f"loss.grad_fn = {loss.grad_fn}")
    
    print(f"\n计算图链条:")
    print(f"loss -> pred -> h2 -> h1 -> x")
    print(f"每个tensor都'记住'了它是如何产生的")
    
    # 反向传播
    print(f"\n反向传播时，autograd沿着这个链条回溯:")
    loss.backward()
    
    print(f"x.grad = {x.grad}")  # 输入的梯度
    for name, param in model.named_parameters():
        print(f"{name}.grad shape = {param.grad.shape}")

if __name__ == "__main__":
    demonstrate_connection()
    demonstrate_tensor_tracking()