#!/usr/bin/env python3
"""
训练对齐可视化脚本
为每个实验生成PyTorch和Jittor框架的训练曲线对比图（前100轮）
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_loss_file(file_path):
    """解析训练日志文件，提取epoch、loss和val_psnr（前100轮）"""
    epochs = []
    losses = []
    val_psnrs = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 匹配格式: epoch: 1  loss: 0.0808  val_psnr: 27.2101
                match = re.match(r'epoch: (\d+)\s+loss: ([\d.]+)\s+val_psnr: ([\d.]+)', line.strip())
                if match:
                    epoch = int(match.group(1))
                    # 只处理前100轮数据
                    if epoch <= 100:
                        epochs.append(epoch)
                        losses.append(float(match.group(2)))
                        val_psnrs.append(float(match.group(3)))
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return [], [], []
    
    return epochs, losses, val_psnrs

def plot_single_experiment(exp_name, save_dir="training_plots"):
    """为单个实验生成对比图（前100轮）"""
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 文件路径
    torch_loss_file = f"modelloss_and_modelscore/torch_loss/{exp_name}_loss.txt"
    jittor_loss_file = f"modelloss_and_modelscore/jittor_loss/{exp_name}_loss.txt"
    
    # 解析数据
    torch_epochs, torch_losses, torch_psnrs = parse_loss_file(torch_loss_file)
    jittor_epochs, jittor_losses, jittor_psnrs = parse_loss_file(jittor_loss_file)
    
    if not torch_epochs or not jittor_epochs:
        print(f"无法读取实验 {exp_name} 的训练数据文件")
        return False
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 实验名称格式化
    exp_title = exp_name.replace('_', ' ').title()
    
    # 绘制Loss对比
    ax1.plot(torch_epochs, torch_losses, 'b-', linewidth=2, label='PyTorch', alpha=0.8)
    ax1.plot(jittor_epochs, jittor_losses, 'r-', linewidth=2, label='Jittor', alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title(f'{exp_title} - 训练Loss对比 (前100轮)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    
    # 绘制PSNR对比
    ax2.plot(torch_epochs, torch_psnrs, 'b-', linewidth=2, label='PyTorch', alpha=0.8)
    ax2.plot(jittor_epochs, jittor_psnrs, 'r-', linewidth=2, label='Jittor', alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation PSNR (dB)', fontsize=12)
    ax2.set_title(f'{exp_title} - 验证PSNR对比 (前100轮)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    
    # 添加统计信息
    torch_final_loss = torch_losses[-1] if torch_losses else 0
    jittor_final_loss = jittor_losses[-1] if jittor_losses else 0
    torch_final_psnr = torch_psnrs[-1] if torch_psnrs else 0
    jittor_final_psnr = jittor_psnrs[-1] if jittor_psnrs else 0
    
    info_text = f"""第100轮收敛结果:
PyTorch: Loss={torch_final_loss:.4f}, PSNR={torch_final_psnr:.2f}dB
Jittor:  Loss={jittor_final_loss:.4f}, PSNR={jittor_final_psnr:.2f}dB
Loss差异: {abs(torch_final_loss - jittor_final_loss):.4f}
PSNR差异: {abs(torch_final_psnr - jittor_final_psnr):.2f}dB"""
    
    fig.text(0.02, 0.02, info_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    filename = f"{save_dir}/{exp_name}_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    
    print(f"实验 {exp_name} 对比图已保存为: {filename}")
    print(f"  PyTorch第100轮Loss: {torch_final_loss:.4f}, PSNR: {torch_final_psnr:.2f}dB")
    print(f"  Jittor第100轮Loss: {jittor_final_loss:.4f}, PSNR: {jittor_final_psnr:.2f}dB")
    
    return True

def plot_all_experiments():
    """为所有实验生成对比图（前100轮）"""
    
    # 实验配置
    experiments = [
        'edsr_liif_baseline',
        'edsr_liif_ablation_c',
        'edsr_liif_ablation_d', 
        'edsr_liif_ablation_e',
        'edsr_liif_ablation_u',
        'edsr_liif_ablation_x2',
        'edsr_liif_ablation_x3',
        'edsr_liif_ablation_x4',
        'edsr_metasr_v1',
        'rdn_liif_v1'
    ]
    
    print("开始为每个实验生成训练对比图（前100轮）...")
    print("=" * 60)
    
    success_count = 0
    for exp_name in experiments:
        print(f"\n处理实验: {exp_name}")
        if plot_single_experiment(exp_name):
            success_count += 1
        print("-" * 40)
    
    print(f"\n完成！成功生成 {success_count}/{len(experiments)} 个实验的对比图")
    print(f"图片保存在: training_plots/ 目录下")

def create_summary_plot():
    """创建所有实验的汇总对比图（前100轮）"""
    
    experiments = [
        'edsr_liif_baseline',
        'edsr_liif_ablation_c',
        'edsr_liif_ablation_d', 
        'edsr_liif_ablation_e',
        'edsr_liif_ablation_u',
        'rdn_liif_v1'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, exp_name in enumerate(experiments):
        torch_file = f"modelloss_and_modelscore/torch_loss/{exp_name}_loss.txt"
        jittor_file = f"modelloss_and_modelscore/jittor_loss/{exp_name}_loss.txt"
        
        torch_epochs, torch_losses, _ = parse_loss_file(torch_file)
        jittor_epochs, jittor_losses, _ = parse_loss_file(jittor_file)
        
        if torch_epochs and jittor_epochs:
            axes[i].plot(torch_epochs, torch_losses, 'b-', linewidth=1.5, label='PyTorch', alpha=0.8)
            axes[i].plot(jittor_epochs, jittor_losses, 'r-', linewidth=1.5, label='Jittor', alpha=0.8)
            axes[i].set_title(f'{exp_name.replace("_", " ").title()}', fontsize=12)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, 100)
            
            # 添加第100轮loss值
            torch_final = torch_losses[-1] if torch_losses else 0
            jittor_final = jittor_losses[-1] if jittor_losses else 0
            axes[i].text(0.02, 0.98, f'P: {torch_final:.4f}\nJ: {jittor_final:.4f}', 
                        transform=axes[i].transAxes, fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.suptitle('所有实验训练Loss对比汇总 (PyTorch vs Jittor) - 前100轮', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_plots/all_experiments_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("汇总对比图已保存为: training_plots/all_experiments_summary.png")

if __name__ == "__main__":
    print("开始生成训练对齐可视化图表（前100轮）...")
    
    # 为每个实验生成单独的对比图
    plot_all_experiments()
    
    # 生成汇总图
    create_summary_plot()
    
    print("\n所有图表生成完成！")
    print("每个实验的单独对比图保存在: training_plots/ 目录下")
    print("注意：所有图表显示的是前100轮的训练数据")
