#!/bin/bash
# 双GPU并行训练脚本 - 充分利用GPU 0和GPU 1

set -e  # 遇到错误时停止执行

# 配置参数
MAX_PARALLEL_JOBS=2  # 最多同时运行的任务数（对应GPU数量）
GPUS=(0 1)           # 可用的GPU设备
LOG_DIR="./logs"     # 日志目录

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 简单的配置列表
configs=(
    "train-div2k/train_rdn-liif.yaml rdn_liif_v1"
    #"train-div2k/train_edsr-baseline-metasr.yaml edsr_metasr_v1"
    "train-div2k/ablation/train_edsr-baseline-liif-c.yaml edsr_liif_ablation_c"
    "train-div2k/ablation/train_edsr-baseline-liif-d.yaml edsr_liif_ablation_d"
    "train-div2k/ablation/train_edsr-baseline-liif-e.yaml edsr_liif_ablation_e"
    "train-div2k/ablation/train_edsr-baseline-liif-u.yaml edsr_liif_ablation_u"
    "train-div2k/ablation/train_edsr-baseline-liif-x2.yaml edsr_liif_ablation_x2"
    "train-div2k/ablation/train_edsr-baseline-liif-x3.yaml edsr_liif_ablation_x3"
    "train-div2k/ablation/train_edsr-baseline-liif-x4.yaml edsr_liif_ablation_x4"
)

# 函数：运行单个训练任务
run_training() {
    local config_path=$1
    local exp_name=$2
    local gpu_id=$3
    
    echo "[$(date '+%H:%M:%S')] 🚀 开始训练: ${exp_name} (GPU ${gpu_id})"
    
    # 重定向输出到日志文件
    python train.py --config ${config_path} --name ${exp_name} --gpu ${gpu_id} \
        > "${LOG_DIR}/${exp_name}_gpu${gpu_id}.log" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✅ 完成训练: ${exp_name} (GPU ${gpu_id})"
    else
        echo "[$(date '+%H:%M:%S')] ❌ 训练失败: ${exp_name} (GPU ${gpu_id}), 退出代码: ${exit_code}"
    fi
    
    return $exit_code
}

# 主训练循环
echo "🎯 开始批量并行训练 (最大并行数: ${MAX_PARALLEL_JOBS})"
echo "📝 日志保存路径: ${LOG_DIR}/"
echo "🖥️  可用GPU: ${GPUS[*]}"
echo "📋 总任务数: ${#configs[@]}"
echo "========================================"

running_jobs=0
gpu_index=0

for config_info in "${configs[@]}"; do
    read -r config_path exp_name <<< "${config_info}"
    
    # 如果达到最大并行数，等待一个任务完成
    if [ $running_jobs -ge $MAX_PARALLEL_JOBS ]; then
        wait -n  # 等待任意一个后台任务完成
        running_jobs=$((running_jobs - 1))
    fi
    
    # 分配GPU（轮流使用）
    gpu_id=${GPUS[$gpu_index]}
    gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))
    
    # 后台启动训练任务
    run_training "$config_path" "$exp_name" "$gpu_id" &
    running_jobs=$((running_jobs + 1))
    
    echo "📊 当前运行任务数: ${running_jobs}/${MAX_PARALLEL_JOBS}"
    echo "----------------------------------------"
    
    # 短暂延迟，避免同时启动导致资源冲突
    sleep 2
done

# 等待所有后台任务完成
echo "⏳ 等待所有训练任务完成..."
wait

echo ""
echo "🎉 所有训练任务已完成！"
echo "📝 检查日志文件: ls ${LOG_DIR}/"
echo "🔍 查看具体日志: tail -f ${LOG_DIR}/任务名_gpu*.log"