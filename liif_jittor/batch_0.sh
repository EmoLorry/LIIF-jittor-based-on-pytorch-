

    #!/bin/bash
# 简化版批量训练脚本 - 如果复杂版本有问题可以用这个


# 简单的配置列表
configs=(
    "train-div2k/train_edsr-baseline-liif.yaml edsr_liif_baseline"
    "train-div2k/ablation/train_edsr-baseline-liif-c.yaml edsr_liif_ablation_c"
    "train-div2k/ablation/train_edsr-baseline-liif-d.yaml edsr_liif_ablation_d"
    "train-div2k/ablation/train_edsr-baseline-liif-e.yaml edsr_liif_ablation_e"
    "train-div2k/ablation/train_edsr-baseline-liif-u.yaml edsr_liif_ablation_u"
    "train-div2k/ablation/train_edsr-baseline-liif-x2.yaml edsr_liif_ablation_x2"
    "train-div2k/ablation/train_edsr-baseline-liif-x3.yaml edsr_liif_ablation_x3"
    "train-div2k/ablation/train_edsr-baseline-liif-x4.yaml edsr_liif_ablation_x4"
    "train-div2k/train_edsr-baseline-metasr.yaml edsr_metasr_v1"
    "train-div2k/train_rdn-liif.yaml rdn_liif_v1"

)

for config_info in "${configs[@]}"; do
    read -r config_path exp_name <<< "${config_info}"
    echo "开始训练: ${exp_name}"
    python trainon0.py --config ${config_path} --name ${exp_name} --gpu 0
    echo "完成训练: ${exp_name}"
    echo "----------------------------------------"
done