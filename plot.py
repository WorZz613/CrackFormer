import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
log_dir = "log/"
dirs = os.listdir(log_dir)
metrics = ['train_loss', 'val_loss', 'pa', 'miou', 'recall', 'precision', 'dice', 'f1']

all_data = {}


for dir in dirs:
    file_path = os.path.join(log_dir,dir,'log.csv')
    df = pd.read_csv(file_path)
    exp_data = {'epoch': df['epoch'].values}
    for metric in metrics:
        exp_data[metric] = df[metric].values
    all_data[dir] = exp_data
    print(f"成功加载: {dir}")

for metric in metrics:
    has_data = False
    for exp_name, exp_data in all_data.items():
        if exp_data[metric] is not None:
            has_data = True
            break
    if not has_data:
        print(f"跳过 {metric}，无可用数据")
        continue
    plt.figure(figsize=(10, 6))
    for exp_name, exp_data in all_data.items():
        if exp_data[metric] is not None:
            plt.plot(exp_data['epoch'], exp_data[metric],
                     label=exp_name,
                     linewidth=2,
                     marker='o',
                     markersize=4)
    plt.title(f'{metric.upper()} Comparison', fontweight='bold')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel(metric.upper(), fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')

    if 'loss' in metric:
        plt.ylim(bottom=0)
    else:
        plt.ylim(0, 1.0)
    if metric in ['dice','f1','miou','precision','precision']:
        plt.ylim(0.5, 1.0)
    elif metric == 'pa':
        plt.ylim(0.8, 1.0)
    else:
        pass
    os.makedirs(f'log_comparison/',exist_ok=True)
    output_path = f'log_comparison/{metric}_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {output_path}")
    plt.close()
print("所有指标对比图表已生成完毕！")