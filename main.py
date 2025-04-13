import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from model import NeuralNetwork, accuracy
from data_loader import load_cifar10
from train import train

# 创建保存目录
base_path = '/home/add_disk/zengchengjie/HW'
os.makedirs(f'{base_path}/models', exist_ok=True)
os.makedirs(f'{base_path}/curves', exist_ok=True)
os.makedirs(f'{base_path}/weights', exist_ok=True)

# 加载数据
data_path = '/home/add_disk/zengchengjie/data/cifar-10-batches-py'
X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(data_path)

# 超参数搜索空间
param_grid = {
    'hidden_size': [512, 1024],
    'lr': [0.05, 0.1],
    'lambda_reg': [0.001, 0.01],
    'batch_size': [256]
}

# 存储所有实验结果
results = []

def run_experiment(hidden_size, lr, lambda_reg, batch_size):
    # 生成唯一标识符
    exp_id = f"h{hidden_size}_lr{lr}_reg{lambda_reg}_bs{batch_size}"
    print(f"\n=== Start Experiment {exp_id} ===")
    
    # 初始化模型
    model = NeuralNetwork(input_size=3072, 
                         hidden_size=hidden_size,
                         output_size=10,
                         activation='relu')
    
    # 训练参数
    params = {
        'lr': lr,
        'batch_size': batch_size,
        'epochs': 100,
        'lambda_reg': lambda_reg,
        'lr_step': 20,
        'lr_decay': 0.1
    }
    
    # 训练模型
    history = train(model, X_train, y_train, X_val, y_val, params)
    best_val_acc = max(history['val_acc'])
    
    # 保存当前实验的模型
    model_path = f'{base_path}/models/{exp_id}_model.npz'
    np.savez(model_path, 
            W1=model.W1, b1=model.b1,
            W2=model.W2, b2=model.b2)
    
    # 保存训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'Loss Curve ({exp_id})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(history['val_acc'])
    plt.title(f'Val Accuracy ({exp_id})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(f'{base_path}/curves/{exp_id}_curves.png')
    plt.close()
    
    # 保存权重可视化
    plt.figure(figsize=(10, 5))
    for i in range(20):
        plt.subplot(4, 5, i+1)
        w = model.W1[:, i].reshape(32, 32, 3)
        w = (w - w.min()) / (w.max() - w.min())
        plt.imshow(w)
        plt.axis('off')
    plt.suptitle(f'First Layer Weights ({exp_id})')
    plt.savefig(f'{base_path}/weights/{exp_id}_weights.png')
    plt.close()
    
    print(f"=== Experiment {exp_id} Completed ===")
    return {
        'exp_id': exp_id,
        'hidden_size': hidden_size,
        'lr': lr,
        'lambda_reg': lambda_reg,
        'batch_size': batch_size,
        'val_accuracy': best_val_acc,
        'history': history
    }

# 执行网格搜索
# for hidden_size in param_grid['hidden_size']:
#     for lr in param_grid['lr']:
#         for lambda_reg in param_grid['lambda_reg']:
#             for batch_size in param_grid['batch_size']:
#                 result = run_experiment(hidden_size, lr, lambda_reg, batch_size)
#                 results.append(result)

# 保存实验结果
df_results = pd.DataFrame([{k:v for k,v in r.items() if k != 'history'} for r in results])
df_results.to_csv(f'{base_path}/hyperparam_results.csv', index=False)

# 生成热力图（添加在最后）
import seaborn as sns
plt.figure(figsize=(10, 6))
pivot_table = df_results.pivot_table(values='val_accuracy',
                                   index='hidden_size',
                                   columns='lr',
                                   aggfunc='mean')
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Validation Accuracy Heatmap")
plt.savefig(f'{base_path}/hyperparam_heatmap.png')

print("\nAll experiments completed!")
print(f"Find results in: {base_path}")
print(f"- Models: {base_path}/models/ (8 models)")
print(f"- Curves: {base_path}/curves/ (8 training curves)")
print(f"- Weights: {base_path}/weights/ (8 weight visualizations)")