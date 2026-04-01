import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from DownstreamModel import DownstreamModel
from my_dataset import MyDataset

# 设置matplotlib默认样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['savefig.dpi'] = 300

def train_and_evaluate(data_path, device, batch_size, lr, epochs, early_stop):
    print(f'\nRunning: {data_path}')
    model = DownstreamModel(class_num=2).to(device)

    # 加载数据
    train_dataset = MyDataset('train', data_path)
    val_dataset = MyDataset('val', data_path)
    test_dataset = MyDataset('test', data_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 类别权重
    label_counts = Counter(train_dataset.labels.tolist())
    total = sum(label_counts.values())
    weights = [total / label_counts[i] for i in range(2)]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    save_dir = f'./Results/{os.path.basename(data_path)}_batch{batch_size}_LR{lr}'
    os.makedirs(save_dir, exist_ok=True)

    best_valid_loss = float('inf')
    early_stop_count = 0
     # 记录曲线
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss, correct_train, total_train = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss_list.append(total_train_loss / len(train_loader))
        train_acc_list.append(correct_train / total_train)

        # 验证阶段
        model.eval()
        total_val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss_list.append(total_val_loss / len(val_loader))
        val_acc_list.append(correct_val / total_val)

        if val_loss_list[-1] < best_valid_loss:
            best_valid_loss = val_loss_list[-1]
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= early_stop:
            print("Early stopping.")
            break

    # 测试
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    model.eval()
    test_probs, test_preds, test_labels = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_probs.extend(probs)
            test_preds.extend(preds)
            test_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(test_labels, test_probs)
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    print(f'{os.path.basename(data_path)} Test AUC: {auc:.4f}')
    return fpr, tpr, auc, os.path.basename(data_path), train_loss_list, val_loss_list, train_acc_list, val_acc_list

# ========= 主流程 ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='mps', type=str)
    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-lr', type=float, default=5e-5)
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-early_stop', type=int, default=10)
    args = parser.parse_args()

    model_paths = [
        './Data/II-Medical-8B',
        './Data/DeepSeek-R1-8B',
        './Data/Llama3.1-8B',
        './Data/Qwen3-8B',
        './Data/GLM4-9B',
        './Data/Qwen3-14B',
        './Data/DeepSeek-R1-14B',  # 如果需要，可以取消注释
    ]

    roc_results = []
    colors = [
    '#1f77b4',  # 蓝 - II-Medical-8B
    '#2ca02c',  # 绿 - DeepSeek-R1-8B
    '#9467bd',  # 紫 - Llama3.1-8B
    '#ff7f0e',  # 橙 - Qwen3-8B
    '#e377c2',  # 粉 - GLM4-9B
    '#8c564b',  # 棕 - Qwen3-14B
    '#17becf',  # 青 - DeepSeek-R1-14B
    ]   

    for idx, path in enumerate(model_paths):
        fpr, tpr, auc, name, train_loss, val_loss, train_acc, val_acc = train_and_evaluate(
            data_path=path,
            device=args.device,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            early_stop=args.early_stop
        )
        roc_results.append((fpr, tpr, auc, name))

        save_dir = f'./Results/{name}_batch{args.batch_size}_LR{args.lr}'
        os.makedirs(save_dir, exist_ok=True)

        # ✅ 合并绘制 Loss 和 Accuracy 曲线
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color_loss = colors[idx % len(colors)]
        color_acc = '#d62728'

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color_loss)
        ax1.plot(train_loss, label='Train Loss', color=color_loss, linestyle='-')
        ax1.plot(val_loss, label='Val Loss', color=color_loss, linestyle='--')
        ax1.tick_params(axis='y', labelcolor=color_loss)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color=color_acc)
        ax2.plot(train_acc, label='Train Acc', color=color_acc, linestyle='-')
        ax2.plot(val_acc, label='Val Acc', color=color_acc, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color_acc)

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        plt.title(f'Loss & Accuracy - {name}')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'loss_accuracy_curve_{name}.png'))
        plt.close()