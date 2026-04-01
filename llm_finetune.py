import os
import json
import random
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import numpy as np
# ============ 路径配置 ============
DATA_CONFIG = {
    'data_path': 'Data/processed_dataV2.jsonl',
    'checkpoint_path': './checkpoints',  # 用于存储训练过程中的检查点
    'best_model_path': './best_model',   # 专门用于存储最佳模型
    'log_dir': './logs'
}
# ============ 模型配置 ============
MODEL_CONFIG = {
    'model_name': "Qwen/Qwen2-1.5B",
    'pad_token_id': 151643,
    'label_map': {
        "survival": 0,
        "death": 1
    }
}

# ============ 训练配置 ============
TRAIN_CONFIG = {
    'num_train_epochs': 10,
    'learning_rate': 2e-5,
    'warmup_steps': 50,
    'weight_decay': 0.01,
    'logging_steps': 500,
    'save_steps': 10000,   # 改为10000，等于eval_steps
    'eval_steps': 10000,
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 1,
    'save_total_limit': 3 # 限制保存的检查点数量
}

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# 创建保存目录
os.makedirs(DATA_CONFIG['checkpoint_path'], exist_ok=True)
os.makedirs(DATA_CONFIG['best_model_path'], exist_ok=True)
os.makedirs(DATA_CONFIG['log_dir'], exist_ok=True)

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data.append(json.loads(line))
    return data

# 初始化tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CONFIG['model_name'], 
    num_labels=1  # 关键：设为1，适配Qwen2二分类
).to(device)
model.config.pad_token_id = MODEL_CONFIG['pad_token_id']

# 打印参数名，确认分类头名称
print("Model parameters:")
for name, param in model.named_parameters():
    print(name)

# 冻结除分类头外的所有参数（Qwen2分类头通常为'score'）
for name, param in model.named_parameters():
    if not name.startswith("score"):  # 只训练分类头
        param.requires_grad = False

# 数据加载和处理
data = read_jsonl(DATA_CONFIG['data_path'])
random.shuffle(data)

# 检查标签范围
for example in data:
    assert 0 <= example['label'] < len(MODEL_CONFIG['label_map']), f"Label out of range: {example['label']}"

# 将数据转换为datasets库的Dataset对象
dataset = Dataset.from_list(data)

# 将数据集拆分为训练集和测试集和验证集,7:2:1
first_split = dataset.train_test_split(test_size=0.3, seed=42)
train_dataset = first_split['train']  # 70% 用于训练

# 将临时测试集再次分割成测试集和验证集
# test_size=0.33 意味着临时测试集的 33% 将成为验证集（即总数据的 10%）
temp_split = first_split['test'].train_test_split(test_size=0.33, seed=42)
test_dataset = temp_split['train']    # 20% 用于测试
val_dataset = temp_split['test']      # 10% 用于验证

def preprocess_function(examples):
    # 不要返回 tensor，直接返回字典，Trainer会自动处理
    return tokenizer(examples['query'], truncation=True, padding=True)

# 对数据集进行预处理
encoded_train = train_dataset.map(preprocess_function, batched=True)
encoded_test = test_dataset.map(preprocess_function, batched=True)
encoded_val = val_dataset.map(preprocess_function, batched=True)

# 配置训练参数
training_args = TrainingArguments(
    output_dir=DATA_CONFIG['checkpoint_path'],  # 改用checkpoint_path存储检查点
    eval_steps=TRAIN_CONFIG['eval_steps'],
    eval_strategy="steps",
    learning_rate=TRAIN_CONFIG['learning_rate'],
    per_device_train_batch_size=TRAIN_CONFIG['per_device_train_batch_size'],
    per_device_eval_batch_size=TRAIN_CONFIG['per_device_eval_batch_size'],
    num_train_epochs=TRAIN_CONFIG['num_train_epochs'],
    weight_decay=TRAIN_CONFIG['weight_decay'],
    logging_dir=DATA_CONFIG['log_dir'],
    logging_steps=TRAIN_CONFIG['logging_steps'],
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_strategy="steps",
    save_steps=TRAIN_CONFIG['save_steps'],
    save_total_limit=TRAIN_CONFIG['save_total_limit'],
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits > 0).astype(int)  # 二分类阈值
    return {"accuracy": accuracy_score(labels, preds)}

# 定义Trainer

# 检查是否有已存在的checkpoint
latest_checkpoint = None
if os.path.isdir(DATA_CONFIG['checkpoint_path']):
    checkpoints = [os.path.join(DATA_CONFIG['checkpoint_path'], d) for d in os.listdir(DATA_CONFIG['checkpoint_path']) if d.startswith("checkpoint-")]
    if checkpoints:
        # 按照checkpoint编号排序，取最新
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        print(f"检测到已有checkpoint，将从 {latest_checkpoint} 继续训练。")
    else:
        print("未检测到已有checkpoint，将从头开始训练。")
else:
    print("未检测到已有checkpoint，将从头开始训练。")


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # 新增
)

# 训练（支持断点续训）
trainer.train(resume_from_checkpoint=latest_checkpoint)


# 保存最终模型
# 获取最佳模型路径
best_model_path = trainer.state.best_model_checkpoint
if best_model_path:
    print(f"Best model checkpoint found at: {best_model_path}")
    # 加载并保存最佳模型
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    # 保存到指定目录
    best_model.save_pretrained(DATA_CONFIG['best_model_path'])
    tokenizer.save_pretrained(DATA_CONFIG['best_model_path'])
    print(f"Best model saved to: {DATA_CONFIG['best_model_path']}")
else:
    print("Warning: No best model checkpoint found!")

# ===== 绘制训练过程中的loss和accuracy曲线 =====
import matplotlib.pyplot as plt

loss_values = []
eval_steps = []
accuracy_values = []

for log in trainer.state.log_history:
    if "loss" in log and "epoch" in log:
        loss_values.append(log["loss"])
        eval_steps.append(log["step"])
    if "eval_accuracy" in log:
        accuracy_values.append((log["step"], log["eval_accuracy"]))

# 绘制loss曲线
plt.figure()
plt.plot(eval_steps, loss_values, label="Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.savefig(os.path.join(DATA_CONFIG['log_dir'], "loss_curve.png"))
plt.close()

# 绘制accuracy曲线（如果有）
if accuracy_values:
    steps, accs = zip(*accuracy_values)
    plt.figure()
    plt.plot(steps, accs, label="Eval Accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Evaluation Accuracy Curve")
    plt.legend()
    plt.savefig(os.path.join(DATA_CONFIG['log_dir'], "accuracy_curve.png"))
    plt.close()
else:
    print("No accuracy logs found, accuracy curve will not be plotted.")


# 重新加载最佳的模型    
best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
best_model.to(device)
# 测试模型  
predictions = trainer.predict(encoded_test)
preds = (predictions.predictions > 0).astype(int)  # 二分类阈值
# 计算准确率、召回率、AUC值和F1值、绘制混淆矩阵、ROC曲线

# 获取真实标签
labels = encoded_test['label']

# 计算各项评估指标
accuracy = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

# 计算ROC曲线和AUC值
fpr, tpr, _ = roc_curve(labels, predictions.predictions)
roc_auc = auc(fpr, tpr)

# 打印评估结果
print("\n========= 模型评估结果 =========")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1值 (F1-score): {f1:.4f}")
print(f"AUC值 (AUC): {roc_auc:.4f}")

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(DATA_CONFIG['log_dir'], "confusion_matrix.png"))
plt.close()

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(DATA_CONFIG['log_dir'], "roc_curve.png"))
plt.close()

# 保存评估结果到文件
evaluation_results = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'auc': float(roc_auc)
}

with open(os.path.join(DATA_CONFIG['log_dir'], 'test_results.json'), 'w') as f:
    json.dump(evaluation_results, f, indent=4)


