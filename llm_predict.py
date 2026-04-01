import os
import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import time 
# 设置 DeepSeek 的 API 密钥和基础 URL
API_KEY = 'sk-hQlK6fpOhsmAdlQe7f6f8164A9B448D9Ac252b698b2882D6'
BASE_URL = 'https://ai-api.wchscu.cn/v1/'

# 初始化 DeepSeek 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 读取问题集并随机抽样20%的数据
query_list = pd.read_json('./Data/processed_data.jsonl', lines=True)
# 设置随机种子确保结果可复现
np.random.seed(41)
# 随机抽样20%的数据
sample_size = int(query_list.shape[0] * 0.1)
query_list_sampled = query_list.sample(n=sample_size, random_state=42)
total_queries = query_list_sampled.shape[0]

print(f"总数据量: {query_list.shape[0]}, 抽样数据量: {total_queries}")


MODEL_LIST = [
    "deepseek-r1",
    "qwen2.5-vl-instruct",  
]

def evaluate_model(model_name, query_list_sampled, total_queries):
    """评估单个模型的性能"""
    predicted_labels = []
    true_labels = []
    
    for i in tqdm(range(total_queries), desc=f"处理模型 {model_name}"):
        query = query_list_sampled.iloc[i]['query']
        true_label = query_list_sampled.iloc[i]['label']

        messages = [
            {"role": "system", "content": "You are an expert in the field of sepsis."},
            {"role": "user", "content": query + " \n Please return only the final prediction result in a JSON format with the field name 'label'."}
        ]
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=False
            )
            
            if isinstance(response, str):
                content = response
            else:
                content = response.choices[0].message.content
            
            json_str = content.replace('```json\n', '').replace('\n```', '')
            predicted_label = json.loads(json_str)['label']
            predicted_labels.append(predicted_label)
            true_labels.append(true_label)
            
        except Exception as e:
            print(f"Error with model {model_name} on query {i}: {str(e)}")
            predicted_labels.append(None)
            true_labels.append(true_label)
    
    return predicted_labels, true_labels

MODEL_COLORS = {
    # "llama3.3-70b-instruct": plt.cm.Greens,     # 深绿，用于对比清晰且不刺眼
    "deepseek-r1": plt.cm.Purples,   
    "deepseek-v2": plt.cm.Reds,             # 深红，适合突出重要模型
    "qwen2.5-72b-instruct": plt.cm.Blues,       # 蓝色，学术图表常用

               # 紫色，专业且不易混淆
    # 可继续添加更多模型
}

def plot_confusion_matrix(y_true, y_pred, model_name):
    """绘制并保存混淆矩阵，为每个模型使用不同的专业配色"""
    cm = confusion_matrix(y_true, y_pred)
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 获取当前模型的配色方案，如果未定义则使用默认的灰度
    cmap = MODEL_COLORS.get(model_name, plt.cm.Greys)
    
    # 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Survival', 'Death']
    )
    
    disp.plot(
        cmap=cmap,
        values_format='d',
        ax=ax,
        colorbar=True
    )
    
    # 设置标题和标签
    plt.title(f"Confusion Matrix - llama3.3-70b", 
             fontweight='bold', 
             fontsize=14, 
             pad=20)
    
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
    
    # 调整刻度参数
    ax.tick_params(axis='both', labelsize=10, width=2)
    
    # 调整矩阵中数字的样式
    for text in disp.text_.ravel():
        text.set_fontsize(14)
        text.set_fontweight('bold')
        # 根据背景色调整文字颜色
        text.set_color('black')
    
    # 保存高质量图片
    plt.savefig(
        f"./Results/{model_name}_confusion_matrix.png",
        format='png',
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

# 创建结果目录
os.makedirs("./Results", exist_ok=True)

# 存储所有模型的结果
all_results = {}

# 遍历每个模型进行评估
for model_name in MODEL_LIST:
    print(f"\n开始评估模型: {model_name}")
    
    # 评估模型
    predicted_labels, true_labels = evaluate_model(model_name, query_list_sampled, total_queries)
    
    # 保存结果到Excel
    df_results = pd.DataFrame({
        'predicted_label': predicted_labels,
        'true_label': true_labels
    })
    df_results.to_excel(f'./Results/predictions_{model_name}.xlsx', index=False)
    
    # 计算指标
    df_clean = df_results.dropna()
    y_true = df_clean['true_label']
    y_pred = df_clean['predicted_label']
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    all_results[model_name] = metrics
    
    # 每个模型评估完后立即保存当前所有结果
    current_results_df = pd.DataFrame(all_results).T
    current_results_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    current_results_df = current_results_df.round(4)
    current_results_df.to_excel('./Results/models_comparison_current.xlsx')
    
    # 打印当前模型的评估结果
    print(f"\n{model_name} 评估结果:")
    print(current_results_df.loc[model_name])
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_true, y_pred, model_name)


# 将所有模型的结果整理成表格
results_df = pd.DataFrame(all_results).T
results_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
results_df = results_df.round(4)

# 保存总结果
results_df.to_excel('./Results/all_models_comparison.xlsx')
print("\n所有模型评估完成！结果已保存到 Results 目录")
print("\n模型性能对比:")
print(results_df)