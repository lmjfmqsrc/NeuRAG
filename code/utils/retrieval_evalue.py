import os
import sys

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import json
import torch
from transformers import AutoTokenizer, AutoModel
from models.neumf import NeuMF
from retrieval import baseline_retrieve, neumf_retrieve
from metrics import compute_precision_recall,compute_map,compute_mrr
from typing import List, Dict
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score  # 导入计算精确率、召回率和F1分数的库

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
neumf_model = NeuMF(mf_dim=128, layers=[1024, 512, 256, 128]).to(device)

# 加载训练的NeuMF模型
neumf_model.load_state_dict(torch.load('neumf_model.pth', map_location=device))

# 加载数据集, 使用绝对路径加载数据文件
data_path = os.path.join(project_root, 'data', 'zh_test copy.json')
if not os.path.exists(data_path):
    print(f"Error: Data file '{data_path}' does not exist.")
    sys.exit(1)

print(f"Data path: {data_path}")

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
'''
# 测试和评估
top_k_values = [1,2,3,5]  # 设置不同的top_k值
# 初始化存储真实标签和预测标签的列表
true_labels = []
predicted_baseline_labels = []
predicted_neumf_labels = []

# 评估前的初始化，确保true_labels和predicted_labels在所有top_k下累积
for top_k in top_k_values:
    print(f"\nEvaluating with top_k = {top_k}")
    
    # 清空每次评估的存储内容
    current_true_labels = []
    current_predicted_baseline_labels = []
    current_predicted_neumf_labels = []
    
    for item in tqdm(data):
        query = item['query']
        relevant_docs = set(item['positive'])
        all_docs = relevant_docs.union(set(item['negative']))
        
        # 真实标签（二元标注，1 表示相关，0 表示不相关）
        true_label = [1 if doc in relevant_docs else 0 for doc in all_docs]
        current_true_labels.append(true_label)
        
        # Baseline 检索结果
        baseline_results = baseline_retrieve(query, all_docs, tokenizer, bert_model, top_k)
        current_predicted_baseline_labels.append([1 if doc in baseline_results else 0 for doc in all_docs])

        # NeuMF 检索结果
        neumf_results = neumf_retrieve(query, all_docs, neumf_model, tokenizer, bert_model, top_k)
        current_predicted_neumf_labels.append([1 if doc in neumf_results else 0 for doc in all_docs])

    # 将当前top_k的标签加入最终结果
    true_labels.extend(current_true_labels)
    predicted_baseline_labels.extend(current_predicted_baseline_labels)
    predicted_neumf_labels.extend(current_predicted_neumf_labels)

    print(f"--- Metrics for Top-{top_k} ---")

    # Baseline 模型
    p_b, r_b, f1_b = compute_precision_recall(true_labels, predicted_baseline_labels, top_k)
    map_b = compute_map(true_labels, predicted_baseline_labels, top_k)
    mrr_b = compute_mrr(true_labels, predicted_baseline_labels, top_k)

    print(f"Baseline - Precision@{top_k}: {p_b:.4f}, Recall@{top_k}: {r_b:.4f}, F1@{top_k}: {f1_b:.4f}")
    print(f"Baseline - MAP@{top_k}: {map_b:.4f}, MRR@{top_k}: {mrr_b:.4f}")

    # NeuMF 模型
    p_n, r_n, f1_n = compute_precision_recall(true_labels, predicted_neumf_labels, top_k)
    map_n = compute_map(true_labels, predicted_neumf_labels, top_k)
    mrr_n = compute_mrr(true_labels, predicted_neumf_labels, top_k)

    print(f"NeuMF - Precision@{top_k}: {p_n:.4f}, Recall@{top_k}: {r_n:.4f}, F1@{top_k}: {f1_n:.4f}")
    print(f"NeuMF - MAP@{top_k}: {map_n:.4f}, MRR@{top_k}: {mrr_n:.4f}")
    print("\n")
'''
import random

# 添加噪声的函数
def add_noise_to_data(data, noise_ratio=0.1):
    """
    向数据集添加噪声，噪声比例由 noise_ratio 控制
    noise_ratio: 噪声的比例 (0.1表示10%的噪声)
    """
    noisy_data = []
    for item in data:
        # 原始的相关文档集合
        relevant_docs = set(item['positive'])
        # 所有文档，包括负面文档
        all_docs = relevant_docs.union(set(item['negative']))
        
        # 计算噪声数量
        noise_count = int(len(relevant_docs) * noise_ratio)
        noisy_docs = random.sample(list(all_docs), noise_count)
        
        # 将噪声文档添加到相关文档中，形成新的"相关文档"
        noisy_relevant_docs = relevant_docs.union(noisy_docs)
        
        # 创建带噪声的数据项
        noisy_item = {
            'query': item['query'],
            'positive': list(noisy_relevant_docs),  # 更新相关文档
            'negative': item['negative']  # 保持负面文档不变
        }
        noisy_data.append(noisy_item)
    return noisy_data

# 计算 Precision, Recall, F1 的函数
def compute_precision_recall_f1(true_labels, predicted_labels):
    # Precision, Recall, F1 计算
    true_positives = sum([1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 1])
    false_positives = sum([1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 1])
    false_negatives = sum([1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 0])
    
    # Precision, Recall, F1
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1

# 实验：计算不同噪声比率对模型的影响
noise_ratios = [1.0,0.2, 0.4, 0.6]  # 噪声比率列表
top_k_values = [1, 3, 5]  # 可自由设置的 top_k

for noise_ratio in noise_ratios:
    print(f"\nEvaluating with Noise Ratio = {noise_ratio}")
    
    # 添加噪声后的数据
    noisy_data = add_noise_to_data(data, noise_ratio)
    
    for top_k in top_k_values:
        print(f"\nEvaluating with top_k = {top_k}")
        
        # 初始化存储真实标签和预测标签的列表
        true_labels = []
        predicted_baseline_labels = []
        predicted_neumf_labels = []
        
        for item in tqdm(noisy_data):
            query = item['query']
            relevant_docs = set(item['positive'])
            all_docs = relevant_docs.union(set(item['negative']))
            
            # 真实标签（二元标注，1 表示相关，0 表示不相关）
            true_label = [1 if doc in relevant_docs else 0 for doc in all_docs]
            true_labels.append(true_label)
            
            # Baseline 检索结果
            baseline_results = baseline_retrieve(query, all_docs, tokenizer, bert_model, top_k)
            predicted_baseline_labels.append([1 if doc in baseline_results else 0 for doc in all_docs])
            
            # NeuMF 检索结果
            neumf_results = neumf_retrieve(query, all_docs, neumf_model, tokenizer, bert_model, top_k)
            predicted_neumf_labels.append([1 if doc in neumf_results else 0 for doc in all_docs])
        
        print(f"--- Metrics for Top-{top_k} ---")
        
        # Baseline 模型
        p_b, r_b, f1_b = compute_precision_recall_f1(true_labels, predicted_baseline_labels)
        print(f"Baseline - Precision@{top_k}: {p_b:.4f}, Recall@{top_k}: {r_b:.4f}, F1@{top_k}: {f1_b:.4f}")

        # NeuMF 模型
        p_n, r_n, f1_n = compute_precision_recall_f1(true_labels, predicted_neumf_labels)
        print(f"NeuMF - Precision@{top_k}: {p_n:.4f}, Recall@{top_k}: {r_n:.4f}, F1@{top_k}: {f1_n:.4f}")
        print("\n")
