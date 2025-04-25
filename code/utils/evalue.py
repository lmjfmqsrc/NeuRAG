import os
import sys

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import json
import torch
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm  # 进度条
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel
from models.models import *  # 导入模型类
from models.neumf import NeuMF
from retrieval import baseline_retrieve, neumf_retrieve
from metrics import compute_precision_recall
from typing import List, Dict

# 创建 result 目录（如果不存在）
result_dir = os.path.join(project_root, "result")
os.makedirs(result_dir, exist_ok=True)

# 生成唯一的文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
result_file = os.path.join(result_dir, f"result_{timestamp}.txt")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
neumf_model = NeuMF(mf_dim=128, layers=[1024, 512, 256, 128]).to(device)

# 加载训练的NeuMF模型
neumf_model.load_state_dict(torch.load('neumf_model.pth', map_location=device))

# 加载数据集
data_path = os.path.join(project_root, 'data', 'zh_test.json')
if not os.path.exists(data_path):
    print(f"Error: Data file '{data_path}' does not exist.")
    sys.exit(1)

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化存储真实标签和预测标签的列表
true_labels = []
predicted_neumf_answers = []

# 生成答案的函数
def generate_answer(query, docs, model, instruction, temperature=0.8):
    """
    使用检索到的文档生成答案。
    """
    docs_text = '\n'.join(docs)
    text = instruction.format(QUERY=query, DOCS=docs_text)
    return model.generate(text, temperature).strip()  # 去除首尾空格

# 计算 F1 Score 作为答案正确率
def check_answer(pred: str, ground_truth):
    """
    检查生成答案是否正确：
    - ground_truth 可能是字符串或列表
    - 只要 ground_truth 中的某个答案是 pred 的子串，就认为答案正确
    """
    pred_lower = pred.lower()  # 统一转换成小写

    # 确保 ground_truth 是 **列表**（无论是 str 还是 list）
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]  # 如果是字符串，转为列表

    for gold in ground_truth:
        if isinstance(gold, str) and gold.lower() in pred_lower:  # 确保 gold 是字符串
            return 1.0  # 只要找到一个匹配，就判定为正确

    return 0.0  # 没有匹配到，则错误



# 检查生成答案是否正确
"""
def check_answer(prediction, ground_truth):
    #检查生成的答案是否与真实答案匹配
    prediction = prediction.lower()
    ground_truth = ground_truth.lower()
    return 1 if ground_truth in prediction else 0
"""

# 选择生成模型
def get_generation_model(model_name: str, api_key: str = None):
    """
    根据给定的模型名称返回对应的生成模型实例
    """
    if model_name == "chatglm":
        return ChatglmModel()
    elif model_name == "qwen":
        return Qwen()
    elif model_name == "qwen2":
        return Qwen2()
    elif model_name == "moss":
        return Moss()
    elif model_name == "vicuna":
        return Vicuna(plm='vicuna-13b')
    elif model_name == "wizardlm":
        return WizardLM(plm='wizardlm-13b')
    elif model_name == "baichuan":
        return Baichuan()
    elif model_name == "llama2":
        return LLama2(plm='llama-2-7b')
    elif model_name == "openai":
        if api_key:
            return OpenAIAPIModel(api_key=api_key)
        else:
            raise ValueError("API key is required for OpenAI model.")
    else:
        raise ValueError(f"Model {model_name} not recognized.")

# 在evalue.py中使用多个生成模型
model_name = "openai"  # 可以选择不同的模型， 比如 'chatglm', 'qwen', 'vicuna' 等

generate_model = get_generation_model(model_name, api_key)

# 打开 txt 文件，写入评估结果
with open(result_file, 'w', encoding='utf-8') as f:
    f.write(f"RAG 生成答案评估 - {timestamp}\n")
    f.write("=" * 50 + "\n")

    # 使用 tqdm 进度条处理数据
    for item in tqdm(data, desc="Processing queries", unit="query"):
        query = item['query']
        relevant_docs = set(item['positive'])
        all_docs = relevant_docs.union(set(item['negative']))

        # 生成答案：基于 NeuMF 检索
        neumf_results = neumf_retrieve(query, list(all_docs), neumf_model, tokenizer, bert_model, top_k=2)
        neumf_answer = generate_answer(query, neumf_results, generate_model, instruction="QUERY: {QUERY}\nDOCUMENTS:\n{DOCS}\nAnswer:")

        # 记录答案
        predicted_neumf_answers.append(neumf_answer)

        # 计算 F1 Score 作为正确性
        ground_truth = item['answer']
        correctness = check_answer(neumf_answer, ground_truth)
        
        # 在终端显示每个问题的正确性
        correctness_label = "✔ 正确" if correctness > 0 else "✘ 错误"
        print(f"Query: {query}")
        print(f"Generated Answer: {neumf_answer}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Correctness: {correctness:.2f} ({correctness_label})")
        print("-" * 50)

        # 记录到文件
        f.write(f"Query: {query}\n")
        f.write(f"Generated Answer: {neumf_answer}\n")
        f.write(f"Ground Truth: {ground_truth}\n")
        f.write(f"Correctness: {correctness:.2f} ({correctness_label})\n")
        f.write("-" * 50 + "\n")

# 计算整体 F1 Score 作为准确率
neumf_answer_accuracy = np.mean([check_answer(pred, item['answer']) for pred, item in zip(predicted_neumf_answers, data)])

# 输出最终评估结果
print(f"NeuMF Model Answer Accuracy: {neumf_answer_accuracy:.4f}")

# 追加写入总的 F1 Score
with open(result_file, 'a', encoding='utf-8') as f:
    f.write(f"\nNeuMF Model Answer Accuracy: {neumf_answer_accuracy:.4f}\n")
    f.write("=" * 50 + "\n")

print(f"Results saved to: {result_file}")