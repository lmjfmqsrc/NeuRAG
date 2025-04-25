import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.neumf import NeuMF
from utils.dataset import RetrievalDataset
from transformers import AutoTokenizer, AutoModel
import os
import json


# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据与模型
# 加载数据
def load_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    training_data = []
    for item in data:
        query = item['query']
        for pos in item['positive']:
            training_data.append((query, pos, 1.0))
        # 仅采样部分负样本
        #negatives = item['negative']
        #hard_negatives = negatives[:len(item['positive']) * 2]  # 负样本数量为正样本的2倍
        #for neg in hard_negatives:
        #    training_data.append((query, neg, 0.0))
        for neg in item['negative']:
            training_data.append((query, neg, 0.0))
    return training_data

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)
training_data1 = load_data('data/zh_train.json')
training_data = RetrievalDataset(training_data1, tokenizer, encoder, 'embedding_cache.pt')

train_loader = DataLoader(training_data, batch_size=16, shuffle=True)
neumf_model = NeuMF(mf_dim=128, layers=[1024, 512, 256, 128]).to(device)

# 优化器与学习率调度
optimizer = optim.AdamW(neumf_model.parameters(), lr=0.0005, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# 训练循环
for epoch in range(80):  # 设置合理的训练轮数
    neumf_model.train()
    total_loss = 0.0
    for query_emb, doc_emb, relevance in train_loader:
        # 确保输入数据在正确的设备上
        query_emb, doc_emb, relevance = query_emb.to(device), doc_emb.to(device), relevance.to(device)

        # 模型计算
        score = neumf_model(query_emb, doc_emb).view(-1)
        
        # 计算损失
        loss = torch.nn.BCELoss()(score, relevance)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f'Epoch [{epoch+1}/80], Loss: {total_loss/len(train_loader):.4f}')
    #scheduler.step()

# 保存模型权重
torch.save(neumf_model.state_dict(), 'neumf_model.pth')
