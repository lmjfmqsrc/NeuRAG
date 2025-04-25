import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import os
import json

# 设置环境变量


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NeuMF模型实现
# 修改 NeuMF 模型定义
class NeuMF(nn.Module):
    def __init__(self, mf_dim, layers, input_dim=768):
        super(NeuMF, self).__init__()
        self.mf_user_layer = nn.Linear(input_dim, mf_dim)
        self.mf_item_layer = nn.Linear(input_dim, mf_dim)

        mlp_modules = []
        for in_size, out_size in zip([1536] + layers[:-1], layers):
            mlp_modules.append(nn.Linear(in_size, out_size))
            mlp_modules.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_modules)

        self.predict_layer = nn.Linear(mf_dim + layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_emb, item_emb):
        mf_vector = self.mf_user_layer(user_emb) * self.mf_item_layer(item_emb)
        mlp_vector = self.mlp(torch.cat([user_emb, item_emb], dim=-1))
        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.sigmoid(self.predict_layer(predict_vector))
        return prediction


# 数据集定义
class RetrievalDataset(Dataset):
    def __init__(self, data, tokenizer, model, cache_file='embedding_cache.pt'):
        self.data = data
        self.tokenizer = tokenizer
        self.model = model
        self.cache = EmbeddingCache(cache_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, doc, label = self.data[idx]
        query_emb = self.cache.get_embedding(query, self.tokenizer, self.model)
        doc_emb = self.cache.get_embedding(doc, self.tokenizer, self.model)
        return query_emb, doc_emb, torch.tensor(label, dtype=torch.float32)

    def save_cache(self):
        self.cache.save_cache()
        #torch.save(self.cache, self.cache_file)  # 保存为 .pt 文件
        
class EmbeddingCache:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        if os.path.exists(cache_file):
            self.cache = torch.load(cache_file)  # 使用 torch.load 加载缓存
        else:
            self.cache = {}  # 初始化为空字典

    def get_embedding(self, text, tokenizer, model):
        if text in self.cache:
            return torch.tensor(self.cache[text])  # 返回缓存中的嵌入
        else:
            # 计算嵌入
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            self.cache[text] = embedding  # 保存到缓存
            return torch.tensor(embedding)

    def save_cache(self):
        torch.save(self.cache, self.cache_file)  # 使用 torch.save 保存缓存
            
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

# 初始化模型和优化器

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)
neumf_model = NeuMF(mf_dim=128, layers=[1024, 512, 256, 128],input_dim=768).to(device)
#optimizer = optim.Adam(neumf_model.parameters(), lr=0.0005)  # 减小学习率
optimizer = optim.AdamW(neumf_model.parameters(), lr=0.0005, weight_decay=1e-4)

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)


# 加载训练数据
training_data = load_data('data/zh_train.json')
# 初始化数据集和模型
train_dataset = RetrievalDataset(training_data, tokenizer, model, cache_file='embedding_cache.pt')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 训练模型
for epoch in range(80):
    neumf_model.train()
    total_loss = 0.0
    for query_emb, doc_emb, relevance in train_loader:
        query_emb, doc_emb, relevance = query_emb.to(device), doc_emb.to(device), relevance.to(device)
        score = neumf_model(query_emb, doc_emb).view(-1)
        loss = nn.BCELoss()(score, relevance)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch+1}/80], Loss: {total_loss/len(train_loader):.4f}')

# 训练完成后保存缓存
train_dataset.save_cache()

# 文档检索函数
def retrieve_documents(question, documents):
    neumf_model.eval()
    question_emb = train_dataset.encode_text(question)
    scores = []
    for doc in documents:
        doc_emb = train_dataset.encode_text(doc)
        #score = neumf_model(question_emb, doc_emb).item()
        score = neumf_model(question_emb, doc_emb).view(-1)  # 展平成一维
        scores.append((doc, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
