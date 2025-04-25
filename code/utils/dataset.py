import torch
from torch.utils.data import Dataset
import os

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EmbeddingCache:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache = torch.load(cache_file) if os.path.exists(cache_file) else {}

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
        torch.save(self.cache, self.cache_file)

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
        
    
