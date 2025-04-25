from embeddings import compute_embedding
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 在检索函数中确保返回 top_k 个文档
def baseline_retrieve(query, documents, tokenizer, model, top_k=5):
    query_emb = compute_embedding(query, tokenizer, model).to(device)
    scores = []
    for doc in documents:
        doc_emb = compute_embedding(doc, tokenizer, model).to(device)
        score = torch.dot(query_emb, doc_emb).item()
        scores.append((doc, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scores[:top_k]]

def neumf_retrieve(query, documents, neumf_model, tokenizer, bert_model, top_k=5):
    query_emb = compute_embedding(query, tokenizer, bert_model).to(device)
    scores = []
    for doc in documents:
        doc_emb = compute_embedding(doc, tokenizer, bert_model).to(device)
        score = neumf_model(query_emb, doc_emb).item()
        scores.append((doc, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scores[:top_k]]


def retrieve_documents(question, documents, dataset, neumf_model):
    neumf_model.eval()
    question_emb = dataset.cache.get_embedding(question, dataset.tokenizer, dataset.model).to(neumf_model.device)
    scores = []
    for doc in documents:
        doc_emb = dataset.cache.get_embedding(doc, dataset.tokenizer, dataset.model).to(neumf_model.device)
        score = neumf_model(question_emb, doc_emb).item()
        scores.append((doc, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)
