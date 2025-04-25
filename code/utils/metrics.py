from typing import List

def compute_precision_recall(true_labels, predicted_labels, top_k):
    # 确保每个查询的true_labels和predicted_labels长度一致，且无重复
    assert len(true_labels) == len(predicted_labels), "true_labels and predicted_labels must have the same length"
    
    precision_list = []
    recall_list = []
    f1_list = []
    
    for true, pred in zip(true_labels, predicted_labels):
        # 针对每个查询，计算精确率、召回率和F1
        true_positive = sum([1 for t, p in zip(true, pred) if t == p == 1])
        false_positive = sum([1 for t, p in zip(true, pred) if t == 0 and p == 1])
        false_negative = sum([1 for t, p in zip(true, pred) if t == 1 and p == 0])

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # 返回平均值
    precision = sum(precision_list) / len(precision_list)
    recall = sum(recall_list) / len(recall_list)
    f1 = sum(f1_list) / len(f1_list)
    
    return precision, recall, f1


def compute_map(true_labels: List[List[int]], predicted_labels: List[List[int]], top_k: int) -> float:
    ap_list = []
    for true, pred in zip(true_labels, predicted_labels):
        relevant_indices = [i for i, label in enumerate(true) if label == 1]
        score, hit_count = 0.0, 0
        for i, doc in enumerate(pred[:top_k]):
            if i in relevant_indices:
                hit_count += 1
                score += hit_count / (i + 1)
        ap = score / len(relevant_indices) if relevant_indices else 0
        ap_list.append(ap)

    return sum(ap_list) / len(ap_list)

def compute_mrr(true_labels: List[List[int]], predicted_labels: List[List[int]], top_k: int) -> float:
    rr_list = []
    for true, pred in zip(true_labels, predicted_labels):
        relevant_indices = [i for i, label in enumerate(true) if label == 1]
        for i, doc in enumerate(pred[:top_k]):
            if i in relevant_indices:
                rr_list.append(1 / (i + 1))
                break
        else:
            rr_list.append(0)

    return sum(rr_list) / len(rr_list)
