import torch
import torch.nn as nn

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
