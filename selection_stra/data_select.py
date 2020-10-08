import torch
from torch import nn

class data_selection(nn.Module):
    def __init__(self, data):
        super(data_selection, self).__init__()
        self.data = data
        self.weight = nn.Parameter(
            torch.ones(len(data))*1./len(data), requires_grad=True)

    def get_weight(self, train_input, train_id):
        weight = self.weight[train_id]
        return torch.sigmoid(weight)

    def forward(self, loss, ls_id):
        weight_logits = self.weight
        weight = weight_logits.softmax(dim = 0) * len(self.data)
        return (weight[ls_id] * loss).mean()


