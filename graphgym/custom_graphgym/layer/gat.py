# import torch.nn as nn
# from torch_geometric.graphgym.models import GATConv
# from torch_geometric.graphgym.register import register_layer

# @register_layer('GATConv')
# class TestGAT(nn.Module):
#     def __init__(self, dim_in, dim_out, bias=False, **kwargs):
#         super().__init__()
#         self.model = GATConv({dim_in}, bias=bias)

#     def forward(self, batch):
#         batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
#         return batch
