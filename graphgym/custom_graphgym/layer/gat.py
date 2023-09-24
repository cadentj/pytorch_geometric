from torch_geometric.nn import GATConv

@register_layer('GATConv')
class ExampleConv2(nn.Module):
    def __init__(self, dim_in, dim_out, heads, bias=False, **kwargs):
        super().__init__()
        self.model = GATConv(dim_in, dim_out, heads, bias=bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch
