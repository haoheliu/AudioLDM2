import torch.nn as nn


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes=[256, 128], dropout_rate=0.5):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_size, out_size)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs


if __name__ == "__main__":
    model = Prenet(in_dim=128, sizes=[256, 256, 128])
    import ipdb

    ipdb.set_trace()
