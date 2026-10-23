import torch
from torch import nn
from torch.nn import functional as F

SENSORS = [f'{s}_{a}' for s in ('acc', 'gyro', 'mag') for a in 'xyz']

class ModalEncoder(nn.Module):
    def __init__(self, hidden=64, dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(3, hidden, 9, padding=4), nn.GELU(),
                                 nn.Conv1d(hidden, hidden, 7, stride=2, padding=3), nn.GELU(),
                                 nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.shared = nn.Linear(hidden, dim)
        self.private = nn.Linear(hidden, dim)

    def forward(self, x):
        z = self.net(x.transpose(1, 2))
        return F.normalize(self.shared(z), dim=-1), F.normalize(self.private(z), dim=-1)


class FocalModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoders = nn.ModuleList(ModalEncoder() for _ in range(3))
        self.classifier = nn.Sequential(nn.Linear(3 * 64, 96), nn.GELU(), nn.Dropout(0.2), nn.Linear(96, num_classes))

    def embeddings(self, x):
        return [self.encoders[i](x[:, :, i * 3:i * 3 + 3]) for i in range(3)]

    def forward(self, x):
        return self.classifier(torch.cat([torch.cat((s, p), dim=-1) for s, p in self.embeddings(x)], dim=-1))


def focal_loss(model, x, tau=0.15):
    views = []
    for _ in range(2):
        noise = 0.03 * torch.randn_like(x)
        scale = 1 + 0.05 * torch.randn(x.shape[0], 1, x.shape[2], device=x.device)
        views.append(model.embeddings(x * scale + noise))
    a, b = views
    n = x.size(0)
    target = torch.arange(n, device=x.device)
    shared_loss = sum(F.cross_entropy(a[i][0] @ a[j][0].T / tau, target)
                      for i, j in [(0, 1), (0, 2), (1, 2)]) / 3
    private_loss = sum(F.cross_entropy(a[i][1] @ b[i][1].T / tau, target) for i in range(3)) / 3
    orth = sum((a[i][0] * a[i][1]).sum(-1).square().mean() for i in range(3)) / 3
    orth = orth + sum((a[i][1] * a[j][1]).sum(-1).square().mean()
                      for i, j in [(0, 1), (0, 2), (1, 2)]) / 3
    # This mini version omits the paper's temporal locality ranking loss.
    return shared_loss + private_loss + 0.5 * orth


