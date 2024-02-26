import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

EPS = 1e-15
MAX_LOGSTD = 10

class GraphEncoder(torch.nn.Module):
 
    def __init__(self,in_channels, out_channels, device):
        super(GraphEncoder, self).__init__()

        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)
        self.device = device

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

    def reparametrize(self, mu, logstd):
        return mu + torch.randn_like(logstd) * torch.exp(logstd)

    def forward(self, data):
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        mu, logvar = self.encode(x,edge_index)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar


class GraphDecoder(torch.nn.Module):
 
    def __init__(self,out_channels,in_channels):
        super(GraphDecoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(out_channels,16),
            nn.ReLU(),
            nn.Linear(16,in_channels)
        )

    def forward(self, z):
        adj = torch.matmul(z, z.t())
        x = self.MLP(z)
        return x, torch.sigmoid(adj)


def VAE_LL_loss(Xbatch,Xest,logvar,mu, device):

    batch_size = Xbatch.shape[0]
    sse_loss = torch.nn.MSELoss(reduction = 'sum') # sum of squared errors
    Xbatch = Xbatch.to(device)
    mse = 1./batch_size * sse_loss(Xest,Xbatch)
    KLD = -0.5 * torch.mean(
            torch.sum(1 + 2 * logvar - mu**2 - logvar.exp()**2, dim=1))
    auto_loss = mse + KLD
    return auto_loss