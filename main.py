import numpy as np
import scipy.io as sio
import os
import os.path as osp
import torch
import torch.nn as nn
import argparse
from scipy.io import loadmat
from utils import load_dataset
import random

parser = argparse.ArgumentParser(description='Graph Generative causal explanations')
parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
parser.add_argument("--data_split_ratio", type=float, default= [0.8, 0.1, 0.1], help="data seperation")
parser.add_argument("--latent_dim", type=int, default=  [128, 128, 128], help="classifier hidden dims")
parser.add_argument('--readout', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over nodes in a graph: sum or average')
parser.add_argument('--dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
parser.add_argument("--mlp_hidden", type=int, default=  [128, 128], help="mlp hidden dims")
parser.add_argument("--emb_normlize", type = bool, default=  False, help="mlp hidden dims")
parser.add_argument("--GVAE_hidden_dim", type = int, default= 64, help="mlp hidden dims")
parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 100)')
parser.add_argument('--Nalpha', type=int, default=56)
parser.add_argument('--Nbeta', type=int, default=100)
parser.add_argument('--lam', type=float, default=0.05)
parser.add_argument("--weight_decay", type=float, default=0.0005, help="Adam weight decay. Default is 5*10^-5.")
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# --- load data ---
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import BAShapes
from dgl.data import BA2MotifDataset
from utils import get_dataloader
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
dataset = TUDataset(path, name='MUTAG')
#dataset = TUDataset(path, name='PROTEINS')
#dataset = TUDataset(path, name='NCI1')
#dataset = TUDataset(path, name='ENZYMES')
#dataset = TUDataset(path, name='BAShapes')
# graph = BA2MotifDataset()
# dataset = BAnotif_load_dataset(graph)
# graph_filename = "data/SCH_graph.mat"
# graph = loadmat(graph_filename)
# dataset = load_dataset(graph)
dataloader = get_dataloader(dataset, args.batch_size, data_split_ratio=args.data_split_ratio, seed=args.seed)

# --- load classifier ---
from GIN_classifier import GINNet
# from GCN_classifier import GCNNet
# from GAT_classifier import GATNet
input_dim = 7
# input_dim = 116
#input_dim = dataset.num_features
output_dim = 2
#output_dim = dataset.num_classes
print(input_dim)
print(output_dim)
classifier = GINNet(input_dim, output_dim, args, device).to(device)


# --- train/load GCE ---
from GraphVAE import GraphEncoder,GraphDecoder
from GCE import GenerativeCausalExplainer
encoder = GraphEncoder(input_dim, args.GVAE_hidden_dim, device).to(device)
decoder = GraphDecoder(args.GVAE_hidden_dim,input_dim).to(device)
for m in range(0,3):
    casual_decoder = GraphDecoder(int(args.Nalpha),input_dim).to(device)
    gce = GenerativeCausalExplainer(classifier, decoder, encoder,casual_decoder, device)
    traininfo = gce.train(dataloader['train'],dataloader['eval'],dataloader['test'],
                    steps=args.epochs,
                    Nalpha=int(args.Nalpha),
                    lam=args.lam,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    ce = 0.01,
                    r = 0.001)
    dataloader = get_dataloader(dataset, args.batch_size, data_split_ratio=args.data_split_ratio, seed=random.randint(0,1000))
    classifier = GINNet(input_dim, output_dim, args, device).to(device)
    encoder = GraphEncoder(input_dim, args.GVAE_hidden_dim, device).to(device)
    decoder = GraphDecoder(args.GVAE_hidden_dim,input_dim).to(device)
    casual_decoder = GraphDecoder(int(args.Nalpha),input_dim).to(device)
    gce = GenerativeCausalExplainer(classifier, decoder, encoder,casual_decoder, device)

