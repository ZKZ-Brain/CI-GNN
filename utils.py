from tkinter.tix import TCL_TIMER_EVENTS
import torch
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np
from scipy.spatial.distance import pdist, squareform
from torch_geometric.utils import dense_to_sparse
import math

eps = 1e-8
def load_dataset(graph):
    print('loading data')
    num_graphs=graph["label"].size
    label1=graph["label"]
    label=np.append(label1,label1)
    data_list = []
    for i in range(num_graphs):
        node_features = torch.FloatTensor(graph["graph_struct"][0][i][1])
        #node_features = (node_features - torch.mean(node_features))/(torch.std(node_features)+eps)
        tepk = node_features.reshape(-1,1)
        tepk, indices = torch.sort(abs(tepk), dim=0, descending=True)
        mk = tepk[int(math.pow(node_features.shape[0],2) / 20*2)]
        adj = torch.Tensor(np.where(node_features > mk, 1, 0))
        data_example = Data(x=node_features,edge_index=dense_to_sparse(adj)[0],y=label[i])
        data_list.append(data_example)

    return data_list

def BAnotif_load_dataset(graph):
    print('loading data')
    data_list = []
    for i in range(len(graph)):
        g,label = graph[i]
        adj = g.adjacency_matrix(transpose=True)._indices()
        node_features = g.ndata['feat']
        #node_features = (node_features - torch.mean(node_features))/(torch.std(node_features)+eps)
        data_example = Data(x=node_features,edge_index=adj,y=label[0])
        data_list.append(data_example)

    return data_list

def get_dataloader(dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=None):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    if not random_split_flag and hasattr(dataset, 'supplement'):
        assert 'split_indices' in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement['split_indices']
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                         generator=torch.Generator().manual_seed(seed))
    num_eval = len(eval)
    num_test = len(test)
    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['eval'] = DataLoader(eval, batch_size=num_eval, shuffle=False)
    dataloader['test'] = DataLoader(test, batch_size=num_test, shuffle=False)
    return dataloader

def pairwise_distances(x):
    #x should be two dimensional
    if x.dim()==1:
        x = x.unsqueeze(1)
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def calculate_sigma(Z_numpy):   

    if Z_numpy.dim()==1:
        Z_numpy = Z_numpy.unsqueeze(1)
    Z_numpy = Z_numpy.cpu().detach().numpy()
    #print(Z_numpy.shape)
    k = squareform(pdist(Z_numpy, 'euclidean'))       # Calculate Euclidiean distance between all samples.
    sigma = np.mean(np.mean(np.sort(k[:, :10], 1))) 
    if sigma < 0.1:
        sigma = 0.1
    return sigma 

# def calculate_sigma(Z, dim=2, p=2):
#     dist_matrix = torch.norm(Z[:, None]-Z, dim, p)
#     sigma = torch.mean(torch.sort(dist_matrix[:, :10], 1)[0])
#     if sigma < 0.1:
#         sigma = 0.1
#     return sigma

def calculate_gram_mat(x, sigma):
    dist= pairwise_distances(x)
    #dist = dist/torch.max(dist)
    return torch.exp(-dist /sigma)

def reyi_entropy(x,sigma):
    alpha = 1.01
    k = calculate_gram_mat(x,sigma)
    k = k/(torch.trace(k)+eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x,y,s_x,s_y):
    alpha = 1.01
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    k = torch.mul(x,y)
    k = k/(torch.trace(k)+eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy

def joint_entropy3(x,y,z,s_x,s_y,s_z):
    alpha = 1.01
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    z = calculate_gram_mat(z,s_z)
    k = torch.mul(x,y)
    k = torch.mul(k,z)
    k = k/(torch.trace(k)+eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy


def calculate_conditional_MI(x,y,z):

    s_x = calculate_sigma(x)**2
    s_y = calculate_sigma(y)**2
    s_z = calculate_sigma(z)**2
    Hyz = joint_entropy(y,z,s_y,s_z)
    Hxz = joint_entropy(x,z,s_x,s_z)
    Hz = reyi_entropy(z,sigma=s_z)
    Hxyz = joint_entropy3(x,y,z,s_x,s_y,s_z)
    CI = Hyz + Hxz - Hz - Hxyz
    
    return CI

def calculate_MI(x,y):

    s_x = calculate_sigma(x)
    s_y = calculate_sigma(y)
    Hx = reyi_entropy(x,s_x**2)
    Hy = reyi_entropy(y,s_y**2)
    Hxy = joint_entropy(x,y,s_x**2,s_y**2)
    Ixy = Hx + Hy - Hxy
    
    return Ixy

def calculate_single_TC(x):
    num_feature = x.size(1)
    HC = 0.0
    for i in range(num_feature):
        sigma = calculate_sigma(x[:,i])
        HC = HC + reyi_entropy(x[:,i],sigma)
    sigma = calculate_sigma(x)
    Hx= reyi_entropy(x,sigma**2)
    TC = HC - Hx
    return TC

def calculate_Condition_TC(x,y):
    num_feature = x.size(1)
    HC = 0.0
    sigmay = calculate_sigma(y)**2
    for i in range(num_feature):
        sigmax = calculate_sigma(x[:,i])**2
        HC = HC + joint_entropy(x[:,i],y,sigmax,sigmay) - reyi_entropy(y,sigmay)

    sigmax = calculate_sigma(x)**2
    Hxy = joint_entropy(x,y,sigmax,sigmay) - reyi_entropy(y,sigmay)
    TC = HC - Hxy
    return TC

def calculate_TC(x,y):
    TCx = calculate_single_TC(x)
    TCxy = calculate_Condition_TC(x,y)
    return TCx - TCxy

def MI_Est(discriminator, embeddings, positive, batch_size):

    shuffle_embeddings = embeddings[torch.randperm(batch_size)]
    joint = discriminator(embeddings,positive)
    margin = discriminator(shuffle_embeddings,positive)
    mi_est = torch.mean(joint) - torch.log(torch.mean(torch.exp(margin)))

    return mi_est

def separate_data(dataset,data_split_ratio=None, seed=None):
    
    num_train = int(data_split_ratio[0] * len(dataset))
    num_eval = int(data_split_ratio[1] * len(dataset))
    num_test = len(dataset) - num_train - num_eval

    train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                        generator=torch.Generator().manual_seed(seed))

    return train, eval, test

def separate_site_ASD(dataset,site,site_idx, seed=None):
    
    site = torch.LongTensor(site)
    test_idx = torch.nonzero(site == site_idx)[:,0].numpy().tolist()
    train_idx = torch.nonzero(site != site_idx)[:,0].numpy().tolist()

    train = Subset(dataset, train_idx)
    test = Subset(dataset, test_idx)

    return train, test

def separate_site_MDD(dataset,site, site_idx, seed=None):
    
    site_name = [1, 2, 4, 7, 8, 9, 10, 11, 13, 14, 15, 17, 3, 5, 6, 12, 16]
    site = torch.LongTensor(site)
    print(site_name[site_idx])
    test_idx = torch.nonzero(site == site_name[site_idx])[:,0].numpy().tolist()
    train_idx = torch.nonzero(site != site_name[site_idx])[:,0].numpy().tolist()
    print(test_idx)
    train = Subset(dataset, train_idx)
    test = Subset(dataset, test_idx)

    return train, test

def get_dataloader_site(dataset, site, site_idx,batch_size):
    
    site = torch.LongTensor(site)
    test_idx = torch.nonzero(site == site_idx)[:,0].numpy().tolist()
    train_idx = torch.nonzero(site != site_idx)[:,0].numpy().tolist()

    train = Subset(dataset, train_idx)
    test = Subset(dataset, test_idx)
    print(test_idx)

    dataloader = dict()
    test_batch_size = len(test_idx)
    print(test_batch_size)
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['test'] = DataLoader(test, batch_size=test_batch_size, shuffle=False)
    return dataloader

def get_dataloader_site_MDD(dataset, site, site_idx,batch_size):
    
    site_name = [1, 2, 4, 7, 8, 9, 10, 11, 13, 14, 15, 17, 3, 5, 6, 12, 16]
    site = torch.LongTensor(site)
    print(site_name[site_idx])
    test_idx = torch.nonzero(site == site_name[site_idx])[:,0].numpy().tolist()
    train_idx = torch.nonzero(site != site_name[site_idx])[:,0].numpy().tolist()
    print(test_idx)
    train = Subset(dataset, train_idx)
    test = Subset(dataset, test_idx)
    test_batch_size = len(test_idx)
    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['test'] = DataLoader(test, batch_size=test_batch_size, shuffle=False)
    return dataloader