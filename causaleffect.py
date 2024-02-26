import numpy as np
import torch
import torch.nn as nn
from utils import calculate_conditional_MI
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool

def clear_masks(model):
    """ clear the edge weights to None """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None


def set_masks(model, edgemask):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = edgemask

def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    ret_readout = []
    for k, v in readout_func_dict.items():
        if k in readout.lower():
            ret_readout.append(v)
    return ret_readout

def joint_uncond(alpha,beta, data, casual_decoder, classifier, device,k):

    # compute casual effect
    edge_index, batch,x = data.edge_index, data.batch,data.x
    readout_layers = get_readout_layers("mean")
    pooled = []
    for readout in readout_layers:
        pooled.append(readout(alpha.to(device), batch.to(device)))
    graph_alpha = torch.cat(pooled, dim=-1)

    pooled = []
    for readout in readout_layers:
        pooled.append(readout(beta.to(device), batch.to(device)))
    graph_beta = torch.cat(pooled, dim=-1)
    labels = torch.LongTensor(data.y).to(device)

    ax, aindex = casual_decoder(alpha.to(device))
    #aindex = get_retain_mask(drop_probs=aindex, shape=aindex.shape, tau=tau, device = device)
    aedge = aindex[edge_index[0], edge_index[1]].to(device)
    aedge = torch.sigmoid(aedge)
    # edge_mask = get_retain_mask(drop_probs=aedge, shape=aedge.shape, device = device)
    clear_masks(classifier)
    set_masks(classifier, aedge)
    logits,graph_emb,node_emb = classifier(x.to(device),edge_index.to(device),batch.to(device))
    CausalEffect = calculate_conditional_MI(graph_alpha, labels.float(), graph_beta)
    # CausalEffect = calculate_conditional_MI(graph_alpha, logits, graph_beta)

    # compute casual effect
    # if k>=50:
    #     #tau = torch.nn.Parameter(torch.tensor(1.))
    #     ax, aindex = casual_decoder(alpha.to(device))
    #     #aindex = get_retain_mask(drop_probs=aindex, shape=aindex.shape, tau=tau, device = device)
    #     aedge = aindex[edge_index[0], edge_index[1]].to(device)
    #     aedge = torch.sigmoid(aedge)
    #     #edge_mask = get_retain_mask(drop_probs=aedge, shape=aedge.shape, device = device)
    #     clear_masks(classifier)
    #     # set_masks(classifier, aedge)
    #     logits,graph_emb,node_emb = classifier(x.to(device),edge_index.to(device),batch.to(device),aedge)
    #     return CausalEffect,logits
    # else:
    return CausalEffect,logits

def get_retain_mask(drop_probs, shape, device):
    tau = 1.0
    uni = torch.rand(shape).to(device)
    eps = torch.tensor(1e-8).to(device)
    tem = (torch.log(drop_probs + eps) - torch.log(1 - drop_probs + eps) + torch.log(uni + eps) - torch.log(
        1.0 - uni + eps))
    mask = 1.0 - torch.sigmoid(tem / tau)
    return mask