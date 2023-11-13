import torch
import numpy as np
import torch.nn.functional as F
from torch_sparse import SparseTensor

def construct_knn_graph(embeddings, k, gcn_norm=False, sym=True, knnfast=True, fast_batch=1000):
    
    if knnfast:
        adj = knn_fast(embeddings, k, fast_batch, gcn_norm=gcn_norm, sym=sym)
    else:
        adj = sparse_knn_graph(embeddings, k, gcn_norm=gcn_norm, sym=sym)

    return adj


def knn_fast(X, k, b, gcn_norm, sym):
    device = X.device
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).to(device)
    rows = torch.zeros(X.shape[0] * (k + 1)).to(device)
    cols = torch.zeros(X.shape[0] * (k + 1)).to(device)
    norm_row = torch.zeros(X.shape[0]).to(device)
    norm_col = torch.zeros(X.shape[0]).to(device)
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)

        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()

    # post-processing
    sparse_adj = SparseTensor(row=rows, col=cols, value=values).to(device)
    cur_adj = sparse_post_processing(sparse_adj, gcn_norm=gcn_norm, sym=sym)
    cur_adj = cur_adj.to_torch_sparse_coo_tensor()
    cur_adj = cur_adj.float()

    return cur_adj


def sparse_post_processing(cur_raw_adj, add_self_loop=True, sym=True, gcn_norm=False):

    from torch_sparse import fill_diag, sum as sparsesum, mul

    if add_self_loop:
        cur_raw_adj = fill_diag(cur_raw_adj, 2)
    
    if sym:
        cur_raw_adj = cur_raw_adj + cur_raw_adj.t()        
        cur_raw_adj = mul(cur_raw_adj,(torch.ones(cur_raw_adj.size(0), device=cur_raw_adj.device())*1/2).view(-1,1))

    deg = sparsesum(cur_raw_adj, dim=1)
    if gcn_norm:
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        cur_adj = mul(cur_raw_adj, deg_inv_sqrt.view(-1,1))
        cur_adj = mul(cur_adj, deg_inv_sqrt.view(1,-1))
    else:
        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        cur_adj = mul(cur_raw_adj, deg_inv_sqrt.view(-1,1))
    
    return cur_adj


def sparse_knn_graph(embeddings, k, gcn_norm=False, sym=True):

    import faiss
    device = embeddings.device
    n_nodes = embeddings.shape[0]

    embeddings = F.normalize(embeddings, dim=1, p=2)
    embeddings = embeddings.detach().to('cpu').numpy()

    index = faiss.IndexFlatIP(embeddings.shape[1])
    embeddings = np.ascontiguousarray(embeddings)
    index.add(embeddings)
    affinity, I = index.search(embeddings, k)
    idx = torch.stack([torch.arange(n_nodes).unsqueeze(1).repeat(1, k).view(-1), torch.tensor(I).view(-1)])
    affinity = torch.tensor(affinity.reshape(-1))

    # post-processing
    sparse_adj = SparseTensor(row=idx[0], col=idx[1], value=affinity).to(device)
    cur_adj = sparse_post_processing(sparse_adj, gcn_norm=gcn_norm, sym=sym)
    cur_adj = cur_adj.to_torch_sparse_coo_tensor()
    cur_adj = cur_adj.float()

    return cur_adj