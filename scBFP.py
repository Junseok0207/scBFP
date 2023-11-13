import time
import torch
import copy
import math
from embedder import embedder
from tqdm import tqdm
from misc.graph_construction import construct_knn_graph

class scBFP_Trainer(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def train(self):
        cell_data = torch.Tensor(self.adata.obsm["train"])
        gene_data = cell_data.t()

        self.model = FeaturePropagation()
        self.model = self.model.to(self.device)
        st = time.time()

        # Graph Construction
        print('Start Construct adjacency')
        adj = construct_knn_graph(gene_data, self.args.gene_k, gcn_norm=True, sym=True, knnfast=self.args.knnfast, fast_batch=self.args.fb)
        adj = adj.to(self.device)

        print(f"Graph Contruction Time : {(time.time() - st):.2f}")
        pt = time.time()

        # Gene-wise Feature Propagation
        print('Start Gene-wise Feature Propagation ...!')
        gene_denoised_matrix = self.model(gene_data, adj, iter=self.args.gene_iter, mask=True)

        print(f"Hard FP Time : {(time.time() - pt):.2f}")
        pt = time.time()

        # Graph Refinement
        print('Construct new adjacency')
        denoised_matrix = gene_denoised_matrix.t()
        adj = construct_knn_graph(denoised_matrix, self.args.cell_k, gcn_norm=False, sym=True, knnfast=self.args.knnfast, fast_batch=self.args.fb)

        print(f"Graph Refinement Time : {(time.time() - pt):.2f}")
        pt = time.time()

        # Cell-wise Feature Propagation
        print('Start Final Cell-wise Feature Propagation ...!')
        denoised_matrix = self.model(denoised_matrix, adj, iter=self.args.cell_iter, mask=False)

        print(f"Soft FP Time : {(time.time() - pt):.2f}")
        et = time.time()

        print(f'Total Runing Time : {(et - st):.2f}')
        denoised_matrix = denoised_matrix.detach().cpu().numpy()
        
        self.adata.obsm['imputation'] = denoised_matrix
        return self.evaluate()



class FeaturePropagation(torch.nn.Module):
    def __init__(self):
        super(FeaturePropagation, self).__init__()

    def forward(self, x, sparse_adj, mask, iter, batch_dimension=20000, ):

        original_x = copy.copy(x)
        device = sparse_adj.device
        total_dimension = x.size(1)
        num_batches = math.ceil(total_dimension / batch_dimension)
        
        out = None
        for j in range(num_batches): 
            batch_original_x = original_x[:, batch_dimension*j : batch_dimension*(j+1)]
            batch_nonzero_idx = torch.nonzero(batch_original_x)
            batch_nonzero_i, batch_nonzero_j = batch_nonzero_idx.t()

            batch_original_x = batch_original_x.to(device)
            batch_out = batch_original_x
            for i in tqdm(range(iter)):
                batch_out = torch.sparse.mm(sparse_adj, batch_out)
                if mask:                
                    batch_out[batch_nonzero_i, batch_nonzero_j] = batch_original_x[batch_nonzero_i, batch_nonzero_j]
                
            batch_out = batch_out.to('cpu')
            if out is None:
                out = batch_out
            else:
                out = torch.cat((out, batch_out), 1)

        return out



