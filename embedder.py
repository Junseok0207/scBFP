import os
import time
import scanpy as sc
import numpy as np

from argument import printConfig, config2string
from misc.utils import drop_data

from sklearn.cluster import KMeans
from misc.utils import imputation_error, cluster_acc
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

class embedder:
    def __init__(self, args):
        self.args = args
        printConfig(args)
        # self.config_str = config2string(args)
        self.device = f'cuda:{args.device}' if self.args.gpu else "cpu"
        self._init_dataset()
        # self.start_time = time.time()

    def _init_dataset(self):

        self.adata = sc.read(f'./dataset/{self.args.name}.h5ad')
        if self.args.eval_clustering and self.adata.obs['celltype'].dtype != int:
            self.label_encoding()

        self.preprocess(filter=self.args.filter, cell_min=self.args.cell_min, gene_min=self.args.gene_min, hvg=self.args.hvg, n_hvg=self.args.n_hvg, size_factors=self.args.sf, logtrans_input=self.args.log)
        self.adata = drop_data(self.adata, rate=self.args.drop_rate)


    def label_encoding(self):
        label_encoder = LabelEncoder()
        celltype = self.adata.obs['celltype']
        celltype = label_encoder.fit_transform(celltype)
        self.adata.obs['celltype'] = celltype

    def preprocess(self, filter, cell_min=1, gene_min=1, hvg=True, n_hvg=2000, size_factors=True, logtrans_input=True):

        if filter:
            sc.pp.filter_cells(self.adata, min_counts=cell_min)
            sc.pp.filter_genes(self.adata, min_counts=gene_min)

        if hvg:
            variance = np.array(self.adata.X.todense().var(axis=0))[0]
            hvg_gene_idx = np.argsort(variance)[-int(n_hvg):]
            self.adata = self.adata[:,hvg_gene_idx]

        self.adata.raw = self.adata.copy()        
        if size_factors:
            sc.pp.normalize_per_cell(self.adata)
            self.adata.obs['size_factors'] = self.adata.obs.n_counts / np.median(self.adata.obs.n_counts)
        else:
            self.adata.obs['size_factors'] = 1.0

        if logtrans_input:
            sc.pp.log1p(self.adata)


    def evaluate(self):
        
        if self.args.drop_rate != 0.0:
            X_test = self.adata.obsm["test"]
            drop_index = self.adata.uns['drop_index']
            rmse, median_l1_distance = imputation_error(X_test, self.adata.obsm['imputation'], drop_index)

        # clustering
        X_imputed = self.adata.obsm['imputation']
        if self.args.eval_clustering:
            celltype = self.adata.obs['celltype'].values
            n_cluster = np.unique(celltype).shape[0]

            ### Imputed
            kmeans = KMeans(n_cluster, n_init=20) #, random_state=0) #, random_state=self.args.seed)
            y_pred = kmeans.fit_predict(X_imputed)

            imputed_ari = adjusted_rand_score(celltype, y_pred)
            imputed_nmi = normalized_mutual_info_score(celltype, y_pred)
            imputed_ca, imputed_ma_f1, imputed_mi_f1 = cluster_acc(celltype, y_pred)

        print(f" ==================== Dataset: {self.args.name} ==================== ")
        if self.args.drop_rate != 0.0:
            print("Drop Rate {} -> RMSE : {:.4f} / Median L1 Dist : {:.4f}\n".format(self.args.drop_rate, rmse, median_l1_distance))

        if self.args.eval_clustering:
            print("Imputed --> ARI : {:.4f} / NMI : {:.4f} / CA : {:.4f}\n".format(imputed_ari, imputed_nmi, imputed_ca))


            

