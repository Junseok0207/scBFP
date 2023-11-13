import numpy as np
import scipy.sparse
from sklearn import metrics
from munkres import Munkres


def drop_data(adata, rate):
    
    X = adata.X
    if scipy.sparse.issparse(X):
        X = np.array(X.todense())
    
    if rate == 0.0:
        adata.obsm['train'] = X
        adata.obsm['test'] = X

    else:
        X_train = np.copy(X)
        i, j = np.nonzero(X)

        ix = np.random.choice(range(len(i)), int(
            np.floor(rate * len(i))), replace=False)
        X_train[i[ix], j[ix]] = 0.0

        drop_index = {'i':i, 'j':j, 'ix':ix}
        adata.uns['drop_index'] = drop_index        
        adata.obsm["train"] = X_train
        adata.obsm["test"] = X

        adata.raw.X[i[ix],j[ix]] = 0.0

    return adata

def imputation_error(X_hat, X, drop_index):
    
    i, j, ix = drop_index['i'], drop_index['j'], drop_index['ix']
    
    all_index = i[ix], j[ix]
    x, y = X_hat[all_index], X[all_index]

    squared_error = (x-y)**2
    absolute_error = np.abs(x - y)

    rmse = np.mean(np.sqrt(squared_error))
    median_l1_distance = np.median(absolute_error)
    
    return rmse, median_l1_distance


def cluster_acc(y_true, y_pred):

        y_true = y_true.astype(int)
        y_true = y_true - np.min(y_true)
        l1 = list(set(y_true))
        numclass1 = len(l1)
        l2 = list(set(y_pred))
        numclass2 = len(l2)

        ind = 0
        if numclass1 != numclass2:
            for i in l1:
                if i in l2:
                    pass
                else:
                    y_pred[ind] = i
                    ind += 1

        l2 = list(set(y_pred))
        numclass2 = len(l2)

        if numclass1 != numclass2:
            print('n_cluster is not valid')
            return

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
                cost[i][j] = len(mps_d)

        m = Munkres()
        cost = cost.__neg__().tolist()
        indexes = m.compute(cost)

        new_predict = np.zeros(len(y_pred))
        for i, c in enumerate(l1):
            c2 = l2[indexes[i][1]]
            ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(y_true, new_predict)        
        f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
        f1_micro = metrics.f1_score(y_true, new_predict, average='micro')

        return acc, f1_macro, f1_micro