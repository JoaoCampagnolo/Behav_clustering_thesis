# Import packages:
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import time
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist
import hdbscan
import seaborn as sns
import sklearn.cluster as cluster
from util import get_time





def _get_distance_matrix_(X, metric='euclidean'):
    '''
    Compute the distance matrix from a vector array X.
Valid values for metric are:
    *From scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]. These metrics support sparse matrix inputs.
    *From scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’].
    *Precomputed (e.g. Kullback-Leibler Divergence, kld)
    '''
    time_start = time.time()
    dist_matx = pairwise_distances(X)
    print('Distance matrix computed. Time elapsed: {} seconds'.format(time.time()-time_start))
    return dist_matx

def _apply_PCA_(train_data, num_components=70):
    '''
    Fit PCA to reduce dimension. This might mattern in case GMM is used. For t-SNE, it serves no purpose since t-SNE 
conserves local structure, as oposed to PCA, which conserves global structure.
    '''
    print("Applying pca")
    pca = PCA(n_components=num_components)
    pca_data = pca.fit_transform(train_data)
    print(f"PCA data shape: {np.shape(pca_data)}")
    print("Total explained variance:", np.cumsum(pca.explained_variance_ratio_)[num_components-1])
    return pca_data

def apply_tsne(dist_matx, per=50, ee=20, lr=100, n=3500, met='precomputed'):
    '''
    Fit t-SNE to reduce dimension. 
    NB. With t-SNE, it might be of interest to check the fitting for different parameters and metrics since the performance varies a lot from dataset to dataset.
    The accepted metrics are those accepted by sklearn.pairwise_distances, or precomputed metrics/divergences... (such as KLD). KLD is suited to assess distances between probability distributions, unlike the Euclidian metric.
    '''
    time_start = time.time()
    tsne = TSNE(n_components=2, perplexity=per, early_exaggeration=ee,
                          learning_rate=lr, n_iter=n, n_iter_without_progress=300, 
                          min_grad_norm=1e-07, metric=met, init='random', verbose=2, 
                          random_state=None, method='barnes_hut', angle=0.3)
    tsne_results = tsne.fit_transform(dist_matx)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    print('t-SNE train data shape:{}'.format(np.shape(tsne_results)))
    return tsne_results

def _kld_(p, q):
    """Kullback-Leibler divergence D(P||Q) for discrete distributions
    Parameters:
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    
    # Add error message when p==0
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))