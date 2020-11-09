# File:             DimReduction.py
# Date:             March 2019
# Description:      Allows the user to lower the dimensioanlity of the dataset, while minimizing the loss of information 
#                   in doing so. Here, the user is free to employ a few techniques such as PCA (preserves +global structure),
#                   t-SNE (preserves +local structure) and Umap. These techniques can also be computed sequentially, following 
#                   the user's preferences. For the sake of simplicity, no more than 2 algorithms ought to be employed at a 
#                   time. The embedded data points can later be subjected to segmentation techniques to acheive discrete
#                   representations of behavior.
# Authors:          Joao Campagnolo
# Python version:   Python 3.7+

# Import packages:
import numpy as np
import time
from util import get_time

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from umap import UMAP #umap package not working properly OR outdated requirements
import seaborn as sns
import sklearn.cluster as cluster

# Improve function description

class DimReduction():
    def __init__(self, train_data, val_data, single=True, use_tsne=True, validation=False, pca_dim=2, tsne_per=150.0):
        
        self.train_data = train_data
        self.val_data = val_data
        self.validation = validation
        self.single = single
        self.use_tsne = use_tsne
        self.pca_dim = pca_dim
        
        # Define the approach and its parameters
        self.params = {'pca': {'n_components': self.pca_dim},
                       'tsne': {'perplexity':tsne_per, 'early_exaggeration':20.0,
                                'learning_rate':100, 'n_iter':2500, 'n_iter_without_progress': 300,
                                'min_grad_norm': 1e-07, 'metric':'euclidean', #or euclidean or 'precomputed' or 'cosine'
                                'init':'random', 'verbose':2, 'random_state':None, 'method':'barnes_hut', 'angle':0.3},
                       'umap': {'n_neighbors': 10, 'spread':5, 'min_dist':4, 
                                'learning_rate':10, 'metric':'euclidean'}
                      }
        
        self.models = {'pca': {'use': not self.use_tsne, 'order':1},
                       'tsne': {'use': self.use_tsne, 'order':1}
                      }
#                        'umap': {'use':False, 'order':2}
#                       }
        # t-SNE metric options: 'euclidean', 'precomputed'; method options: 'barnes_hut', 'exact';
    
        self.parameters_dict = {'parameters': self.params,
                           'models': self.models
                          }
        self.results_dict = {} # should include (x,y) coordinates, activity ...
        
        self.time_start = time.time()
        
        # Perform "single" dimensionality reduction on the dataset - user chooses one method
        if self.single:
            self.reducer_model = self.fit_single()
            self.low_dim_train = self.transform_single(self.train_data)
            if self.validation:
                self.low_dim_val = self.transform_single(self.val_data)
            if use_tsne==False:
                print(f'Total explained variance: {self.reducer_model.explained_variance_ratio_.cumsum()}')
        
        else:
            # Perform PCA to X dimensions and then use t-SNE to embedd onto 2
            self.use_dist_matx = False
            self.reducer_model = PCA(n_components=self.pca_dim)
            self.train_X_dim = self.transform_single(self.train_data)
            print(f'Total explained variance: {self.reducer_model.explained_variance_ratio_.cumsum()}')
            assert self.models['tsne']['use']
            self.reducer_model = self.fit_single()
            self.low_dim_train = self.transform_single(self.train_X_dim)
            self.reducer_tag = '$PCA_3$'+'$_0$'+'-$tSNE_2$' #TODO: Correct when X!=30
        
        self.dt = time.time() - self.time_start
        print(f'Dimensionality reduction completed. Time elapsed: {self.dt} seconds')

        
        # Functions to use
        
    def fit_single(self):
        #dim_red = DimReduction(...).fit(X_train)
        # if the user chooses the "single" dimensionality reduction technique, 
        # only one of the listed techniques are employed. The default is t-SNE
        self.use_dist_matx = False
        if self.models['tsne']['use']:
            self.reducer_tag = '$tSNE_2$'
            if self.params['tsne']['metric'] == 'precomputed':
                self.use_dist_matx = True
                self.metric = self.kl_divergence
            self.reducer_model = TSNE(**self.params['tsne'])
            print(f'Applying t-SNE with parameters {self.params["tsne"]}')
            
        if self.models['pca']['use']:
            self.reducer_tag = '$PCA_2$'
            if self.models['tsne']['use']:
                print('Cannot apply PCA. Already chose t-SNE')
            else:
                self.reducer_model = PCA(**self.params['pca'])
                print(f'Applying PCA with parameters {self.params["pca"]}')
                
#         if self.models['umap']['use']:
#             if  self.models['pca']['use'] or self.models['tsne']['use']:
#                 if self.models['tsne']['use']:
#                     print('Cannot apply Umap. Already chose t-SNE')
#                 else:
#                     print('Cannot apply Umap. Already chose PCA')
#             else:
#                 if self.params['umap']['metric'] == 'precomputed':
#                     self.use_dist_matx = True
#                     self.metric = self.kl_divergence # FIXME: find another metric more suited to Umap
#                 self.reducer_model = UMAP(**self.params['umap']) 
#                 print(f'Applying Umap with parameters {self.params["umap"]}')

        return self.reducer_model

    def transform_single(self, X):
        #X_transformed = dim_red.transform(X_test)
        if self.use_dist_matx:
            self.distance_matx = pairwise_distances(X, metric=self.metric)
            X = self.reducer_model.fit_transform(self.distance_matx)
        else:
            X = self.reducer_model.fit_transform(X)
            
        return X

    def get_distance_matrix(self, X):
        '''
        Compute the distance matrix from a vector array X.
    Valid values for metric are:
        *From scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]. These metrics support sparse matrix inputs.
        *From scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’].
        *Precomputed (e.g. Kullback-Leibler Divergence, 'kl_divergence' - slower)
        '''
        time_start = time.time()
        dist_matx = pairwise_distances(X, self.metric)
        print('Distance matrix computed. Time elapsed: {} seconds'.format(time.time()-time_start))

        return dist_matx

    def kl_divergence(self, p, q):
        """Kullback-Leibler divergence D(P||Q) for discrete distributions
        Parameters:
        p, q : array-like, dtype=float, shape=n
        Discrete probability distributions.
        """
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)

        # Add error message when p==0
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))




