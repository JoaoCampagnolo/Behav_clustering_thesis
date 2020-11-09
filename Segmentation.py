# File:             Segmentation.py
# Date:             March 2019
# Description:      Having embedded the data into a 2-dimension space, the user is faced with discretization.
#                   This will be acheived through the creation of a grid, which will be sustain the positions of
#                   data points and where a segmentation technique will be employed.
# Authors:          Joao Campagnolo
# Python version:   Python 3.7+

# Import packages:
import numpy as np
import time
from util import get_time
from scipy.ndimage.filters import gaussian_filter
from sklearn import mixture
from skimage import exposure
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk
from skimage.filters.rank import gradient
from sklearn.decomposition import PCA
import random
import hdbscan


# Improve function description

class Segmentation():
    def __init__(self, low_dim_data, high_dim_data, mode=0, mesh_mag=1, xmax=600, ymax=600, 
                 wat_ksize=6, wat_mdist=5,
                 gmm_comps=75,
                 hdbscan_cut=6, hdbscan_min=5, n_comp=125, hdbscan_hdim=False):
        
        # Define the approach and its parameters
        self.params = {'mesh': {'magnitude': mesh_mag, 'xmax': xmax, 'ymax': ymax, 'margin': 10},
                       
                       '2D_Gauss_Conv': {'function': {'order':0, 'output':None, 'mode':'nearest', 'cval':0.0, 'truncate':4.0},
                                         'other': {'kernel_size':wat_ksize, 'gamma':0.4}},
                       
                       'Watershed': {'local_maxima': {'min_distance':wat_mdist, 'threshold_rel':.001 , 'indices':False}, 
                                     'mask': {'min_distance':0, 'threshold_rel':0 , 'indices':False}},
                       
                       'HDBSCAN': {'min_cluster_size':hdbscan_min, 'min_samples':10, 'metric':'euclidean', 'alpha':1.0,
                                   'leaf_size':40, 'cluster_selection_method':'eom', 'p':None},
                       
                       'single_linkage_tree': {'cut_distance': hdbscan_cut, 'min_cluster_size': hdbscan_min},
                       
                       'GMM': {'n_components': gmm_comps, 'covariance_type': 'full'}
                      }
        
        # Count run time
        self.time_start = time.time()
        
        self.low_dim_data = low_dim_data
        self.mode = mode
        if hdbscan_hdim:
            pca = PCA(n_components=n_comp)
            self.low_dim_data = pca.fit_transform(high_dim_data)
            print(f'Percentage of explained variance: {pca.explained_variance_ratio_.cumsum()[-1]} with {n_comp} principal components')
            self.mode = 1
        self.mesh_mag = mesh_mag
        self.post_proba = list()
        
        self.mesh = np.zeros(shape=(xmax,ymax))
        self.prob_dens_f = np.zeros(shape=(xmax,ymax))
        self.point_idx_to_pix = list()
        self.pix_to_point_idx = list()
        self.xmax = 1
        self.ymax = 1
        self.mask = np.ones(shape=(xmax,ymax))
        
        self.kernel_size = self.params['2D_Gauss_Conv']['other']['kernel_size']
        self.gamma = self.params['2D_Gauss_Conv']['other']['gamma']
                
        # First approach: Gaussian 2D convolution and then Watershed
        if self.mode == 0:
            self.seg_tag = '-$Watershed$'
            # Build mesh:
            self.low_dim_data *= self.params['mesh']['magnitude']
            self.low_dim_data[:,0]+=self.params['mesh']['xmax']/2
            self.low_dim_data[:,1]+=self.params['mesh']['ymax']/2
            self.low_dim_data = np.round(self.low_dim_data).astype(np.int)
            self.xmax, self.ymax = self.params['mesh']['xmax'], self.params['mesh']['ymax']
            self.mesh, self.pix_to_point_idx, self.point_idx_to_pix = self.map_mesh(self.low_dim_data, self.xmax, self.ymax)
            
            # Apply 2D Gaussian convolution:
            self.prob_dens_f = gaussian_filter(self.mesh, self.kernel_size, **self.params['2D_Gauss_Conv']['function'])
            self.prob_dens_f = exposure.adjust_gamma(self.prob_dens_f, self.gamma)
            self.prob_dens_f = self.prob_dens_f / self.prob_dens_f.sum()
            
            # Apply Watershed:
            print('Performing Watershed on the 2D PDF mesh')
            self.distances = ndi.distance_transform_edt(self.prob_dens_f)
            self.loc_maxima = peak_local_max(self.prob_dens_f, **self.params['Watershed']['local_maxima']) #
            self.mask = peak_local_max(self.prob_dens_f, **self.params['Watershed']['mask'])
            self.markers, self.num_class = ndi.label(self.loc_maxima)
            self.labels = watershed(-self.prob_dens_f, self.markers, mask=self.mask)
            print(f'Number of clusters: {np.ndarray.max(self.labels)}')
            self.is_mesh_labels = True

        # Second approach: HDBSCAN
        elif self.mode == 1:
            if hdbscan_hdim:
                self.seg_tag = '_-$HDBSCAN$'
                print('Performing HDBSCAN in high-dimensional posture-dynamics space')
            else:
                print('Performing HDBSCAN in the 2D embedded posture-dynamics space')
                self.seg_tag = '-$HDBSCAN$'
            self.classifier = hdbscan.HDBSCAN(**self.params['HDBSCAN'])
            self.model_fit = self.classifier.fit(self.low_dim_data)
            self.post_proba = self.classifier.probabilities_
#             self.labels = self.model_fit.labels_
            self.labels = self.classifier.single_linkage_tree_.get_clusters(**self.params['single_linkage_tree'])
            print(f'Number of clusters: {max(self.labels)+1}')
                #.plot(select_clusters=True,selection_palette=sns.color_palette('deep', 8))
            self.single_linkage_tree = self.model_fit.single_linkage_tree_#.plot()
            #self.min_span_tree = self.model_fit.minimum_spanning_tree_#.plot()
            self.cond_tree = self.model_fit.condensed_tree_#.plot()
            self.hdbscan_scores = self.model_fit.outlier_scores_
            self.is_mesh_labels = False
            
        # Third approach: GMM + posterior probability assignment
        elif self.mode == 2:
            self.seg_tag = '-$GMM$'
            print('Performing GMM on the 2D embedded posture-dynamics space')
            self.classifier = mixture.GaussianMixture(**self.params['GMM'])
            self.model_fit = self.classifier.fit(self.low_dim_data)
            self.labels = self.classifier.predict(self.low_dim_data)
            print(f'Number of clusters: {max(self.labels)+1}')
            self.post_proba = self.classifier.predict_proba(self.low_dim_data)
            self.score = self.classifier.score(self.low_dim_data)
            self.bic = self.classifier.bic(self.low_dim_data)
            self.aic = self.classifier.aic(self.low_dim_data)
            self.is_mesh_labels = False
            
        # Random labels:
        elif self.mode == 3:
            self.seg_tag = '-$Random$'
            print(f'Assigning {gmm_comps} random labels')
            self.labels = np.asarray([random.randint(0,gmm_comps-1) for _ in range(np.shape(low_dim_data)[0])])
            self.is_mesh_labels = False

        # Invalid mode
        else:
            print('Invalid mode value: mode=0,1,2 for Gaussian convolution kernel + Watershed, HDBSCAN and GMM, respectively ')
        
        # Elapsed time:
        self.dt = time.time()-self.time_start
        print(f'Segmentation completed. Time elapsed: {self.dt} seconds')
        
        # Functions to use:
    def map_mesh(self, low_dim_data, xmax, ymax):
        # frame the x,y coordinates from the low dimensional data.
        self.mesh = np.zeros(shape=(xmax, ymax))
        self.point_idx_to_pix = dict()
        self.pix_to_point_idx = dict()
        for x in range(xmax):
            for y in range(ymax):
                self.pix_to_point_idx[(x,y)] = list()
        for idx, p in enumerate(low_dim_data):
            x, y = p[0], p[1]
            if x<0 or x>xmax or y<0 or y>ymax:
                continue
            self.mesh[x, y] += 1
            self.pix_to_point_idx[(x,y)].append(idx) #each entry(x,y) in this dict is the index of the frame
            self.point_idx_to_pix[idx] = (x,y) #each entry(frame) in this dict is the postition of the pixel

        return self.mesh, self.pix_to_point_idx, self.point_idx_to_pix
    
    
    