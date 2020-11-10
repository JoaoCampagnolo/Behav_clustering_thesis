# File:             plot_results_util.py
# Date:             March 2019
# Description:      Auxiliary functions to plot the data. Some functions are adapted to BehavPreprocess.py.
#                   Usefull to check the outputs of each module, and also to display informative results.
# Authors:          Joao Campagnolo
# Python version:   Python 3.7+

# Import packages:
import skimage
from skimage import io
from skimage.morphology import disk
from skimage.filters.rank import gradient
from skimage.feature import peak_local_max
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import euclidean_distances
from skimage import exposure
from scipy import ndimage as ndi
from skimage.morphology import watershed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib import pyplot
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import seaborn as sns
import cmasher as cmr
import cv2
import pygal
import numpy as np
import pandas as pd
import pickle
from scipy import stats
import joint_functions as jf
import math
import pywt
from scipy.signal import argrelextrema
from skimage import filters
from find_wavelets import find_wav
from behav_annotation import Behav
from preprocess import get_name_3point_angles


# sns.set(color_codes=True)


# DATA PREPROCESSING ##################################################################################################

        
def see_post_series_stat(post_series):
    '''
    Plots time series with statistical error information. Takes a ton of time nonetheless.
    '''
    A = np.asarray(post_series)
    df = using_multiindex(A, list('ZYX')) #z=subject, y=frame, x=joint, A=angle_val
    df.columns = ['Subject', 'Frame', 'Joint Index', 'Angle']
    sns.relplot(x="Frame", y="Angle", col="Joint Index", col_wrap=5, kind="line", data=df)

def using_multiindex(A, columns):
    '''
    Auxiliary function for see_post_series_stat. Takes joint angles and reshapes them into a 
    Pandas dataframe suited for sns.replot.
    '''
    shape = A.shape
    index = pd.MultiIndex.from_product([range(s)for s in shape], names=columns)
    df = pd.DataFrame({'A': A.flatten()}, index=index).reset_index()
    return df
        
        
def see_joint_angles(j_ang, j_ang1, o_ang, o_ang1, exp1, exp2):
    '''
    This function outputs the full set of joint angles that are given by joint_functions. In BehavPreprocess, 2 of the experiments are kept as control for the joint angles function, in self.ctrl_list_angles. These experiments, at this point, consist in joint angles (and other angles) calculations from the 3D positions of the leg joints, abdominal stripes and antennae. The first 2 entries correspond to leg joint angles, and the ramining 2, to other angles (head tilt and abdominal tilt).
    '''
    
    name_vector, extended_name_vector = jf.get_name_vector()
    extended_name_vector = np.append(extended_name_vector, ['L_STRIPES', 'R_STRIPES', 'ANTENNAE'])
    for i in range(np.shape(j_ang)[1]):
        fig = plt.figure()
        plt.plot(j_ang[:,i], color='k', linewidth=.7, alpha=.5, label='Control subject {}'.format(exp1))
        plt.plot(j_ang1[:,i], color='b', linewidth=.7, alpha=.5, label='Control subject {}'.format(exp2))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title(extended_name_vector[i])
        plt.ylabel('Angle (rad)')
        plt.xlabel('time')
    for i in range(np.shape(o_ang)[1]):
        fig = plt.figure()
        plt.plot(o_ang[:,i], color='r', linewidth=.7, alpha=.5, label='Control subject {}'.format(exp1))
        plt.plot(o_ang1[:,i], color='g', linewidth=.7, alpha=.5, label='Control subject {}'.format(exp2))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title(extended_name_vector[i+36])
        plt.ylabel('Angle (rad)')
        plt.xlabel('time')
        
        
def mean_center(j_ang, nj_ang, o_ang, no_ang, exp):
    '''
    This function outputs a plot for each angle, where the meancentered time-sequence is compared to itself, prior to the mean centering step. The point was just to check how meaningfull the mean centering step was at this stage. Meancentering of the time series leaves the rest angle at zero.
    '''
    
    name_vector, extended_name_vector = jf.get_name_vector()
    extended_name_vector = np.append(extended_name_vector, ['L_STRIPES', 'R_STRIPES', 'ANTENNAE'])
    for i in range(np.shape(j_ang)[1]):
        fig = plt.figure()
        figsize=(24,6)
        plt.plot(j_ang[:,i], color='k', linewidth=.7, alpha=.5, label='Not meancentered, subject {}'.format(exp))
        plt.plot(nj_ang[:,i], color='b', linewidth=.7, alpha=.5, label='Meancentered, subject {}'.format(exp))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title(extended_name_vector[i])
        plt.ylabel('Angle (rad)')
        plt.xlabel('time')
    for i in range(np.shape(o_ang)[1]):
        fig = plt.figure()
        figsize=(24,6)
        plt.plot(o_ang[:,i], color='r', linewidth=.7, alpha=.5, label='Not meancentered, subject {}'.format(exp))
        plt.plot(no_ang[:,i], color='g', linewidth=.7, alpha=.5, label='Meancentered, subject {}'.format(exp))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title(extended_name_vector[i+36])
        plt.ylabel('Angle (rad)')
        plt.xlabel('time')
        
def see_postural_series(pose_data, fly_tags, exp, joint, limb, H_angle=True):
    '''
    Plots a postural time series from a given experiment at a given joint. If H_angle=True, it reflects joint angles 
    from Halla's function, otherwise default angle computing.
    '''

    fig = plt.figure()
    if H_angle:
        name_vector, extended_name_vector = jf.get_name_vector()
        extended_name_vector = np.append(extended_name_vector, ['L_STRIPES', 'R_STRIPES', 'ANTENNAE'])
        
        limb_data = np.transpose(np.stack((pose_data[exp][:,limb*6], pose_data[exp][:,limb*6+1], pose_data[exp][:,limb*6+2],
                              pose_data[exp][:,limb*6+3], pose_data[exp][:,limb*6+4], pose_data[exp][:,limb*6+5])))
        limb_df = pd.DataFrame(limb_data, columns=extended_name_vector[limb*6:(limb+1)*6])   
        sns.lineplot(data=limb_df, palette="Set2", linewidth=1, dashes=False, alpha=.7)
        sns.despine(left=True, right=True, top=True, bottom=True)
        plt.legend(loc='upper left', bbox_to_anchor=(-0.01, -.15), fancybox=False, shadow=False, ncol=3, prop={'size': 8})
        plt.title(f'Pose data from {fly_tags[exp]}')
        plt.ylabel('Angle (rad)')
        plt.xlabel('time')

    else:
        extended_name_vector = get_name_3point_angles()
        extended_name_vector = np.append(extended_name_vector, ['L_STRIPES', 'R_STRIPES', 'ANTENNAE'])
        limb_data = np.transpose(np.stack((pose_data[exp][:,limb*5], pose_data[exp][:,limb*5+1], pose_data[exp][:,limb*5+2],
                              pose_data[exp][:,limb*5+3], pose_data[exp][:,limb*5+4])))
        limb_df = pd.DataFrame(limb_data, columns=extended_name_vector[limb*5:(limb+1)*5])
        sns.lineplot(data=limb_df, palette="Set2", linewidth=1, dashes=False, alpha=.7)
        sns.despine(left=True, right=True, top=True, bottom=True)
        plt.legend(loc='upper left', bbox_to_anchor=(-0.01, -.25), fancybox=False, shadow=False, ncol=3, prop={'size': 8})
#         plt.plot(pose_data[exp][:,joint], color='k', linewidth=.7, alpha=.5, label=f'Fly {exp}')
#         # recreate extended_name_vector for this just by removing body-coxa T2 angle.
        plt.title(f'Pose data from {fly_tags[exp]}')
        plt.ylabel('Angle (rad)')
        plt.xlabel('time')
    return

def plt_wav(angl_exp_list, test_rest_mask, fly_tags, exp, j_ang=0, num_ang=1, fps=100, chan=25, fmin=1, fmax=50, x_size=9, y_size=3, show_rest=False, show_pose=True, H_angles=False):
    '''
    Simply plots the wavelet transformation of a chosen joint angle. The only purpose it serves here is to check how the wavelets would look for any given number of channels, minimum frequency and maximum frequency (DO NOT go beyond the Nyquist frequency (=fps/2) due to aliasing).
    '''
    # Joint names
    if H_angles:
        _, extended_name_vector = jf.get_name_vector()
        extended_name_vector = np.append(extended_name_vector, ['L_STRIPES', 'R_STRIPES', 'ANTENNAE'])
    else:
        extended_name_vector = np.append(get_name_3point_angles(), ['L_STRIPES', 'R_STRIPES', 'ANTENNAE'])
    
    # Pre frame normalization
    angles_ts = angl_exp_list[exp]
    spect, f = find_wav(angles_ts[:,j_ang:j_ang+num_ang].copy(), chan=chan, omega0=5, fps=fps, fmin=fmin, fmax=fmax)
    r_mask = test_rest_mask[exp]*1
    rest = ranges(np.where(r_mask)[0])
    colors = ['white', 'lavender', 'palegreen', 'beige', 'coral', 'pink']
    plt.figure(figsize=(x_size,y_size))
    ax1 = plt.subplot(2,1,1)
    nframes = spect.shape[0]
    nfeats = spect.shape[1]
    ax1.imshow(np.transpose(spect), extent = [1, nframes, nfeats, 0], aspect='auto')
    d = angles_ts[:,j_ang:j_ang+num_ang].copy()
    if show_pose:
        for i in range(num_ang):
            d[:,i] = ((-d[:,i] + np.mean(d[:,i]))*(chan/2)/(max(d[:,i])-min(d[:,i]))) + ((num_ang-i)-1/2)*chan
        ax1.plot(np.arange(0,d.shape[0]), d, alpha=0.8, c=colors[0%len(colors)])
    if show_rest:    
        for i in range(np.shape(rest)[0]):
            ax1.axvspan(rest[i][0], rest[i][1], facecolor='0.5', alpha=0.45)
    ax1.set(title=f'Wavelet transform, {fly_tags[exp]}',
            xlabel='time', ylabel='Wavelet data (background) & joint angles (white)')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.set_yticklabels([])
    for i in range(num_ang):
        ax1.annotate(extended_name_vector[j_ang+num_ang-i-1], xy=(-5, chan*(i-1/2)+chan),  xycoords='data',
            rotation=90, size=5, va='center', horizontalalignment='right', verticalalignment='top')
    ax1.tick_params(axis='both', which='both', length=0)
    
    
    
    # Post frame normalization
    nspect = normalize_ts(spect, ax=0)
    ax2 = plt.subplot(2,1,2)
    ax2.imshow(np.transpose(nspect), extent = [1, nspect.shape[0], nspect.shape[1], 0], aspect='auto')
    if show_rest:    
        for i in range(np.shape(rest)[0]):
            ax2.axvspan(rest[i][0], rest[i][1], facecolor='0.5', alpha=0.35)
    ax2.plot(np.arange(0,d.shape[0]), d, alpha=0.01, c=colors[0%len(colors)])
    ax2.set(title=f'Frame normalization, {fly_tags[exp]}',
            xlabel='time', ylabel='Normalied wavelet data')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_yticklabels([])
    for j in range(num_ang):
        ax2.annotate(extended_name_vector[j_ang+num_ang-j], xy=(-5, chan*(j-1/2)+chan),  xycoords='data',
            rotation=90, size=5, va='center', horizontalalignment='right', verticalalignment='bottom')
    ax2.tick_params(axis='both', which='both', length=0)
    return
    
    
def plt_full_wav(r_mask, spectrogram, exp):
    '''
    Check the aspect of the current wavelets for a single experiment. The removed frames appear in gray. This allows to visualise if the low energy frame removal is removing frames appropriately.
    '''
    
    rest = ranges(np.where(r_mask)[0])
    nframes = np.shape(spectrogram)[0]
    feat_scale = np.shape(spectrogram)[1]
    fig = plt.figure(figsize=(6,12))
    ax1 = plt.imshow(np.transpose(spectrogram), extent=[1, nframes, feat_scale, 0], 
               aspect='auto')
    for i in range(np.shape(rest)[0]):
        plt.axvspan(rest[i][0], rest[i][1], facecolor='0.5', alpha=0.35)
    plt.title('Spectrogram of control subject {}'.format(exp))
    plt.ylabel('Scale x Feat')
    plt.xlabel('Time')
    plt.show
    
    
def is_bimodal(var_series, fly_tags, exp, nbins=300):
    '''
    Simply plots a histogram that depicts the low energy frame removal according to Otsu's method for a bimodal distribution. Here, the distribution matches that of the magnitudes of the wavelets for all scales and angles. 
    '''
    
    var_map = var_series[exp]
    thr = filters.threshold_otsu(var_map, nbins=nbins); 
    print('log10(variance threshold):',thr)
    print('variance threshold:',10**thr)
    
    heights, bins, _  = plt.hist(var_map, bins=nbins)
    bin_width = np.diff(bins)[0]
    bin_pos = bins[:-1] + bin_width / 2
    mask = (bin_pos >= thr)
    # plot data in two steps
    plt.bar(bin_pos[mask], heights[mask], width=bin_width)
    plt.bar(bin_pos[~mask], heights[~mask], width=bin_width, color='grey')
    
    plt.axvline(x=thr, c='r', ls='--', linewidth=1.5)
    plt.title(f'Log(Variance) histogram for {fly_tags[exp]}')
    plt.ylabel('Abs freq')
    plt.xlabel('log10(Var)')
#     plt.xlim(-.75,2.25)
    plt.show()
    
    
def normalize_ts(time_series, ax=0):
    '''
    Simply normalizes any vector by subtracting each entry by the total sum. This serves a purpose since dimensionality reduction methods, such as t-SNE, are suited for measuring distances between probability distributions. Hence, I make use of this function to get probability distributions from the magnitudes of the spectrogram, for every frame.
    '''
    
    # for shape (frame,feat)
    eps = 0.0001
    print("shapes:", np.shape(np.transpose(time_series)), np.shape(np.mean(np.transpose(time_series), axis=ax)))
    n_time_series = np.transpose(time_series) / np.sum(np.transpose(time_series), axis=ax)
    n_time_series = np.transpose(n_time_series)
    return n_time_series


def wavelet_transform(angles, fps=100, chan=25, fmin=1, fmax=50, pca_dim=None, cor=True): 
    '''
    Performs complex wavelet transformations for a time-series shape=(frame, angle). Take into account the sampling rate of the data (fps) and some parameters of choice, such as the number of channels, minimum and maximum frequencies, and the use of a correction factor for lower frequencies.
    '''
    
    nframes = angles.shape[0]
    nfeats = angles.shape[1]
    dt = 1/fps # sampling period
    pi = math.pi 
    w0 = 2
    kk = (w0 + math.sqrt(2 + w0 ** 2)) / (4 * pi * fmax)
    scales = kk * (2 ** ((np.linspace(0, chan-1, chan) / (chan-1)) * math.log(fmax / fmin, 2))) * (2.824228 * fps)
    scales = scales[0:chan]
    fs = pywt.scale2frequency('cmor1-1', scales) / dt
    print('Range of center frequencies covered for {0}fps:{1}'.format(fps, fs))
    # create correction factor array:
    x = np.arange(nframes)
    cf = np.zeros(chan)
    resp = np.zeros(shape=(nframes,chan))
    for i in range(chan):
        resp[:,i] = abs((pywt.cwt(np.exp((2*np.pi*fs[i]*(x/fps))*1j), scales[i], 'cmor1-1'))[0])
        cf[i] = max(resp[:,i])
    if cor:
        print('Correction factor:', cf)
    # Create matrix for every wavelet transform
    wav = np.zeros(shape=(nfeats, len(fs), nframes))
    for i in range(nfeats):
        for j in range(len(scales)):
            # Get correction factor:
            if cor:
                ccf = cf[j] #((pi**-1/4)/(np.sqrt(2*scales[j])))*np.exp(1/(4*(w0-np.sqrt(2+w0**2))**2)); 
            else:
                ccf = 1
            wav[i,j,:] = abs((pywt.cwt(angles[:, i], scales, 'cmor1-1', sampling_period=dt))[0])[j,:] / ccf
    wav_c = np.concatenate(wav, axis = 0)
    del wav
    wav_c = wav_c.transpose()    
    return wav_c

# DIMENSIONALITY REDUCTION ############################################################################################

def see_2D_data(data_embed, activity, fly_tags, train_exp_index, pipeline, short_tags=False, tag_end=13):
    '''
    Shows embedded 2D results from t-SNE or PCA. Also stores the data in a Pandas dataframe.
    '''
    # Plots from Matplotlib
    t = np.arange(np.shape(data_embed)[0])
    z1 = data_embed[:,0]; z2 = data_embed[:,1]
#     fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
#     axs[0].scatter(data_embed[:,0], data_embed[:,1], c=t, cmap='viridis', s=.7) # Sequence
#     plt.ylabel('z2'); plt.xlabel('z1'); plt.title('t-SNE 2D')#; plt.colorbar()

#     axs[1].scatter(data_embed[:,0], data_embed[:,1], c=np.log(activity), cmap='coolwarm', s=.7) # Sequence
#     plt.ylabel('z2'); plt.xlabel('z1'); plt.title('t-SNE 2D')#; plt.colorbar()
    
    # Pands data frame and plots from Seaborn: 
    if short_tags:
        tag_arr = [fly_tags[i][7:-4] for i in train_exp_index]
    else:
        tag_arr = [fly_tags[i] for i in train_exp_index]
    gen_arr = [fly_tags[i][7:-tag_end] for i in train_exp_index]
    data_frame_2d = pd.concat([pd.DataFrame(data_embed, columns=['$Z_1$','$Z_2$']), 
                               pd.DataFrame(activity, columns=['Frame activity']),
                               pd.DataFrame(tag_arr, columns=['Subject']),
                               pd.DataFrame(gen_arr, columns=['Genotype']),
                               pd.DataFrame(train_exp_index, columns=['Subject index'])],
                               axis=1)
    
#     # Kernel density estimation plot    
#     kde = sns.jointplot(data_frame_2d['$Z_1$'], data_frame_2d['$Z_2$'], kind="kde", height=5, space=0, color="lavender")
    
#     # Regular scatter plot
#     sc = sns.jointplot(data_frame_2d['$Z_1$'], data_frame_2d['$Z_2$'], kind="scatter", color="plum", 
#                        s=4, height=5, alpha=0.7)
    
    # Scatter plot while assigning point colors and sizes to different variables in the dataset
    n_subj = len(np.unique(np.asarray(data_frame_2d['Subject'])))
    sns_scatter("$Z_1$", "$Z_2$", "Subject", "Frame activity", data_frame_2d, pipeline,
                cpal=sns.color_palette('Spectral', n_subj, desat=.95), fsize=12, xloc=-.12)
    sns_scatter("$Z_1$", "$Z_2$", "Frame activity", "Frame activity", data_frame_2d, pipeline,
                cpal='coolwarm', fsize=8, cols=1, xloc=1.10, yloc=0.90)
    
    return data_frame_2d

def sns_scatter(x, y, hue, size, dataframe, title, cpal="Set2", fsize=10, alpha=0.2, cols=6, xloc=-0.20, yloc=-0.08):
        f, ax = plt.subplots(figsize=(fsize, fsize))
        ax.set_title(title)
        sns.despine(f, left=True, bottom=True)
        sns.scatterplot(x=x, y=y,
                        hue=hue, size=size,
                        palette=cpal,
                        sizes=(1, 70), linewidth=0.2,
                        data=dataframe, ax=ax, alpha=alpha)
        plt.legend(loc='upper left', bbox_to_anchor=(xloc, yloc), fancybox=True, shadow=False, ncol=cols)

    

# SEGMENTATION ########################################################################################################

def plt_segmentation(low_dim_train, cluster_labels, prob_dens_f, mesh, b_mask,
                     is_mesh, data_frame_2d, mesh_magnitude, post_prob, mode, xmax, ymax, pix_to_point_idx, pipeline,
                     classvtime=False):
    if is_mesh:
        # Plot embedded clusters
        borders = see_clusters(low_dim_train, mesh, cluster_labels, prob_dens_f, b_mask, is_watershed=is_mesh)
        
        # Plot clusters per genome
        watershed_genomes(data_frame_2d, mesh_magnitude, borders, b_mask, dim=xmax)
        
        # cluster Dataframe
        labels_arr, _ = reshape_mesh_labels(mesh, cluster_labels, pix_to_point_idx, xmax, ymax)
        n_labels = shuffle_labels(labels_arr)
        data_frame_cluster = pd.concat([data_frame_2d, pd.DataFrame(n_labels, columns=['Cluster ID'])],
                               axis=1)
        n_clusters = len(np.unique(np.asarray(data_frame_cluster['Cluster ID'])))
        sns_scatter("$Z_1$", "$Z_2$", "Cluster ID", None, data_frame_cluster, pipeline,
                    cpal = sns.color_palette('icefire_r', n_clusters, desat=.95),
                    fsize=7, cols=8)


    else:
        # cluster Dataframe
        data_frame_cluster = pd.concat([data_frame_2d, pd.DataFrame(cluster_labels, columns=['Cluster ID'])],
                               axis=1)
        # Plot embedded clusters
        colors = get_colors()
        n_clusters = len(np.unique(np.asarray(data_frame_cluster['Cluster ID'])))
#         cmap = cmr.redshift
        sns_scatter("$Z_1$", "$Z_2$", "Cluster ID", None, data_frame_cluster, pipeline,
                    cpal = sns.color_palette('icefire_r', n_clusters, desat=.95),
                    fsize=7, cols=8)
        
        # Plot clusters per genome
        direct_cluster_genomes(data_frame_cluster)
    
    if mode == 2:
        if classvtime == True:
            data = pd.DataFrame(post_prob, columns=range(0,np.shape(post_prob)[1]))
            dims = (24, 4)
            fig, ax = pyplot.subplots(figsize=dims)
            sns.lineplot(ax=ax, data=data, palette="Set2", dashes=False, legend=False)
        
    return data_frame_cluster
                     
def see_clusters(data_embed, mesh, cluster_labels, pdf, b_mask, is_watershed, shade=0.0001):
    '''
    Shows embedded clusters
    '''
    if is_watershed:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), sharex=True, sharey=True)
        ax = axes.ravel()
        im0 = ax[0].imshow(mesh, interpolation='nearest')
        ax[0].set_title('2D Mesh')
        fig.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(pdf, interpolation='nearest') #cmap='gray', ='magma'
        ax[1].set_title('PDF')
        fig.colorbar(im1, ax=ax[1])
        im2 = ax[2].imshow(cluster_labels, interpolation='nearest') # cmap=plt.cm.nipy_spectral
        ax[2].set_title('Watershed Labels')
        fig.colorbar(im2, ax=ax[2])
        pdf_seg, borders, borders_b = pdf_segments(pdf, cluster_labels, shade)
        im3 = ax[3].imshow(pdf_seg, interpolation='nearest')
        ax[3].set_title('PDF + Watershed boundaries')
        fig.colorbar(im3, ax=ax[3])
        img = plot_mask(pdf, b_mask, borders_b)
#         im4 = ax[4].imshow(img, interpolation='bicubic')
#         ax[4].set_title('PDF + Watershed boundaries')
#         ax[4].axis('off')
# #         ax[4].get_yaxis().labelpad = 15
# #         ax[4].set_ylabel('PDF', rotation=270)
# #         fig.colorbar(im3, ax=ax[4], cmap='viridis')
#         im5 = ax[5].imshow(np.invert(borders), interpolation='nearest')
#         ax[5].set_title('Organization of behavioral space')
        fig.text(0.5, 0.04, '$Z_1$', ha='center')
        fig.text(0.04, 0.5, '$Z_2$', va='center', rotation='vertical')
        
        # Plot just the area borders
        plt.figure(figsize=(7,7))
        plt.imshow(borders, interpolation='nearest', cmap='Greys') #border_pdf as alternative
        plt.axis('off')
        
        #  Extra Plot
        plt.figure(figsize=(7,7))
        plt.title('All Flies', fontsize=20)
        plt.imshow(img) #border_pdf as alternative
        plt.axis('off')

        
    else:
        color_palette = sns.color_palette('viridis', np.unique(cluster_labels).max() + 1)
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in cluster_labels]
        scatter = plt.scatter(data_embed[:,0], data_embed[:,1], s=20, linewidth=0, c=cluster_colors, alpha=0.5)
        plt.title('Embedded clusters')
        plt.xlabel('$Z_1$')
        plt.ylabel('$Z_2$')
        
    return borders
        
#         recs = []
#         cluster_values = np.unique(cluster_labels)
#         for i in range(0, max(cluster_labels)+1):
#             recs.append(mpatches.Circle((0, 0), 1, fc=cluster_colors[50*cluster_values[i]]))
#         print(recs)
        
#         fig, ax = plt.subplots()
#         for g in np.unique(cluster_labels):
#             ix = np.where(cluster_labels == g)
#             ax.scatter(data_embed[:,0][ix], data_embed[:,1][ix], c=cluster_colors[g], label=g, s=20, alpha=0.5)
#         ax.legend()
#         plt.show()
#         plt.legend(handles=scatter.legend_elements()[0], labels=cluster_labels,
#                     loc="lower right", title="cluster labels")


def watershed_genomes(data_frame_2d, mesh_magnitude, borders, b_mask, dim):
    for i in range(len(np.unique(np.asarray(data_frame_2d['Genotype'])))):
        genotype = np.unique(np.asarray(data_frame_2d['Genotype']))[i]
        gen_mask = pd.DataFrame([data_frame_2d['Genotype'] == genotype for key, val in data_frame_2d.items()]).T.all(axis=1)
        z1 = np.asarray(data_frame_2d[gen_mask]['$Z_1$'])
        z2 = np.asarray(data_frame_2d[gen_mask]['$Z_2$'])
        embed_data = np.array([[z1[i],z2[i]] for i in range(len(z1))])
        # Mesh
        mesh = map_mesh(embed_data, magnitude=mesh_magnitude, xmax=dim, ymax=dim)[0]
        # PDF
        pdf = gaussian_filter(mesh, sigma=7)
        pdf = exposure.adjust_gamma(pdf, gamma=.4)
        pdf = pdf / pdf.sum()
        # Watershed
        distances = ndi.distance_transform_edt(pdf)
        loc_maxima = peak_local_max(pdf, min_distance=5, threshold_rel=.01, indices=False) #
        mask = peak_local_max(pdf, min_distance=0, threshold_rel=0, indices=False)
        markers, num_class = ndi.label(loc_maxima)
        labels = watershed(-pdf, markers, mask=mask)
        
        # PDF + Watershed borders
        border_pdf, _, _ = pdf_segments(pdf, labels, base=0.1, borders=borders, import_borders=True)
        borders_b = borders.astype(bool)
        borders_b = ndimage.binary_dilation(borders_b, iterations=1)
        img = plot_mask(border_pdf, b_mask, borders_b, exp=.1)
        # Plot
        plt.figure(figsize=(7,7))
        plt.title(genotype, fontsize=20)
        plt.imshow(img) #border_pdf as alternative
        plt.axis('off')
#         plt.ylabel('$Z_2$')
#         plt.xlabel('$Z_1$')
#         plt.colorbar()
#         plt.imshow(map_mesh(embed_data, mesh_magnitude)[0])
#         plt.imshow(pdf_segments(pdf, labels)[0])

def direct_cluster_genomes(data_frame_cluster):
    num_class = data_frame_cluster['Cluster ID'].max()
    colors = get_colors()
    for i in range(len(np.unique(np.asarray(data_frame_cluster['Genotype'])))):
        genotype = np.unique(np.asarray(data_frame_cluster['Genotype']))[i]
        df = data_frame_cluster.copy()
        mask = df.Genotype.str.contains(genotype, regex=False)
        df.loc[~mask, 'Cluster ID'] = 999 # Change the value of the cluster label for subjects that arent {genotype}
#         plt.figure()
#         plt.title(genotype)
        n_clust_loc = len(np.unique(np.asarray(df['Cluster ID'])))
        sns_scatter("$Z_1$", "$Z_2$", "Cluster ID", None, df, None,
                    cpal=sns.color_palette('icefire_r',n_clust_loc), fsize=7, alpha=0.1)
        ax = plt.gca()
        ax.set_title(genotype)
    
def pdf_segments(pdf, labels, base=0.1, borders=1, import_borders=False):
    
    grad = gradient(labels, disk(1))
    edges = peak_local_max(grad, min_distance=0, threshold_rel=0, indices=False)*1
    if import_borders:
        edges = borders
    baseline = pdf.max() * base
    
    border_pdf = np.zeros(np.shape(edges))
    for i in range(np.shape(edges)[0]):
        for j in range(np.shape(edges)[1]):
            if edges[i,j] != 0:
                edges[i,j] = 1
            else:
                border_pdf[i,j] = pdf[i,j] + baseline
                     
    edges_b = edges.astype(bool)
    edges_b = ndimage.binary_dilation(edges_b, iterations=1)
    return border_pdf, edges, edges_b
    
def plot_mask(pdf_seg, b_mask, borders_b, exp=.2):
    
    norm = plt.Normalize()
    img = plt.cm.viridis(norm(pdf_seg))
    img[np.logical_not(b_mask)] = 1
    
#     pdf_seg_c = exposure.adjust_gamma(pdf_seg, exp)
#     img = cm.viridis(pdf_seg_c)
#     img[np.logical_not(b_mask)] = 1
    img[borders_b] = [0,0,0,1]
    return img

# CLUSTER ANALYSIS ####################################################################################################

def transition_matrix(arr, n=1):
    '''
    Computes the transition matrix from Markov chain sequence of order `n`, M,
    where M[i][j] is the probability of transitioning from i to j.
    :param arr: Discrete Markov chain state sequence in discrete time with states in 0, ..., N
    :param n: Transition order
    '''
    M = np.zeros(shape=(max(arr) + 1, max(arr) + 1))
    for (i, j) in zip(arr, arr[1:]):
        M[i, j] += 1
    T = (M.T / M.sum(axis=1)).T
    return np.linalg.matrix_power(T, n)

def reshape_mesh_labels(mesh, label_mesh, pix_to_point_idx, xmax, ymax):
    times = list()
    labels = list()
    count = 0
    for x in range(xmax):
        for y in range(ymax):
            if mesh[x,y] != 0:
                pix_labels = [label_mesh[x,y] for n in range(mesh[x,y].astype(int))]
                labels.append(pix_labels) # append label N times
                times.append(pix_to_point_idx[(x,y)])
    labels = np.concatenate(labels)
    times = np.concatenate(times)
    labels_arr = np.transpose([x for _,x in sorted(zip(times,labels))])
    return labels_arr, times

def plot_Markov_matx(mesh, labels, pix_to_point_idx, xmax, ymax, mode=0):
    if mode == 0:
        labels_arr, times = reshape_mesh_labels(mesh, labels, pix_to_point_idx, xmax, ymax)
        m = transition_matrix(labels_arr)
    else:
        m = transition_matrix(labels)
    m[~(m % 7).astype(bool)] = np.nan #bad values defined as NaN
    
    plt.figure()
    cmap = plt.cm.get_cmap('magma')
    new_cmap = truncate_colormap(cmap, 0.12, 1)
    new_cmap.set_bad('black',1.)
    plt.imshow(np.log10(m), cmap=new_cmap)
    plt.title('Markov transition matrix')
    plt.colorbar()   
    return m
    
def cluster_frame_dist(cluster_dict):
    clusters = list()
    totals = list()
    outliers = list()
    dwell_times = list()
    clst = 0
    for cluster in cluster_dict.keys():
        total_fr = 0
        outlier_fr = 0
        cluster_x = list()
        cluster_y = list()
        for fly in cluster_dict[cluster].keys():
    #         print(f'fly: {fly}', cluster_dict[cluster][fly])
            fly_x = list()
            fly_y = list()
            for sequence in cluster_dict[cluster][fly].keys():
    #             print(f'frame#: {len(cluster_dict[cluster][fly][sequence]["frames"])}')       
                if sequence[0:3] in ['out']: #is outlier
                    outlier_fr += len(cluster_dict[cluster][fly][sequence]['frames'])
                else: #not outlier
                    fly_x.append(cluster_dict[cluster][fly][sequence]['x'])
                    fly_y.append(cluster_dict[cluster][fly][sequence]['y'])
                total_fr += len(cluster_dict[cluster][fly][sequence]['frames'])
                dwell_times.append(len(cluster_dict[cluster][fly][sequence]['frames']))
            cluster_x.append(fly_x)
            cluster_y.append(fly_y)
    #     print(f'total frames: {total_fr}; outlier frames: {outlier_fr}; cluster: {clst}')
        clusters.append(clst)
        totals.append(total_fr)
        outliers.append(outlier_fr)
        clst += 1
    assert len(clusters) == len(totals) == len(outliers)
    frame_numbers = pd.DataFrame(np.transpose([[ int(np.round(x)) for x in clusters], totals, 
                                               np.log(totals), outliers, np.log(outliers)]), 
                                 columns = ['Cluster No', 'Total frames', 'Ln(#frames)', 'Outlier frames', 'Ln(#outliers)'])

    mean_dwell = np.mean(dwell_times)
    print(f'Mean dwell time is {mean_dwell} frames')
    cluster_entropy = stats.entropy(frame_numbers['Total frames'])
    print(f'Cluster entropy is {cluster_entropy}')
    
    return frame_numbers, mean_dwell, cluster_entropy

def plot_cluster_frame_dist(frame_numbers, dims=(20, 5)):
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=dims)

    # Plot total frames
#     sns.set_color_codes("pastel")
    sns.barplot(x='Cluster No', y='Total frames', data=frame_numbers,
                label='Total frames', color=sns.color_palette("PuOr", n_colors=12, desat=.9)[8], ax=ax)
    # Plot outlier frames
#     sns.set_color_codes("muted")
    sns.barplot(x='Cluster No', y='Outlier frames', data=frame_numbers,
                label='Outlier Frames', color=sns.color_palette("PuOr", n_colors=12, desat=.7)[3], ax=ax)
    # Add a legend and informative axis label
    ax.set_xlabel('Cluster frames')
    ax.legend(ncol=2, loc='upper right', frameon=True)
    ax.set(ylabel='Number of frames per cluster',
           xlabel='Cluster Number')
    sns.despine(left=True, bottom=True)
    
def plot_markov_hist(mtm, frame_numbers, titl, dims=(10,10)):
    
    # Initialize the matplotlib figure
    f = plt.figure(figsize=dims)   
    f.suptitle(titl, fontsize=15, y=1.02, x=0.05+3/8) # title eg.'$PCA_3$'+'$_0$'+'-$tSNE_2$'+'-$HDBSCAN$'
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], height_ratios=[1]) 
    #MARKOV TRANS. MATX.
    ax0 = plt.subplot(gs[0])
    cmap = plt.cm.get_cmap('magma')
    new_cmap = truncate_colormap(cmap, 0.12, 1)
    new_cmap.set_bad('black',1.)
    ax0.imshow(np.log10(mtm), cmap=new_cmap)
    ax0.set_ylabel('Cluster #')
    
    # HISTOGRAM, vertical
    ax1 = plt.subplot(gs[1])
    # Plot total frames
    sns.barplot(x='Ln(#frames)', y='Cluster No', data=frame_numbers,
                label='Total frames', color=sns.color_palette("PuOr", n_colors=12, desat=.9)[8], ax=ax1, orient = 'h')
    # Plot outlier frames
    sns.barplot(x='Ln(#outliers)', y='Cluster No', data=frame_numbers,
                label='Outlier Frames', color=sns.color_palette("PuOr", n_colors=12, desat=.7)[3], ax=ax1, orient = 'h')
    # Add a legend and informative axis label
    ax1.set_xlabel('Ln(# frames)')
    ax1.set(yticklabels=[])  # remove the tick labels
    ax1.set(ylabel=None)  # remove the axis label
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
#     plt.savefig('grid_figure.pdf')
    
    
    
    
    
def fly_frame_dist(data_frame_cluster, titl):
    # Find fly frames centroids:
    fly_df = data_frame_cluster.groupby(['Subject'])['$Z_1$', '$Z_2$'].mean()
    flies = np.unique(data_frame_cluster['Subject'])
    fly_centroids = [(x,y) for x, y in zip(fly_df['$Z_1$'], fly_df['$Z_2$'])]
    
    # Find fly frames uncompactness
    fly_uncomp = np.zeros(len(fly_df['$Z_1$']))
    for i in range(len(fly_df['$Z_1$'])): 
        mask = data_frame_cluster['Subject'] == flies[i]
        x = np.asarray(data_frame_cluster['$Z_1$'][mask])
        y = np.asarray(data_frame_cluster['$Z_2$'][mask])
        cent = fly_centroids[i]
        distances = [math.hypot(x[j] - cent[0], y[j] - cent[1]) for j in range(len(x))]
        fly_uncomp[i] = np.average(distances)
#         print(f'fly: {flies[i]}; centroid: {cent}; uncompactness: {fly_uncomp[i]}')
    
    # Pairwise centroid distances:    
    centroid_dists = pdist(fly_centroids)
#     print(f'Fly centroid pairwise distances {centroid_dists}')

    # Frame info with Pandas:
    fly_arrays = pd.concat([pd.DataFrame(flies, columns=['Fly']),
                        pd.DataFrame(np.asarray(fly_df['$Z_1$']), columns=['Centroid_$Z_1$']),
                        pd.DataFrame(np.asarray(fly_df['$Z_2$']), columns=['Centroid_$Z_2$']),
                        pd.DataFrame(2*np.pi*fly_uncomp**2, columns=['Uncompactness'])],
                        axis=1)
    
    # Scatter plot the centroids
    f, ax = plt.subplots(figsize=(7, 7))
    f.suptitle('Fly centroid scatter plot')
    sns.despine(f, left=True, bottom=True)
    sns.scatterplot(x='Centroid_$Z_1$', y='Centroid_$Z_2$',
                            hue='Fly', size='Uncompactness',
                            palette=sns.color_palette("Spectral", len(np.unique(np.asarray(fly_arrays['Fly'])))),
                            sizes=(1000, 5000), linewidth=0.2,
                            data=fly_arrays, ax=ax, alpha=.5)
    
    # EXTRACT CURRENT HANDLES AND LABELS
    h,l = ax.get_legend_handles_labels()
    # COLOR LEGEND (FIRST 30 ITEMS)
    col_lgd = plt.legend(h[:-6], l[:-6], loc='upper left', 
                         bbox_to_anchor=(-0.18, -0.12), fancybox=False, shadow=False, ncol=5)
#     plt.legend(loc='upper left', bbox_to_anchor=(-0.18, -0.12), fancybox=False, shadow=False, ncol=5)

    # Plot distance matrix
    g, ax2 = plt.subplots(figsize=(7, 5))
    g.suptitle(titl, fontsize=15, y=.95)
    d_mat = euclidean_distances(fly_centroids,fly_centroids)
    im = ax2.imshow(d_mat, cmap='magma_r')
    cbar = g.colorbar(im)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Centroid pairwise distances', rotation=270)

    ax2.set_yticks(np.arange(len(np.unique(np.asarray(fly_arrays['Fly'])))))
    ax2.set_yticklabels(np.unique(np.asarray(fly_arrays['Fly'])), fontsize=9)
    ax2.set_xticklabels([])
#     g.savefig('/Users/joaohenrique/Documents/EPFL/joao_pose/results/distances.png')
    
    # Embedding homogeneity
    s_dmat = np.take(d_mat,np.random.permutation(d_mat.shape[0]),axis=0)
    embed_homog = kl_divergence(d_mat+.001,s_dmat+.001)
    print(f'Fly centroid distances std dev: {embed_homog}')
    return fly_centroids, fly_uncomp, centroid_dists, fly_arrays, embed_homog
    
# STORE RESULTS #######################################################################################################   
    
def store_metrics(titl, cluster_entropy, mean_dwell, embed_homog, run_t, new_df=False):
    '''Group the metrics from the dataset, and store them with pickle. Then open and store new values...'''
    mets = [titl, cluster_entropy, mean_dwell, embed_homog, run_t]
    if new_df:
        metrics_df = pd.DataFrame(columns=['Pipeline', 'Entropy', 'Mdt', 'Homogeneity', 'Time'])
        metrics_df.loc[0] = mets
    else:
        metrics_df = pd.read_pickle('/Users/joaohenrique/Documents/EPFL/joao_pose/results' + '/metrics_df.pickle')
        metrics_df.loc[metrics_df.shape[0]] = mets
        
    # Save the dict
    with open('/Users/joaohenrique/Documents/EPFL/joao_pose/results' + '/metrics_df.pickle', 'wb') as handle:
        pickle.dump(metrics_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return metrics_df

def delete_df_line(direc, lin):
    metrics_df = pd.read_pickle(direc) #'/Users/joaohenrique/Documents/EPFL/joao_pose/results' + '/metrics_df.pickle'
    metrics_df.drop(metrics_df.index[lin], inplace=True) 
    
    # Save the dict
    with open('/Users/joaohenrique/Documents/EPFL/joao_pose/results' + '/metrics_df.pickle', 'wb') as handle:
        pickle.dump(metrics_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return metrics_df
# OTHER ###############################################################################################################   

def ranges(array):
    '''
    This function simply finds consecutive integer sequences in any array, and then returns a list with their boundaries.
    '''
    array = sorted(set(array))
    gaps = [[s, e] for s, e in zip(array, array[1:]) if s+1 < e]
    edges = iter(array[:1] + sum(gaps, []) + array[-1:])
    return list(zip(edges, edges))

def map_mesh(low_dim_data, magnitude=1, xmax=400, ymax=400):
# frame the x,y coordinates from the low dimensional data (shape is (len(z1||z2),2)).
    low_dim_data *= magnitude
    low_dim_data[:,0]+=xmax/2
    low_dim_data[:,1]+=ymax/2
    low_dim_data = np.round(low_dim_data).astype(np.int)
    mesh = np.zeros(shape=(xmax, ymax))
    point_idx_to_pix = dict()
    pix_to_point_idx = dict()
    for x in range(xmax):
        for y in range(ymax):
            pix_to_point_idx[(x,y)] = list()
    for idx, p in enumerate(low_dim_data):
        x, y = p[0], p[1]
        if x<0 or x>xmax or y<0 or y>ymax:
            continue
        mesh[x, y] += 1
        pix_to_point_idx[(x,y)].append(idx) #each entry(x,y) in this dict is the index of the frame
        point_idx_to_pix[idx] = (x,y) #each entry(frame) in this dict is the postition of the pixel

    return mesh, pix_to_point_idx, point_idx_to_pix

def reshape_mesh_labels(mesh, label_mesh, pix_to_point_idx, xmax, ymax):
    times = list()
    labels = list()
    count = 0
    for x in range(xmax):
        for y in range(ymax):
            if mesh[x,y] != 0:
                pix_labels = [label_mesh[x,y] for n in range(mesh[x,y].astype(int))]
                labels.append(pix_labels) # append label N times
                times.append(pix_to_point_idx[(x,y)])
    labels = np.concatenate(labels)
    times = np.concatenate(times)
    labels_arr = np.transpose([x for _,x in sorted(zip(times,labels))])
    return labels_arr, times

def shuffle_labels(labels):
    '''
    Randomly shuffles cluster labels, preserving cluster identities.
    '''
    no = len(np.unique(labels))
    cva = np.random.permutation(no)+1 #watershed clusters start at 1
    ncl = [np.where(cva==i)[0][0]+1 for i in labels]
    return ncl

def get_colors(num_cols=50):
    colors = ["#cd34ab","#48ca5c","#9f66ed","#78c946","#7450cd","#b2c634","#9a35b0","#60a624","#c36bea",
     "#359f34","#ea6fdc","#3cca7c","#d22c91","#86ca62","#824cbb","#91a824","#4e72eb","#d3ba37",
     "#4957bc","#e89b26","#957bea","#9bb64b","#b841a9","#479b4c","#f04ba6","#51ce9d","#e32b7d",
     "#54b475","#b55abf","#4f8429","#d58ce8","#708a20","#e683d9","#327126","#b02b7f","#a5c87b",
     "#993f8a","#b1a93f","#3f8ce6","#e1651b","#41d5e7","#e12c50","#3dbcc2","#ea5737","#57b5e3",
     "#be3220","#6dccb2","#e84173","#3ea587","#f75e97","#3c8b5b","#ef6fb5","#445a06","#aa6bc1",
     "#89a853","#794a98","#d4a847","#8b9aef","#c9751e","#707ccc","#a27f1e","#445c9f","#e88441",
     "#4e92ca","#d9643f","#2d9693","#c83b58","#115e41","#cc4579","#235e31","#d063a8","#5e7222",
     "#bb94e4","#656413","#8f72b7","#dd9549","#6b7aaf","#a24819","#b0a8e6","#415a1f","#bd4383",
     "#81b174","#ac2c54","#367042","#eb6065","#115e41","#b4323a","#368166","#e46981","#115e41",
     "#e27499","#115e41","#dd94c7","#115e41","#9b345e","#6b8b49","#b46ea0","#115e41","#ed897c",
     "#115e41","#f199aa","#115e41","#d67381","#115e41","#e3855c","#115e41","#b3576f","#115e41",
     "#c7695c","#115e41","#f1a484","#115e41","#8e4d77","#bdb26f","#7e3e53","#115e41","#a04b3b",
     "#115e41","#8f3f4a","#115e41","#c0757a","#115e41","#dfad77","#115e41","#965262","#115e41",
     "#a06625","#115e41","#c38c58","#115e41","#96643d","#115e41","#8e8335","#115e41","#765a1e",
     "#115e41","#89834d","#115e41","#5e622c","#115e41","#536c31","#115e41","#226a4d","#115e41",
     "#115e41","#115e41","#115e41","#115e41","#115e41","#115e41"]
    sns.set_palette(colors[:num_cols])
    return colors

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def kl_divergence(p, q):
    """Kullback-Leibler divergence D(P||Q) for discrete distributions
    Parameters:
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    # Add error message when p==0
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))