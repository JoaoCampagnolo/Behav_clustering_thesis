# File:             preprocess.py
# Date:             Winter 2018-2019
# Description:      Some outils that are usefull in the data preprocessing stage. 
# Authors:          Joao Campagnolo, Semih Gunel
# Python version:   Python 3.7+

# Import packages

import math
import numpy as np
import pywt
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
from ai import cs
import matplotlib.pyplot as plt
from sklearn import svm

from OneEuroFilter import OneEuroFilter
import skeleton
import joint_functions as jf
import transformations


def angle_three_points(a, b, c):
    '''
    Given a set of any 3 points, (a,b,c), returns the angle ba^bc.
    '''
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle


def get_seperating_plane(X, y):
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)

    w = clf._get_coef()
    b = clf.intersept

    return w, b

def reshape_other(data):
    '''
    Reshapes the data with structure (frm,feat,xyz) onto (frm,featXxyz). Might come in handy.
    '''
    shaped_data = data.reshape(data.shape[0],-1)
    return shaped_data

def normalize_pose(points3d, median3d=False):
    # normalize experiment
    if median3d:
        points3d -= np.median(points3d.reshape(-1, 3), axis=0)
    else:
        for i in range(np.shape(points3d)[1]): #frames
            for j in range(np.shape(points3d)[2]): #xyz
                points3d[:,i,j] = normalize_ts(points3d[:,i,j]) 
    return points3d
 






'''   
def get_angle(data, webots=True):
    # data shape (frm,joint,xyz)
    if webots:
        # introduce Halla's joint angle functions
        datamat_j = jf.get_joint_angles(data, project_1DOF = True, interpolate = False)
        joint_names = jf.get_name_vector(joint_names = ['THORAX_COXA', 'COXA_FEMUR', 'FEMUR_TIBIA', 'TIBIA_TARSUS'], 
                  coxa_names = ['LD', 'T_1', 'T_2'], side_names = ['L', 'R'], 
                  leg_names = ['F', 'M', 'H'], 
                  last_point_name = 'CLAW'
                  )
        print("Joint names:", joint_names[1])
        data_angle = np.asarray(datamat_j[joint_names[1]]) # check Halla's new function data shape

    else:
        data_angle = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32) # (frm,pln_angl)
        for img_id in range(data.shape[0]):
            a_id = 0
            for p_id in range(1, data.shape[1] - 1, 5):           
                data_angle[img_id, a_id] = angle_three_points(
                    data[img_id, p_id - 1, :],
                    data[img_id, p_id, :],
                    data[img_id, p_id + 1, :])
                p_id = p_id + 1; a_id = a_id +1; 
                data_angle[img_id, a_id] = angle_three_points(
                    data[img_id, p_id - 1, :],
                    data[img_id, p_id, :],
                    data[img_id, p_id + 1, :])
                p_id = p_id + 1; a_id = a_id +1; 
                data_angle[img_id, a_id] = angle_three_points(
                    data[img_id, p_id - 1, :],
                    data[img_id, p_id, :],
                    data[img_id, p_id + 1, :])
                a_id = a_id +1 #write this in a more compact, more efficient fashion later. For now, it works
            
#     print("Number of nan or inf after angles: {}".format(np.sum(np.logical_or(np.isnan(data_angle), np.isinf(data_angle)))))
    data_angle[np.logical_or(np.isnan(data_angle), np.isinf(data_angle))] = 0
    return data_angle
'''
    
def get_angle_leg(data, webots=True):
    # input data shape (frame,joint,xyz)
    if webots:
        datamat_j = jf.get_joint_angles(data, project_1DOF=True, interpolate=False)
        joint_names = jf.get_name_vector(joint_names=['THORAX_COXA', 'COXA_FEMUR', 'FEMUR_TIBIA', 'TIBIA_TARSUS'],
                                         coxa_names=['LD', 'T_1', 'T_2'], side_names=['L', 'R'],
                                         leg_names=['F', 'M', 'H'],
                                         last_point_name='CLAW'
                                         )
        data_angle = np.asarray(datamat_j[joint_names[1]])

    else:
        data_angle = np.zeros((data.shape[0], 30), dtype=np.float32) # result is 5 angles per limb (6x5)
        # Defining forward reference of the fly
        mean_hind_bodycoxa = np.mean((data[:,10,:]+data[:,25,:])/2,0) #check skeleton.py to confirm the landmark IDs
        mean_mid_bodycoxa = np.mean((data[:,5,:]+data[:,20,:])/2,0)
        v_forw = mean_mid_bodycoxa - mean_hind_bodycoxa
        for img_id in range(data.shape[0]):
            a_id = 0
            for p_id in range(1, data.shape[1] - 1, 5):           
                data_angle[img_id, a_id] = angle_three_points(
                    data[img_id, p_id - 1, :],
                    data[img_id, p_id, :],
                    data[img_id, p_id + 1, :]) # hinge COXA_FEMUR
                p_id += 1; a_id += 1
                data_angle[img_id, a_id] = angle_three_points(
                    data[img_id, p_id - 1, :],
                    data[img_id, p_id, :],
                    data[img_id, p_id + 1, :]) # hinge FEMUR_TIBIA
                p_id += 1; a_id += 1
                data_angle[img_id, a_id] = angle_three_points(
                    data[img_id, p_id - 1, :],
                    data[img_id, p_id, :],
                    data[img_id, p_id + 1, :]) # hinge TIBIA_TARSUS
                a_id += 1
                o_v = data[img_id, p_id - 2, :] - data[img_id, p_id - 3, :] # body-coxa shifted to origin 
                r, data_angle[img_id, a_id], data_angle[img_id, a_id + 1] = cs.cart2sp(
                o_v[0], o_v[1], o_v[2]) # convert to cartesian | elipsiod joint (ass.) BODY_COXA T1 & T2
                a_id += 2
                #write this in a more compact, more efficient fashion later. For now, it works.
            
    print(f'Number of nan or inf after angles: {np.sum(np.logical_or(np.isnan(data_angle), np.isinf(data_angle)))}')
    data_angle[np.logical_or(np.isnan(data_angle), np.isinf(data_angle))] = 0
    # output data shape (frame, angle)
    return data_angle

def get_name_3point_angles():
    names = ['LF_COXA_FEMUR', 'LF_FEMUR_TIBIA', 'LF_TIBIA_TARSUS', 'LF_THORAX_COXA_T_1', 'LF_THORAX_COXA_T_2',
             'LM_COXA_FEMUR', 'LM_FEMUR_TIBIA', 'LM_TIBIA_TARSUS', 'LM_THORAX_COXA_T_1', 'LM_THORAX_COXA_T_2',
             'LH_COXA_FEMUR', 'LH_FEMUR_TIBIA', 'LH_TIBIA_TARSUS', 'LH_THORAX_COXA_T_1', 'LH_THORAX_COXA_T_2',
             'RF_COXA_FEMUR', 'RF_FEMUR_TIBIA', 'RF_TIBIA_TARSUS', 'RF_THORAX_COXA_T_1', 'RF_THORAX_COXA_T_2',
             'RM_COXA_FEMUR', 'RM_FEMUR_TIBIA', 'RM_TIBIA_TARSUS', 'RM_THORAX_COXA_T_1', 'RM_THORAX_COXA_T_2',
             'RH_COXA_FEMUR', 'RH_FEMUR_TIBIA', 'RH_TIBIA_TARSUS', 'RH_THORAX_COXA_T_1', 'RH_THORAX_COXA_T_2']
    return names

def get_angle_antenna_stripe(data):
    # input data shape angles (frame,landmark,xyz)
    data_angle = np.zeros((data.shape[0], 3), dtype=np.float32)
    # get 3 stripes L [:,1:4,:] angle
    for img_id in range(data.shape[0]):
        data_angle[img_id, 0] = angle_three_points(
            data[img_id, 1, :],
            data[img_id, 2, :],
            data[img_id, 3, :])
    # get 3 stripes R [:,5:8,:] angle
    for img_id in range(data.shape[0]):
        data_angle[img_id, 1] = angle_three_points(
            data[img_id, 5, :],
            data[img_id, 6, :],
            data[img_id, 7, :])
    # get antennae [:,[0,4],:] angle
    mean_ref = np.mean((data[:,0,:]+data[:,4,:])/2,0)
    for img_id in range(data.shape[0]):
        data_angle[img_id, 2] = angle_three_points(
            data[img_id, 0, :],
            mean_ref,
            data[img_id, 4, :])
    return data_angle

    
    
    
    
    
    '''
    # find the largest margin plane
    import skeleton
    x_left = data[:15, :]
    y_left = np.zeros(shape=(x_left.shape[0]))
    y_left[:] = 1
    x_right = data[skeleton.num_joints // 2:skeleton.num_joints + 15, :]
    y_right = np.zeros(shape=(x_right.shape[0]))
    y_right[:] = -1
    X = np.concatenate((x_left, x_right))
    y = np.concatenate((y_left, y_right))
    w, b = get_seperating_plane(X, y)
    # find the angles between femur and the largest margin plane
    '''



            
def normalize_frame(time_series, ax=0):
    '''
    Expects 1D array with the amplitudes of wavelet values for every frequency channel X angle comibantion. Normalizes the array by dividing each element by the sum of all of the array's elements. The final result from this step can then be seen as a probability distribution over the frequency channels. This might be usefull if the user is considering using some techniques, such as t-SNE, that are suited for measuring distances between probability distributions.
    '''
    eps = 0.0001
    amp_sum = np.transpose(np.sum(np.transpose(time_series), axis=ax)) #shape = 1,frames
#     amp_sum = np.transpose(amp_sum) #shape = 1,frames
    n_time_series = np.transpose(time_series) / np.sum(np.transpose(time_series), axis=ax)
    n_time_series = np.transpose(n_time_series)

    return n_time_series, amp_sum

def shuffle_ts(time_series):
    '''
    Time-shuffles time series with structure (frame_number, feature).
    '''
    s_time_series = np.take(time_series, np.random.rand(time_series.shape[0]).argsort(), axis=0, out=time_series)
    return s_time_series


def meancenter_ts(time_series):
    '''
    Meancenter one dimensional time series.
    '''
    c_time_series = time_series - np.mean(time_series, axis=0)
    
    return c_time_series


def remove_low(wavelets, per=10):
    # wavelets shape (frame, transf)
    var_map=np.var(wavelets,axis=1) #variance of the 39x25 dimensions at each frame
    th = np.percentile(var_map, per)
    under = np.where(var_map<th)[0]
    wavelets = np.delete(wavelets, under, 0) # removes frames
    return wavelets

def see_spectrograms(sub):
    # sub corresponds to the concatenation of all the spectrograms produced: shape (frame
    fig = plt.figure(figsize=(6,16))
    _ = plt.imshow(sub, extent=[1, nnframes, np.shape(wav)[0], 0], 
               aspect='auto')
    plt.title('Normalized feats')
    plt.ylabel('Scale x Feat')
    plt.xlabel('Time')
    return fig

def filter_batch(pts, filter_indices=None, config=None, freq=None):
    assert (pts.shape[-1] == 2 or pts.shape[-1] == 3)
    if filter_indices is None:
        filter_indices = np.arange(skeleton.num_joints)
    if config is None:
        config = {
            'freq': 100,  # Hz
            'mincutoff': 0.1,  # FIXME
            'beta': 2.0,  # FIXME
            'dcutoff': 1.0  # this one should be ok
        }
    if freq is not None:
        config['freq'] = freq

    f = [[OneEuroFilter(**config) for j in range(pts.shape[-1])] for i in range(skeleton.num_joints)]
    timestamp = 0.0  # seconds
    pts_after = np.zeros_like(pts)
    for i in range(pts.shape[0]):
        for j in range(pts.shape[1]):
            if j in filter_indices:
                pts_after[i, j, 0] = f[j][0](pts[i, j, 0], i * 0.1)
                pts_after[i, j, 1] = f[j][1](pts[i, j, 1], i * 0.1)
                pts_after[i, j, 2] = f[j][2](pts[i, j, 2], i * 0.1)
            else:
                pts_after[i, j] = pts[i, j]

    return pts_after
