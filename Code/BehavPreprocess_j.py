# File:             BehavPreprocess.py
# Date:             Winter 2019
# Description:      Reads the 3d positions of the flies's morphological features (joints, calws, stripes...)
#                   Then, smooths the data, normalizes across individuals through angles calculation, and performs 
#                   time-frequency analysis with Complex Morlet wavelets. Finally, the activity frames are labeled
# .                 and the rest frames removed.
# Authors:          Joao Campagnolo, Semih Gunel
# Python version:   Python 3.7+

# Import packages
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import filters
import statistics
import time

import skeleton
from behav_annotation import Behav, label_gt_list
from preprocess import filter_batch, reshape_other, normalize_pose, normalize_frame, meancenter_ts, get_angle_leg, get_angle_antenna_stripe
from find_wavelets import find_wav
import joint_functions as jf
from util import get_time


leg_mask = [not (skeleton.is_antenna(i) or skeleton.is_stripe(i)) for i in range(skeleton.num_joints)]
stripe_antenna_mask = [skeleton.is_antenna(i) or skeleton.is_stripe(i) for i in range(skeleton.num_joints)] #stripes and antennae

INTERVAL = 0
LABEL = 1


class BehavPreprocess(Dataset):
    def __init__(self, root_dir, stride=10, clip_len=100, dilation=1, min_frequency=5, max_frequency=50, num_channels=25, wavelet=True, smooth=True, normalize=True, halla_method=True, mean_center=True):

        self.root_path_list = root_dir
        self.stride = stride
        self.clip_len = clip_len
        self.smooth = smooth
        self.dilation = dilation
        self.wavelet = wavelet
        self.halla_method = halla_method
        self.mean_center = mean_center
        self.normalize = normalize
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.num_channels = num_channels

        self.exp_path_list = list()
        self.pts3d_exp_list = list()
        self.angl_exp_list = list()
        self.wav_exp_list = list()
        self.data_concat = list()
        self.frame_var_list = list()
        self.frame_activity_list = list()
        
        self.results_dict = {}
        self.parameters_dict = {}

        self.time_start = time.time()
        
        # Reading data
        for dir in self.root_path_list:
            self.exp_path_list.extend(glob.glob(os.path.join(dir, "./**/pose_result*.pkl"), recursive=True))
        self.exp_path_list = [p.replace('./', '') for p in self.exp_path_list]
        print(f'Training directories: {self.exp_path_list}')
        self.pts3d_exp_list = [np.load(path)["points3d"] for path in
                               self.exp_path_list]
        print("Number of experiments: {}".format(np.shape(np.array(self.pts3d_exp_list))[0]))

        # Removing illegal values
        count = 0
        for exp_idx, exp in enumerate(self.pts3d_exp_list):
            self.pts3d_exp_list[exp_idx][np.logical_or(np.isnan(exp), np.isinf(exp))] = 0
            count += np.sum(np.logical_or(np.isnan(exp), np.isinf(exp)))
        print("Total of {} nan or inf".format(count))

        # Smoothing the data
        if self.smooth:
            for exp_idx, experiment in enumerate(self.pts3d_exp_list):
                self.pts3d_exp_list[exp_idx] = filter_batch(experiment) 
        
        # Transforming to angles
        print("Transforming to angles")
        self.results_dict['joint_angles'] = {}
        for exp_idx, experiment in enumerate(self.pts3d_exp_list):
            if self.halla_method:
                legs_angle = get_angle_leg(experiment[:, leg_mask, :], webots=True)
            else:
                legs_angle = get_angle_leg(experiment[:, leg_mask, :], webots=False)
            stripe_antenna_angle = get_angle_antenna_stripe(experiment[:, stripe_antenna_mask, :])
            self.angl_exp_list.append(np.concatenate((legs_angle, stripe_antenna_angle), axis=1))
            self.results_dict['joint_angles'][f'subject{exp_idx}'] = np.concatenate((legs_angle, stripe_antenna_angle), axis=1)
            
        
        # Mean Centering
        if self.mean_center:
            print("Mean centering the angles")
            if self.mean_center:
                mean, std = self._get_mean_()
                for exp_idx, experiment in enumerate(self.angl_exp_list):
                    self.angl_exp_list[exp_idx] -= np.mean(self.angl_exp_list[exp_idx]) # OR meancenter_ts(experiment) w. _get.mean_

        # Time-frequency analysis
        print("Performing time-frequency analysis")
        if self.wavelet:
            for exp_idx, experiment in enumerate(self.angl_exp_list):
                wav_exp, freq_channels = find_wav(experiment, chan=self.num_channels,
                                      omega0=5, fps=100,
                                      fmin=self.min_frequency,
                                      fmax=self.max_frequency)
                self.wav_exp_list.append(wav_exp)
        else:
            for exp_idx, experiment in enumerate(self.pts3d_exp_list):
                self.wav_exp_3 = list()
                for i in range(np.shape(experiment)[2]):
                    wav_exp, freq_channels = find_wav(experiment[:,:,i], chan=self.num_channels,
                                          omega0=5, fps=100,
                                          fmin=self.min_frequency,
                                          fmax=self.max_frequency)
                    self.wav_exp_3.append(wav_exp)
                self.wav_exp_list.append(np.hstack(self.wav_exp_3))
        print(f'Wavelets center frequencies: {freq_channels}')
        self.results_dict['wavelet_list'] = self.wav_exp_list
        self.parameters_dict['wavelet_frequencies'] = freq_channels
                
        # Frame variance and spectrogram normalization
        for exp_idx, experiment in enumerate(self.wav_exp_list):
            self.frame_var_list.append(np.log10(np.var(experiment, axis=1))) # FIX ME: LOG10(VAR)vs(SUM)?
            
        print("Normalizing each frame")
        for exp_idx, experiment in enumerate(self.wav_exp_list):
            if self.normalize:
                self.wav_exp_list[exp_idx], self.frame_amp_sum = normalize_frame(experiment, ax=0)
            else:
                self.frame_amp_sum = normalize_frame(experiment, ax=0)[1]
            self.frame_activity_list.append(self.frame_amp_sum)
        self.results_dict['frame_activity'] = self.frame_activity_list
    
        # set experiment and frame id for each instance
        self.exp_idx = [np.ones(shape=(exp.shape[0],)) * idx for (idx, exp) in
                        enumerate(self.pts3d_exp_list)]
        self.exp_idx = np.concatenate(self.exp_idx, axis=0).astype(np.int)
        self.frame_idx = [np.arange(0, exp.shape[0]) for exp in self.pts3d_exp_list]
        self.frame_idx = np.concatenate(self.frame_idx, axis=0).astype(np.int)
        
        # set the behavior for each frame
        self.data_behav_concat = [np.zeros(shape=(exp.shape[0]), dtype=np.int64) for exp in
                                  self.pts3d_exp_list]

        # setting annotated behaviors
        for d in self.data_behav_concat:
            d[:] = Behav.NONE.value
        for exp_idx in range(len(self.data_behav_concat)):
            for offset in range(self.clip_len // 2,
                                self.pts3d_exp_list[exp_idx].shape[0] - self.clip_len // 2):
                offset_mid = offset + self.clip_len // 2
                experiment_path = self.exp_path_list[exp_idx]
                for label_gt in label_gt_list:
                    (offset_gt_start, offset_gt_end), label_gt_behav, folder_name = label_gt
                    if folder_name not in experiment_path:
                        continue
                    if offset_gt_start <= offset_mid < offset_gt_end:
                        if min(abs(offset_mid - offset_gt_start), abs(offset_mid - offset_gt_end)) > self.clip_len // 3:
                            self.data_behav_concat[exp_idx][offset_mid] = label_gt_behav.value

        if self.wavelet:
            # setting the rest frames
            self.test_rest_mask = []
            for exp_idx, experiment in enumerate(self.exp_path_list):
                thr = filters.threshold_otsu(self.frame_var_list[exp_idx], nbins=len(experiment) // 2)
                rest_mask = self.frame_var_list[exp_idx] < thr
                self.test_rest_mask.append(rest_mask)
                self.data_behav_concat[exp_idx][rest_mask] = Behav.REST.value

        if self.wavelet:
            self.data_concat = np.concatenate(self.wav_exp_list, axis=0)
        else:
            self.data_concat = np.concatenate(self.wav_exp_list, axis=0) # Replace by pts3d_exp_list
            
        self.data_behav_concat = np.concatenate(self.data_behav_concat, axis=0)
        print("Experiment concat shape {}".format(np.shape(np.array(self.data_concat))))
        print(
            "Loaded {} frames from {} folders ".format(self.data_concat.shape[0], len(self.exp_path_list)))
        assert (self.frame_idx.shape[0] == self.exp_idx.shape[0] == self.data_concat.shape[0])
        
        self.dt = time.time() - self.time_start
        print(f'Data preprocessing completed. Time elapsed: {self.dt} seconds')

        
    # Functions to use:
    
    def _get_mean_(self):
        filename = 'mean_{}'.format("train")
        if os.path.isfile(filename + '.npy'):
            filename = filename + '.npy'
            print("Loading mean file {}".format(filename))
            d = np.load(filename).item()
            mean = d["mean"]
            std = d["std"]
        else:
            print("Calculating mean file")
            experiment_concat = np.concatenate(self.pts3d_exp_list, axis=0)
            mean = np.mean(experiment_concat, axis=0)
            std = np.std(experiment_concat, axis=0)
            print("Mean: {} std: {}".format(mean, std))
            np.save(filename, {"mean": mean, "std": std})
        print("Mean shape {}".format(mean.shape))
        return mean, std

    def _len_(self):
        return self.data_concat.shape[0]
    
    def _get_item_(self, idx):
        data = self.data_concat[idx]
        behav = self.data_behav_concat[idx]

        sample = {'data': data, 'behav': behav, 'idx': idx, 'fly_id': 0}
        
        return sample
    
    def _get_data_(self, exp_name=None, exp_idx=None, start_frame=0, end_frame=900):
        if exp_idx is None:
            for idx, exp_name_iter in enumerate(self.exp_path_list):
                if exp_name in exp_name_iter:
                    exp_idx = idx
                    break
        if exp_idx is None:
            print(exp_name)
        pts3d = self.pts3d_exp_list[exp_idx][start_frame:end_frame, :]
        data_mask_frame = np.logical_and(start_frame < self.frame_idx, self.frame_idx < end_frame)
        data_mask_exp = self.exp_idx == exp_idx
        data_mask = np.logical_and(data_mask_frame, data_mask_exp)

        d = self.data_concat[data_mask]
        
        return pts3d, d
    
    