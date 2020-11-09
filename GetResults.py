# File:             GetResults.py
# Date:             April 2019
# Description:      After performing segmentation on the 2-dimensional behavioral space, or having direclty clustered
#                   the higher-dimensional data, the user gets frame sequences within each cluster. In this section, the
#                   clustering results are saved in a dictionary, and the cluster videos are created and stored within 
#                   the given directories.
# Authors:          Joao Campagnolo
# Python version:   Python 3.7+

# Import packages:
import numpy as np
import os
import time
import datetime
import subprocess
from get_results_util import create_dict, cluster_video_frames, create_all_videos, ffmpeg_cmd
from util import get_time


class GetResults():
    def __init__(self, labels, data, is_mesh, pix_to_point_idx, point_idx_to_pix, train_exp_index, train_frame_index, xmax, ymax, train_dataset_exp_path_list, fly_vid_dir, cluster_vid_dir, min_cluster_size=10, low_dim=True, get_videos=True):

        self.labels = labels
        self.data = data
        self.low_dim = low_dim
        self.is_mesh = is_mesh
        self.pix_to_point_idx = pix_to_point_idx
        self.point_idx_to_pix = point_idx_to_pix
        self.train_exp_index = train_exp_index
        self.train_frame_index = train_frame_index
        self.xmax = xmax
        self.ymax = ymax
        self.train_dataset_exp_path_list = train_dataset_exp_path_list
        self.min_cluster_size = min_cluster_size
        self.get_videos = get_videos
        
        self.fly_vid_dir = fly_vid_dir
        self.cluster_vid_dir = cluster_vid_dir
        
        self.time_start = time.time()

        # Create results dictionary:
        
        self.cluster_dict, self.init_cluster, self.end_cluster = create_dict(self.labels, self.data, self.is_mesh, self.pix_to_point_idx, self.point_idx_to_pix, self.train_exp_index, self.train_frame_index, self.xmax, self.ymax, self.train_dataset_exp_path_list, min_cluster_size=self.min_cluster_size, is_low_dim=self.low_dim)

        
        # Create videos for every area:
        if self.get_videos:
            for cluster in range(self.init_cluster, self.end_cluster+1):
                self.cluster_frames = cluster_video_frames(cluster, self.cluster_dict)
                if not self.cluster_frames:
                    print(f'No video sequences can be recovered from cluster {cluster}, only outliers')
                else:
                    print(f'Video sequences from cluster {cluster}')
                    self.video_paths = create_all_videos(cluster, self.cluster_frames, self.fly_vid_dir, 
                                                         train_dataset_exp_path_list, cid=1, fps=10)
                    self.cmd = ffmpeg_cmd(cluster, self.video_paths, self.fly_vid_dir, self.cluster_vid_dir)

                    print('Creating video for cluster', cluster)
                    subprocess.call(self.cmd , shell=True)
                    
        self.dt = time.time()-self.time_start
        print(f'Results created and stored. Time elapsed: {self.dt} seconds')