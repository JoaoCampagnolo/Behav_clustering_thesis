# Import packages:
import numpy as np
import os
import cv2
import time
import datetime
from Camera import Camera


# EMBEDDED DATA DICTIONARY:

def create_dict(labels, data, is_mesh, pix_to_point_idx, point_idx_to_pix, train_exp_index, train_frame_index, xmax, ymax, train_dataset_exp_path_list, is_low_dim=True, min_cluster_size=10):
    '''
    Creates a dictionary with the frames and, if dimensionality reduction takes place, the (x,y) positions associated
    with each frame. The dictionary nests each cluster label, the experiments that are present in them, al, the uninterrupted 
    sequences of frames that make up for behavioral bouts, and also the outlier frames that, if too short, are unnable 
    to make up for a behavioral bout.
    '''
    cluster_dict = {}
    
    if is_mesh:
        init_cluster = 1
        end_cluster = np.ndarray.max(labels)
    else:
        init_cluster = 0 # for Watershed 0 class is background, but not for HDBSCAN/GMM.
        end_cluster = max(labels)
        
    for cluster in range(init_cluster, end_cluster+1):
        cluster_dict[f'cluster{cluster}'] = {}

        if is_low_dim:

            ar_frm = _list_frames_in_cluster_(labels, cluster, pix_to_point_idx, is_mesh) # gets all frames from that cluster
            exp_list_unique = np.unique(train_exp_index[ar_frm]) # number of directories in that cluster
            
            if is_mesh:
                x = np.asarray([point_idx_to_pix[ar_frm[i]][0] for i, value in enumerate(ar_frm)])
                y = np.asarray([point_idx_to_pix[ar_frm[i]][1] for i, value in enumerate(ar_frm)])
            else:
                x = data[np.where(labels==cluster)][:,0] # axis z1 from embedded data
                y = data[np.where(labels==cluster)][:,1] # axis z2 from embedded data

            for exp_idx in list(exp_list_unique):
                seq_count = 1
                out_count = 1
                frames = train_frame_index[ar_frm][train_exp_index[ar_frm]==exp_idx] # frames from same directory only + INDEX TRANSLATION
                x_arr = x[train_exp_index[ar_frm]==exp_idx]/xmax # normalize x from same directory
                y_arr = y[train_exp_index[ar_frm]==exp_idx]/ymax
                assert len(frames) == len(x_arr) == len(y_arr)
                frames_order = np.argsort(frames) # arrange frames and (x,y) correspondingly
                sort_frames = frames[frames_order[::1]]
                sort_x = x_arr[frames_order[::1]]
                sort_y = y_arr[frames_order[::1]]
                path_dir = train_dataset_exp_path_list[exp_idx] # a list with all the directories for the dataset
                cluster_dict[f'cluster{cluster}'][f'{path_dir}'] = {}
                split_frames = np.split(sort_frames,np.where(sort_frames[1:]-sort_frames[:-1]>1)[0]+1)
                split_x = np.split(sort_x,np.where(sort_frames[1:]-sort_frames[:-1]>1)[0]+1)
                split_y = np.split(sort_y,np.where(sort_frames[1:]-sort_frames[:-1]>1)[0]+1)
                for i in range(np.shape(split_frames)[0]):
                    assert len(split_frames[i]) == len(split_x[i]) == len(split_y[i])
                    if len(split_frames[i]) >= min_cluster_size:
                        cluster_dict[f'cluster{cluster}'][f'{path_dir}'][f'sequence_{seq_count}'] = {}
                        cluster_dict[f'cluster{cluster}'][f'{path_dir}'][f'sequence_{seq_count}']['frames'] = split_frames[i]
                        cluster_dict[f'cluster{cluster}'][f'{path_dir}'][f'sequence_{seq_count}']['x'] = split_x[i]
                        cluster_dict[f'cluster{cluster}'][f'{path_dir}'][f'sequence_{seq_count}']['y'] = split_y[i]
                        seq_count+=1
                    else:
                        cluster_dict[f'cluster{cluster}'][f'{path_dir}'][f'outlier_{out_count}'] = {}
                        cluster_dict[f'cluster{cluster}'][f'{path_dir}'][f'outlier_{out_count}']['frames'] = split_frames[i]
                        cluster_dict[f'cluster{cluster}'][f'{path_dir}'][f'outlier_{out_count}']['x'] = split_x[i]
                        cluster_dict[f'cluster{cluster}'][f'{path_dir}'][f'outlier_{out_count}']['y'] = split_y[i]
                        out_count+=1

    #                 cluster_dict['2D_projection'] = {}
    #                 cluster_dict['2D_projection'] = projection

        else: #if high dimension data was clustered (unlikely)
        # finish this later: just directories, clusters, frame sequences and outliers.
            print('Segmentation in high dmensional space not ready yet')

    return cluster_dict, init_cluster, end_cluster

def _list_frames_in_cluster_(labels, cluster, pix_to_point_idx, is_mesh):
    if is_mesh:
        max_reg = labels.max()
        pos = np.where(labels == cluster)
        cluster_frames_list = list() # list of all the frames from this cluster
        for (x, y) in zip(pos[0], pos[1]):
            cluster_frames_list.extend(list(pix_to_point_idx[(x,y)]))
    else:
            cluster_frames_list = np.where(labels == cluster)[0]

    return cluster_frames_list



# VIDEO GENERATION THROUGH ffmpeg

def _get_image_(path, img_id):
    path = os.path.join(os.path.dirname(path), 'camera_1_img_{:06d}.jpg'.format(img_id))
    img = cv2.imread(path)
    assert(img is not None)

    return img


def _create_video_(img_list, img_id_list, save_path, fps, shape):
    print("Writing {0} frames, w. min {1}".format(len(img_list), min(img_id_list)))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#     fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(save_path, fourcc, fps, shape)
    #out = cv2.VideoWriter(save_path,fourcc, 20.0, (640,480))
    for img in img_list:
        out.write(img)
    out.release()

def cluster_video_frames(cluster, frame_list):
    #frame_list is cluster_dict
    l = frame_list[f'cluster{cluster}']
    cluster_videos = list()
    n_exp = len(l.keys())
    for i in range(n_exp):
        seq_list = list(list(l.values())[i])
        n_seq = len(seq_list)
        k = 1
        for j in range(n_seq):
            if list(list(l.values())[i])[j][0:3]=='seq':
                frames = list(l.values())[i][f'sequence_{k}']['frames']
                cluster_videos.append((cluster, i, frames))
                k += 1

    return cluster_videos


def create_all_videos(cluster, cluster_videos, save_dir_path, exp_path_list, cid=1, fps=7):
    vid_paths = []
    for cluster, exp_idx, img_id_list in cluster_videos:
        
#         save_dir_path = '/home/user/Desktop/joao_pose/results/14h45m_27_03_2019/fly_vid'
        save_path = os.path.join(save_dir_path, 'cluster_{}_{}_{}.mp4'.format(cluster, os.path.dirname(exp_path_list[exp_idx]).replace('/','_'), min(img_id_list)))
        vid_paths.append(save_path)
    #     img_list = [_get_image_(experiment_path_list[exp_idx], img_id) for img_id in img_id_list]
        data = np.load(exp_path_list[exp_idx])
        nimg_list = [Camera(image_folder=os.path.dirname(exp_path_list[exp_idx]), cid=cid, points2d=data["points2d"][1]).plot_2d(img_id=img_id) for img_id in img_id_list] # +'/behData/images' ^ corrects the directory's extra folders 
        nimg_list = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in nimg_list]

        _create_video_(nimg_list, img_id_list, save_path, fps=fps, shape=(960,480))

    return vid_paths


def _frame_xy_(n_videos, sqr=True):
    if np.sqrt(n_videos) > int(np.sqrt(n_videos)):
        side = int(np.sqrt(n_videos)) + 1
    if np.sqrt(n_videos) == int(np.sqrt(n_videos)):
        side = int(np.sqrt(n_videos))

    if sqr:
        rows = side; columns = side
    else: # start from square and remove rows
        rows = side - int((side**2 - n_videos) / side)
        columns = side

    return rows, columns


def _get_paths_(video_paths, k=10):
    short_paths = []
    for i in range(len(video_paths)):
        keyseg = video_paths[i].split("_")
        newstr = keyseg[k]+'_'+keyseg[k+1]+'_'+keyseg[k+2]+'_'+keyseg[k+3]+'_'+keyseg[k+4]+'_i'+keyseg[k+8][:-4]
        short_paths.append(newstr)

    return short_paths


def ffmpeg_cmd(cluster, video_paths, fly_vid_dir, cluster_vid_dir, height_in_px=480, width_in_px=960):
    # TODO add input_path & output_path

    short_id = _get_paths_(video_paths, k=13)
    num_vid = len(video_paths)

    rows, columns = _frame_xy_(num_vid, sqr=False) #; print('#vid:{0}, rows:{1}, columns:{2}'.format(n_v, rows, columns))
    height = height_in_px * rows; width = width_in_px * columns #; print('width:{0}, height:{1}'.format(width, height))
    font_type = '/Users/joaohenrique/Downloads/helvetica_fonts/Helvetica.ttf'
    rel_sz_font = .075
    font_sz = int(480*rel_sz_font); #print('font size:',font_sz)
    rel_sz_pad = .2
    pad_sz = int(rel_sz_pad*960)

    p_h = int(height / rows); p_w = int(width / columns)
    p_dim = '{0}x{1}'.format(p_w, p_h)

    if num_vid==1:
        args = 'ffmpeg -y'
        args += (' -i {}'.format(video_paths[0]))
        args += ' -filter_complex '
        filters = '"color=s={0}x{1}:c=Black[base];'.format(width,height+pad_sz)
        filters += '[base]drawtext=fontfile={0}:text="Cluster_{1}":fontcolor=White:fontsize={2}:x=(w-text_w)/2:y=(h-text_h)*.02[nbase];'.format(font_type, cluster, 5*font_sz)
        filters += '[0:v]drawtext=fontfile={2}:text="{0}":fontcolor=Black:fontsize={1}:x=(w-text_w)/2:y=(h-text_h)*.99[a0];'.format(short_id[0], font_sz, font_type)
        filters += '[a0]setpts=PTS-STARTPTS,scale={0}[b0];'.format(p_dim)
        filters += '[nbase][b0]overlay=shortest=1:x=0:y={0}'.format(pad_sz)
        output = cluster_vid_dir + '/videos_cluster_{}.mp4'.format(cluster)
        cmd = args + filters + '" -an -crf 12 '+ output
        print(cmd)

    if num_vid>1:
        args = 'ffmpeg -y'

        for vid_idx, vid in enumerate(video_paths):
            args += (' -i {}'.format(vid))

        args += ' -filter_complex '

        filters = '"color=s={0}x{1}:c=Black [base];'.format(width,height+pad_sz)
        filters += '[base]drawtext=fontfile={0}:text="Cluster_{1}":fontcolor=White:fontsize={2}:x=(w-text_w)/2:y=(h-text_h)*.02[nbase];'.format(font_type, cluster, 5*font_sz)

        for i in range(rows):
            for j in range(columns):
                idx = (i * columns) + j
                if idx < len(video_paths):
                    filters += '[{0}:v]drawtext=fontfile={3}:text="{1}":fontcolor=Black:fontsize={2}:x=(w-text_w)/2:y=(h-text_h)*.99[a{0}];'.format(idx, short_id[idx], font_sz, font_type)
                    filters += '[a{0}]setpts=PTS-STARTPTS,scale={1}[b{0}];'.format(idx, p_dim)

        cur_ffmpeg_handle_ptr = 'tmp0'
        tmp = 'tmp0'
        filters += '[nbase][b0]overlay=shortest=1:x=0:y={0}[{1}]'.format(pad_sz,tmp)
    #     for i, j in product(range(rows), range(columns)):
        for i in range(rows):
            for j in range(columns): # the first video is set before the cicle
                idx = (i * columns) + j
                if idx != 0:
                    tmp_next = 'temp{}'.format(idx)
                    xy = ':x={0}:y={1}'.format(p_w * j, p_h * i + pad_sz)
                    if idx < len(video_paths) - 1:
                        filters += ';[{0}][b{1}]overlay=repeatlast=1{2}[{3}]'.format(tmp, idx, xy, tmp_next)
                    if idx == len(video_paths) - 1:
                        filters += ';[{0}][b{1}]overlay=repeatlast=1{2}'.format(tmp, idx, xy)

                    tmp = tmp_next

        output = cluster_vid_dir + '/videos_cluster_{}.mp4'.format(cluster)

        cmd = args + filters + '" -an -crf 12 '+ output
        print(cmd)

    return cmd

