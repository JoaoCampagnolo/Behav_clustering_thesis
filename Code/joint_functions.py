# File:             joint_functions.py
# Date:             Winter 2018 - 2019
# Description:      A file that keeps functions used to compute the joint angles from 
#                   3D joint positions (inverse kinematics) recorded from real fly experiments
# Authors:          Halla Bjorg Sigurthorsdottir
# Python version:   Python 2.7+

# Import packages
import numpy as np
from numpy.linalg import norm
import pandas as pd
import copy
import scipy.optimize
import functools

import transformations as T

### The main joint angle calculator
def get_joint_angles(path, 
fix_body_coxa = True, gravity_reference = 'y', forward_reference = 'HMcoxa', convention = 'syzx', negative_starting_angles = [], deul_error_margin = .6, negative_angles = True, coxa_x_ref = [3,4], plane_angle = False, plane_thr = 0.8, leg_plane_points = [1,3,4], project_1DOF = True, eul_ref_type = 'coxa_x', 
ld_correction = np.pi/2, plot_large_difference = False, verify = False, verify_plot = False,
joint_names = ['THORAX_COXA', 'COXA_FEMUR', 'FEMUR_TIBIA', 'TIBIA_TARSUS'], coxa_names = ['LD', 'T_1', 'T_2'], eul_coxa_order = [0, 2, 1], side_names = ['R', 'L'], leg_names = ['F', 'M', 'H'], last_point_name = 'CLAW',
no_legs = 6, points_per_leg = 5,
interpolate = True, time_step = 10, webots_factor = 0.1, webots_time_step = 2, wt_get_to_position = 1000,
joint_angle_filename = None, diff_filename = None, plane_dist_filename = None, segment_length_filename = None, plane_angle_name = None, error_name =  None
):
    '''
    This function takes in the path to a file of 3D joint positions over time and returns the corresponding joint angles
    The data should be on a .npy format with dimensions time x joints x [x,y,z]
    It is assumed that the six legs have 6 degrees of freedom, 3 on the coxa joint and 1 on the three others
    Assumption: none of the leg joints will ever rotate a 360 degrees
    Parameters:
        path - this is either the path to the data or a 3D numpy array containing the data
        fix_body_coxa - If true, the body-coxa points are fixed to the median of their position over time
        gravity_reference - this variable defines the method of defining the gravity axis of the fly. "y" uses the y-axis of the data (second axis), "coxa" uses the normal to the body-coxa plane
        forward_reference - this vairable defines the method of determining the direction of forward of the fly. "x" uses the x-axis of the data (first axis) projected onto the plane defined by the gravity axis [has gravity axis as normal], "coxa" uses the difference between mean of the hind body-coxa points and the mean of the front body-coxa points projected onto the plane defined by the gravity axis, "HMcoxa" uses the difference between the hind body-coxa point and middle body-coxa point on each side projected onto the plane defined by the gravity axis
        convention - the order of which the axes are considered when calculating Euler angles exmple: 'sxyz'. Default is 'syzx'.
        negative_starting_angles - all angles except the ones stated here are assumed to start out positive at t = 0
        deul_error_margin - the estimated error allowed from abs(deul - abs(abs(eul) - eul_ref)) = 0 before we consider the angle negative
        negative_angles - True: allows 1DOF joint angles to be negative, False: ignores this part of the algorithm
        coxa_x_ref - the point/points that are used as reference to define the rotationa round the coxa. It is either an int or a vector of translation from the thorax-coxa joint (toward the claw). 1 - coxa-femur, 2 - femur-tibia, 3 - tibia-tarsus, 4 - claw.
        plane_angle - if True, a threshold and comparison between the angles between different sub-planes of the leg are used to exclude points that are very far from the leg plane
        plane_thr - the threshold that any of the anlges above need to cross so a point will be discarded from the leg plane calculations
        leg_plane_points = the points that are used (as well as the coxa point) to define the leg plane
        project_1DOF - if = True, projects the vectors in question onto the leg plane before computing the angle between them; if = False, computes the angle directly without projecting
        eul_ref_type - if = 'coxa_x' the sign of the x-coordinate of the projected femur vector to the coxa frame is used, if = 'normal', it will use the normal of the leg plane and the normal of the computed angles to determine angle sign, otherwise it will use triangle method and if = 'first' it will always use the first angle as a reference for angle sign; if = 't-1', it will always use the angle before the current angle as reference
        ld_correction - "large difference" correction. If this is not None, the value (float) is used to correct the data if there are big jumps (big defined by this value) in the joint angle in the same direction within the file
        plot_large_difference - if this is True, a plot of the corrected files will be shown
        verify - whether to calculate error of the coxa-femur and femur-tibia positions
        verify_plot - whether to plot verification plots of all rotation matrices
        Data parameters:
        joint_names - these are the names of the joints, the order here determines the order in the input file
        coxa_names - these are the names of the extensions to the name of the coxa joint (the 3DOF joint). The order determines the order in the output file.
        eul_coxa_order - the axes of rotation of the names in coxa_names (according to the convention specified in the parameter convention)
        side_names - the names of the sides (L and R). The order defines the order in the input file.
        leg_names - the names of the legs (F, M, H). The order defines the order in the input file.
        last_point_name - the name of the final point of each leg. Default is 'CLAW'.
        DOF_3_point_name - the name of the 3DOF joint. Default is 'COXA'.
        no_legs - the number of legs. Default is 6.
        points_per_leg - the number of datapoints per leg. Default is 5.
        time_step - time step between datapoints in real data (in ms). Default is 50.
        webots_factor - the factor by which Webots multiplies time. Default is 0.1.
        webots_time_step - the time step in Webots (as it is inputted). Default is 2.
        wt_get_to_position - the time (in Webots time) needed to get safely into position. Default is 1000.
        joint_angle_filename - the name of the csv file (including .csv extension) that the joint angles should be exported to. Default value is None, meaning the function does not export the joint angles as a .csv file.
        diff_filename - the name of the pickle file (including .p extension) that the matrix of the differences deul - abs(abs(eul) - eul_ref) should have
        plane_dist_filename - the name of the pickle file (including .p extnsion) that the distances of the leg points to the leg plane are saved to
        segment_length_filename - the name of the txt file (including .txt extension) keeping mean leg segment lengths of the fly
        plane_angle_name - the name of the .p file (including .p extension) that keeps the angle between the coxa-femur plane and coxa-[tibia-tarsus joint] plane, coxa-femur plane and coxa-claw plane and coxa-[tibia-tarsus joint] and coxa-claw plane (they are in this order in the last coordinate of the 3D output array) over time for each leg 
    '''
    # Load the data
#     print('Loading data...')
    if type(path) == str:
        data = np.load(path)
    else:
        data = path

    data = axes_swap(data)
    #data = filter_batch(data)

    if verify_plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D 
        from matplotlib import rc
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  
        for i in range(no_legs):
            d = data[0,i*points_per_leg:((i+1)*points_per_leg), :]
            x = d[:,0]
            y = d[:,1]
            z = d[:,2]
            ax.scatter(x,y,z)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.gca().set_color_cycle(colors)
        for i in range(no_legs):
            d = data[0,i*points_per_leg:((i+1)*points_per_leg), :]
            x = d[:,0]
            y = d[:,1]
            z = d[:,2]
            ax.plot(x,y,z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.legend(['Right front', 'Right middle', 'Right hind', 'Left front', 'Left middle', 'Left hind'])#np.arange(1, no_legs + 1, 1))
        plt.show()

    name_vector, extended_name_vector = get_name_vector(joint_names = joint_names, coxa_names = coxa_names, side_names = side_names, leg_names = leg_names, last_point_name = last_point_name)
    
#     if len(name_vector) != data.shape[1]:
#         print('WARING: the name length does not match the data length')

    if fix_body_coxa:
#         print('Fixing body-coxa point...')
        data = fix_coxa(data, segment = 0, no_legs = no_legs, points_per_leg = points_per_leg) # Fixing the coxa position to the mean position over time
        R_fl = find_fly_R(data, gravity_reference, forward_reference) # Find the fly frame

    datamat_j = pd.DataFrame(np.zeros((len(data)+1, len(extended_name_vector))), columns = extended_name_vector)
    
    # Estimate leg segment length for each leg
    segment_lengths = estimate_segment_length(data, no_legs = no_legs, points_per_leg = points_per_leg)

    

    
    if diff_filename is not None:
        diff = {'diff': None, 'eul': None, 'eul_ref': None, 'dist': None}
        for key in diff.keys():
            diff[key] = np.zeros([len(data), len(name_vector)]) # Collect the abs(deul - abs(abs(eul) - eul_ref))

    if plane_dist_filename is not None:
        plane_dist = np.zeros([len(data), no_legs, points_per_leg])
    
    if plane_angle_name is not None:
        plane_ang = np.zeros([len(data), no_legs, 3])

    if eul_ref_type == 'first':
        t_ref = np.zeros(len(name_vector))

    if eul_ref_type == 'normal':
        sign_ref = np.ones(len(name_vector))

    if eul_ref_type == 't-1':
        eul_ref = [None]*len(name_vector) # Initialize euler angle reference for sign

    if not eul_ref_type == 'coxa_x':
        first_angle_assumption = np.ones(len(name_vector))

        for i in range(len(name_vector)):
            if name_vector[i] in negative_starting_angles:
                first_angle_assumption[i] = -1
    if error_name is not None:
        verify = True


    if verify:
        error = np.zeros((len(data), data.shape[1]))
        plane_error = np.zeros((len(data), no_legs))
        cox_err = np.zeros((len(data), no_legs))
        fem_err = np.zeros((len(data), no_legs))
        fem_plane_err = np.zeros((len(data), no_legs))

    print('Calculating joint angles...')
    for t in range(len(data)):
#         if t%10 == 0 or t == len(data)-1:
#             print('t = ' + str(t))
        for joint in range(len(name_vector)):
            if joint_names[0] in name_vector[joint]:

                # if t in [198, 864, 866]: #w6 #w20 #[405, 406]: #w22 #491 #w16 #637 w23
                #     import matplotlib.pyplot as plt
                #     from mpl_toolkits.mplot3d import Axes3D 
                #     verify_plot = True
                # else:
                #     verify_plot = False


                coxa_name_vector = []
                for coxa_name in coxa_names:
                    coxa_name_vector = np.append(coxa_name_vector, name_vector[joint] + '_' + coxa_name)
                
                if plane_angle: # Whether to use angles between leg sub-planes to decide which point to discard from leg plane calculation
                    coxa_femur = (data[t, joint + np.array([0,1,2]), :] - data[t, joint, :]).T
                    femur_tibia =(data[t, joint + np.array([0,1,3]), :] - data[t, joint, :]).T # Not actually femur-tibia
                    tibia_tarsus = (data[t, joint + np.array([0,1,4]), :] - data[t, joint, :]).T # Nota ctually tibia-tarsus
                    cf_n = find_plane_normal(coxa_femur)
                    ft_n = find_plane_normal(femur_tibia)
                    tt_n = find_plane_normal(tibia_tarsus)
                    ang = np.array([angle_between(cf_n, ft_n) , angle_between(cf_n, tt_n), angle_between(ft_n, tt_n)])

                    ang[ang > np.pi/2] = ang[ang > np.pi/2] - np.pi # To account for when the normal switches sides
                    ang = abs(ang)

                    if sum(ang > plane_thr) > 1:
                        if np.argmin(ang) == 0:
                            leg_plane = joint + np.arange(0,points_per_leg - 1) # Removing claw from leg plane calculations
                        elif np.argmin(ang) == 1:
                            leg_plane = joint + np.array([0,1,2,4]) # Removing tibia-tarsus joint from leg plane calculations
                        else:
                            leg_plane = joint + np.array([0,1,3,4]) # Removing femur-tibia joint from leg plane calculations
                    else:
                        leg_plane = joint + np.arange(0,points_per_leg, 1)
                    
                    if plane_angle_name is not None:
                        plane_ang[t, joint/points_per_leg, :] = ang

                elif leg_plane_points is not None:
                    leg_plane = joint + np.append(0, leg_plane_points)

                else:
                    leg_plane = joint + np.arange(0,points_per_leg, 1)

                P = data[t, leg_plane, :] - data[t, joint, :] # Translate all leg points by body-coxa point (move body-coxa point to origin)
                P = np.array(P).T
                n = find_plane_normal(P) # Find the normal to the closest plane to the points in P (all leg points of current leg)

                if plane_dist_filename is not None:
                    for i in range(points_per_leg):
                        plane_dist[t, joint/points_per_leg, i] = find_dist(data[t, joint + i, :], n)#/segment_lengths[1]
                    
                p1 = data[t, joint, :] # The current joint point
                p2 = data[t, joint + 1, :] # The pointa after the current joint (coxa-femur)
                
                # The point that decides the direction of the rotation around the coxa
                
                if len(np.array(coxa_x_ref).shape) == 0:
                    p3 = data[t, joint + coxa_x_ref, :]
                else:
                    p3 = np.mean(data[t, joint + np.array(coxa_x_ref), :], axis = 0) 
                    

                v1 = p1 - p2 #p2 - p1 # # Define a vector along the coxa (segment after current joint)
                v2 = p3 - p2 # Define a vector along the leg. #the femur (segment two after the current joint). This is to define the rotation around the coxa.

                R_cl = find_rotation_matrix(n, v1, v2) # The coxa frame to lab frame matrix

                

                if not fix_body_coxa:
                    R_fl = find_fly_R(data[t, :, :], g_ref = gravity_reference, f_ref = forward_reference)

                if verify or verify_plot:
                    if joint <= data.shape[1]/2:    
                        R_f = R_fl[0]
                    else:
                        R_f = R_fl[1]
                
                    # if verify_plot:
                        # # Plot the coxa frame
                        # fig = plt.figure(figsize = (5,5))
                        # #ax = fig.add_subplot(111, projection='3d')
                        
                        # #ax.set_xlabel('x')
                        # #ax.set_ylabel('y')
                        # #ax.set_zlabel('z')
                        
                        # print(np.dot(R_cl[:,0],R_cl[:,1]), np.dot(R_cl[:,0],R_cl[:,2]), np.dot(R_cl[:,1],R_cl[:,2]))
                        
                        # #ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], '--')
                        # if joint < points_per_leg*3:
                        #     hl_tc_joint = 2*points_per_leg
                        #     first_leg = 0
                        # else:
                        #     hl_tc_joint = 5*points_per_leg
                        #     first_leg = 3
                        # # for i in range(len(R_f)):
                        # #     d = R_f[:,i]
                        # #     x = [0, d[0]] + data[t,hl_tc_joint, 0]
                        # #     y = [0, d[1]] + data[t,hl_tc_joint, 1]
                        # #     z = [0, d[2]] + data[t,hl_tc_joint, 2]
                        # #     plt.plot(x,y,z, color = 'k')

                        # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  
                        # for i in [first_leg]:#range(first_leg, no_legs/2 + first_leg):
                        #     #plt.gca().set_color_cycle(colors[])
                        #     d = data[t,i*points_per_leg:((i+1)*points_per_leg), :]
                        #     x = d[:,0]
                        #     y = d[:,1]
                        #     z = d[:,2]
                        #     plt.scatter(x,y)#ax.scatter(x,y,z)
                    
                        # plt.gca().set_color_cycle(colors)
                        # for i in [first_leg]:#range(first_leg, no_legs/2 + first_leg):
                        #     #plt.gca().set_color_cycle(colors[6])
                        #     d = data[t,i*points_per_leg:((i+1)*points_per_leg), :]
                        #     x = d[:,0]
                        #     y = d[:,1]
                        #     z = d[:,2]
                        #     plt.plot(x,y)#ax.plot(x,y,z)
                    
                    
                    # # if joint == 2*points_per_leg:# For 3D coordinate system

                        # for i in range(len(R_cl)):
                        #     d = R_cl[:,i]
                        #     x = [0, d[0]] + p2[0]
                        #     y = [0, d[1]] + p2[1]
                        #     z = [0, d[2]] + p2[2]
                            # # plt.plot(x,y,z) # For 3D coordinate system
                        # # plt.plot([p2[0], v1[0] + p2[0]], [p2[1], v1[1] + p2[1]],[p2[2], v1[2] + p2[2]], 'k')
                            #plt.plot(x,y)#,z)
                        # # Create cubic bounding box to simulate equal aspect ratio
                        # # [x_min, x_max] = ax.get_xlim()
                        # # [y_min, y_max] = ax.get_ylim()
                        # # [z_min, z_max] = ax.get_zlim()
                        # # max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max()
                        # # Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x_max+x_min)
                        # # Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y_max+y_min)
                        # # Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z_max+z_min)
                        # # # Comment or uncomment following both lines to test the fake bounding box:
                        # # for xb, yb, zb in zip(Xb, Yb, Zb):
                        # #     ax.plot([xb], [yb], [zb], 'w')
                        # # ax.set_ylim(-max_range*0.5 + 0.5*(y_max+y_min),max_range*0.5 + 0.5*(y_max+y_min))
                        
                        # plt.legend(['Leg', r'$\hat{i}_{c}$', r'$\hat{j}_{c}$', r'$\hat{k}_{c}$'])#[r'$\Hat{i}_{c}$', r'$\Hat{i}_{c}$', r'$\Hat{i}_{c}$', 'RFL', 'RML', 'RHL'])#['x_cl', 'y_cl', 'z_cl', 'x_fl', 'y_fl', 'z_fl', 'data'])
                        # plt.xlabel('x', fontsize = 12)
                        # plt.ylabel('y', fontsize = 12)
                        # plt.axis('equal')
                        # plt.show()


                        # # Plot the fly frame
                        # fig = plt.figure(figsize = (5,5))

                        # if joint < points_per_leg*3:
                        #     hl_tc_joint = 2*points_per_leg
                        #     first_leg = 0
                        # else:
                        #     hl_tc_joint = 5*points_per_leg
                        #     first_leg = 3

                        # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  
                        # for i in range(first_leg, no_legs/2 + first_leg):
                        #     #plt.gca().set_color_cycle(colors[3:])
                        #     d = data[t,i*points_per_leg:((i+1)*points_per_leg), :]
                        #     x = d[:,0]
                        #     #y = d[:,1]
                        #     z = -d[:,2]
                        #     plt.scatter(x,z)
                    
                        # plt.gca().set_color_cycle(colors)
                        # for i in range(first_leg, no_legs/2 + first_leg):
                            
                        #     d = data[t,i*points_per_leg:((i+1)*points_per_leg), :]
                        #     x = d[:,0]
                        #     #y = d[:,1]
                        #     z = -d[:,2]
                        #     plt.plot(x,z)

                        


                        # d = data[t,[first_leg,(first_leg+points_per_leg),(first_leg+2*points_per_leg)], :]
                        # d = d[:,[0,2]]
                        # #d = d.T
                        # plt.plot(d[:,0], -d[:,1], '*')#, 'r*')
                        # # plt.plot([d[1,0]], [d[1,1]], 'r*')
                        # # plt.plot([d[2,0]], [d[2,1]], 'r*')
                        # for i in range(len(R_f)):
                        #     d = R_f[:,i]
                        #     x = [0, d[0]] + data[t,hl_tc_joint, 0]
                        #     #y = [0, d[1]] + data[t,hl_tc_joint, 1]
                        #     z = [0, d[2]] + data[t,hl_tc_joint, 2]
                        #     if i == 1:
                        #         plt.plot(x,-z, '*', markersize=12)
                        #     else:
                        #         plt.plot(x,-z)#, color = 'k')
                        # # Create cubic bounding box to simulate equal aspect ratio
                        # [x_min, x_max] = ax.get_xlim()
                        # [y_min, y_max] = ax.get_ylim()
                        # # max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max()
                        # # Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x_max+x_min)
                        # # Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y_max+y_min)
                        # # Comment or uncomment following both lines to test the fake bounding box:
                        # #for xb, yb, zb in zip(Xb, Yb, Zb):
                        # #    ax.plot([xb], [zb], 'w')
                        # #ax.set_ylim(-max_range*0.5 + 0.5*(y_max+y_min),max_range*0.5 + 0.5*(y_max+y_min))
                        # # plt.xlim = [-max_range*0.5 + 0.5*(x_max+x_min),max_range*0.5 + 0.5*(x_max+x_min)]
                        # # plt.ylim = [-max_range*0.5 + 0.5*(y_max+y_min),max_range*0.5 + 0.5*(y_max+y_min)]
                        # plt.axis('equal')
                        # plt.ylabel('z', fontsize = 12)
                        # plt.xlabel('x', fontsize = 12)
                        # plt.legend(['Front leg', 'Middle leg', 'Hind leg', 'Thorax-coxa joints', r'$\hat{i}_{f}$', r'$\hat{j}_{f}$', r'$\hat{k}_{f}$' ])#[r'$\Hat{i}_{c}$', r'$\Hat{i}_{c}$', r'$\Hat{i}_{c}$', 'RFL', 'RML', 'RHL'])#['x_cl', 'y_cl', 'z_cl', 'x_fl', 'y_fl', 'z_fl', 'data'])
                        # plt.show()

                        # Plot leg planes
                        # fig = plt.figure(figsize = (10,5))
                        
                        # print(np.dot(R_cl[:,0],R_cl[:,1]), np.dot(R_cl[:,0],R_cl[:,2]), np.dot(R_cl[:,1],R_cl[:,2]))
                        
                        # #ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], '--')
                        # if joint < points_per_leg*3:
                        #     hl_tc_joint = 2*points_per_leg
                        #     first_leg = 0
                        # else:
                        #     hl_tc_joint = 5*points_per_leg
                        #     first_leg = 3
                        # # for i in range(len(R_f)):
                        # #     d = R_f[:,i]
                        # #     x = [0, d[0]] + data[t,hl_tc_joint, 0]
                        # #     y = [0, d[1]] + data[t,hl_tc_joint, 1]
                        # #     z = [0, d[2]] + data[t,hl_tc_joint, 2]
                        # #     plt.plot(x,y,z, color = 'k')

                        # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  
                        # for i in [first_leg]:#range(first_leg, no_legs/2 + first_leg):
                        #     #plt.gca().set_color_cycle(colors[])
                        #     d = data[t,i*points_per_leg:((i+1)*points_per_leg), :]
                        #     x = d[:,0]
                        #     y = d[:,1]
                        #     z = d[:,2]
                        #     plt.scatter(x,y)#ax.scatter(x,y,z)
                    
                        # plt.gca().set_color_cycle(colors)
                        # for i in [first_leg]:#range(first_leg, no_legs/2 + first_leg):
                        #     #plt.gca().set_color_cycle(colors[6])
                        #     d = data[t,i*points_per_leg:((i+1)*points_per_leg), :]
                        #     x = d[:,0]
                        #     y = d[:,1]
                        #     z = d[:,2]
                        #     plt.plot(x,y, linewidth = 7)#ax.plot(x,y,z)
                        # for i in range(2,5):
                        #     d = data[t,[0,1,i, 0], :]
                            
                        #     plt.plot(d[:,0], d[:,1], alpha = 0.5)
                        
                        # for i in range(2,5):
                        #     plt.gca().set_color_cycle(colors[i-1:])
                        #     plt.plot(data[t,i, 0], data[t,i, 1], 'o')
                        
                        
                        # plt.legend(['Leg', 'CFp', 'CTp', 'CCp'], fontsize = 14)#[r'$\Hat{i}_{c}$', r'$\Hat{i}_{c}$', r'$\Hat{i}_{c}$', 'RFL', 'RML', 'RHL'])#['x_cl', 'y_cl', 'z_cl', 'x_fl', 'y_fl', 'z_fl', 'data'])
                        # plt.xlabel('x', fontsize = 14)
                        # plt.ylabel('y', fontsize = 14)
                        # #plt.axis('equal')
                        # plt.show()

                if forward_reference == 'HMcoxa':
                    if joint > data.shape[1]/2: 
                        R = np.dot(R_cl, np.linalg.inv(R_fl[1]))
                    else:
                        R = np.dot(R_cl, np.linalg.inv(R_fl[0]))
                else:
                    R = np.dot(R_cl, np.linalg.inv(R_fl)) # The coxa frame to fly frame matrix
                
                eul = T.euler_from_matrix(R, convention)
                if verify or verify_plot:
                    R_ver = T.euler_matrix(eul[0], eul[1], eul[2])
                    point_y = -R_f[:,1]#segment_lengths[joint/points_per_leg*4] 
                    point_x = R_f[:,0]#segment_lengths[joint/points_per_leg*4+1]
                    
                    if verify_plot:
                        if joint == 0:
                            fig = plt.figure()
                            ax = fig.add_subplot(111, projection='3d')
                            colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
                            ax.plot([0],[0],[0], '--', color = colors[6])
                            ax.plot([0],[0],[0], colors[0])
                        
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        ax.set_zlabel('z')

                        # py = point_y*norm(p1-p2)
                        # px = point_x*norm(p2-p3)
                        # x = [0, py[0]] + p1[0]
                        # y = [0, py[1]] + p1[1]
                        # z = [0, py[2]] + p1[2]
                        # ax.plot(x, y, z)
                    # x = [0, px[0]] + py[0] + p1[0]
                    # y = [0, px[1]] + py[1] + p1[1]
                    # z = [0, px[2]] + py[2] + p1[2]
                    # ax.plot(x, y, z)

                    
                    y_rot = np.dot(R,point_y)
                    #x_rot = np.dot(R,point_x)
                    
                    yr = y_rot*norm(p1-p2)
                    point_old = yr + p1
                    # xr = x_rot*norm(p2-p3)
                    if verify_plot:
                        x = [0, yr[0]] + p1[0]
                        y = [0, yr[1]] + p1[1]
                        z = [0, yr[2]] + p1[2]
                        ax.plot(x, y, z, colors[0])
                    if verify:
                        # cox_err[t, joint/points_per_leg] = norm(p2-(yr+p1))
                        error[t, joint] = norm(p2-(yr+p1))
                        #print(name_vector[joint] + ' error', error[t, joint])
                    # x = [0, xr[0]] + p2[0]
                    # y = [0, xr[1]] + p2[1]
                    # z = [0, xr[2]] + p2[2]
                    # ax.plot(x, y, z)

                    


                eul = np.array([eul[eul_coxa_order[0]], eul[eul_coxa_order[1]], eul[eul_coxa_order[2]]]) # reordering the euler angles according to the order in coxa_names
                
                count = 0
                for coxa_name in coxa_name_vector:
                    datamat_j[coxa_name][t+1] = eul[count]
                    count += 1

            elif not last_point_name in name_vector[joint]:
                p1 = data[t, joint - 1, :]
                p2 = data[t, joint, :]
                p3 = data[t, joint + 1, :]

                # Calculate vectors of the segments around the joint (pointing away from the joint)
                v1b = np.array(p1 - p2)                 
                v2b = np.array(p3 - p2)

                

                # Project the vectors onto the plane of the leg
                # if joint%points_per_leg == 1:
                #     v1 = v1b
                # else:
                #     v1 = find_proj(v1b, n) 
                # v2 = find_proj(v2b, n)

                v1 = v1b
                v2 = v2b

                # Find the simple angle (in the plane) between the vectors
                if project_1DOF:
                    eul = angle_between(v1, v2) 
                else:
                    eul = angle_between(v1b, v2b)
                
                if diff_filename is not None:
                    diff['eul'][t, joint] = eul

                # Allowing angles to be negative
                if eul_ref_type == 'coxa_x' and negative_angles and 'F_COXA_FEMUR' in name_vector[joint]:
                    x_c = np.dot(np.linalg.inv(R_cl), v2)[0]
                    
                    if x_c < 0:
                        eul = - eul

                    if diff_filename is not None:
                        diff['dist'][t, joint] = x_c
                elif eul_ref_type == 'normal' and negative_angles:
                    n_lf = np.dot(n, np.linalg.inv(R_fl)) # The normal to the leg plane in the lab frame
                    
                    angle_normal = v2 #np.cross(v1, v2) 
                    angle_normal = angle_normal/norm(angle_normal)
                    an_lf = np.dot(angle_normal, np.linalg.inv(R_cl))
                    if diff_filename is not None:
                        diff['dist'][t, joint] = an_lf[0]
                    angle_normal = v2b #np.cross(v1b, v2b)
                    angle_normal = angle_normal/norm(angle_normal)
                    an_lf = np.dot(angle_normal, np.linalg.inv(R_cl))
                    if diff_filename is not None:
                        diff['eul_ref'][t, joint] = an_lf[2]
                    s = np.dot(n, angle_normal)
                    if diff_filename is not None:
                        diff['diff'][t,joint] = s
                    if t == 0:
                        sign_ref[joint] = np.sign(s)
                        eul = first_angle_assumption[joint]*eul
                    if np.sign(s) == sign_ref[joint]:
                        eul = first_angle_assumption[joint]*eul
                    elif np.sign(s) == -sign_ref[joint]:
                        eul = - first_angle_assumption[joint]*eul
                elif eul_ref_type == 't-1' or eul_ref_type == 'first':
                    if eul_ref[joint] is not None: #and (joint == 1 or joint == 16):
                        
                            
                        if not (eul == 0 or eul > np.pi/2):
                            # Femur vector from t-1 projected onto the current leg plane
                            if eul_ref_type == 't-1':
                                v2b_t_ = data[t - 1, joint + 1, :] - data[t - 1, joint, :]
                            elif eul_ref_type == 'first':
                                v2b_t_ = data[int(t_ref[joint]), joint + 1, :] - data[int(t_ref[joint]), joint, :]
                                
                            else:
                                print('WARNING: eul_ref_type = ' + str(eul_ref_type) + ' not known.')
                            
                            v2_t_ = find_proj(v2b_t_, n)
                            distance_traveled = norm(v2_t_ - v2)
                            deul = 2*np.arcsin(distance_traveled/2)
                            if diff_filename is not None:
                                diff['diff'][t, joint] = deul - abs(abs(eul) - eul_ref[joint]) # Collecting for analysis
                                diff['eul_ref'][t, joint] = eul_ref[joint]
                                diff['dist'][t, joint] = distance_traveled
                            if negative_angles:
                                if eul_ref_type == 't-1' and abs(abs(abs(eul) - eul_ref[joint]) - deul) > deul_error_margin :
                                    eul = - eul
                                elif eul_ref_type == 'first' and first_angle_assumption[joint]*(abs(abs(eul) - eul_ref[joint]) - deul) < 0 and abs(abs(abs(eul) - eul_ref[joint]) - deul) > deul_error_margin: # Use the sign of the difference
                                    eul = - eul
                        else:
                            if diff_filename is not None:
                                diff['diff'][t, joint] = 0 # Collecting for analysis
                                diff['eul_ref'][t, joint] = eul_ref[joint]
                                diff['dist'][t, joint] = distance_traveled


                    # Calculating reference angle for angle sign calculations
                    if abs(eul) > 0: #and (joint == 1 or joint == 16):
                        if eul_ref[joint] is None:
                            eul = first_angle_assumption[joint]*eul
                            if abs(eul) < np.pi/2:
                                eul_ref[joint] = eul
                            
                                if eul_ref_type == 'first':
                                    t_ref[joint] = int(t)
                        elif eul_ref_type == 't-1' and abs(eul) < np.pi/2 :
                            eul_ref[joint] = eul

                if (verify or verify_plot):
                    if joint%points_per_leg == 1:
                        R_fe = T.rotation_matrix(np.pi/2 - eul, R_f[:,2])[:3, :3]
                        R_v = R_fe
                    elif joint%points_per_leg == 2:
                        R_ti = T.rotation_matrix(-(np.pi - eul), R_f[:,2])[:3, :3]
                        R_v = np.dot(R_ti, R_fe)
                    elif joint%points_per_leg == 3:
                        R_ta = T.rotation_matrix(np.pi - eul, R_f[:,2])[:3, :3]
                        R_v = np.dot(R_ta, R_v)
                    point = np.dot(R_v, point_x)
                    point = point/norm(point)*norm(p2-p3)
                    p_rot = np.dot(R, point)
                    p_rot = p_rot/norm(p_rot)*norm(p2-p3)
                    
                    if verify:
                        # fem_err[t, joint/points_per_leg - 1] = norm(p3 - (fe_rot + p2))
                        # fem_plane_err[t, joint/points_per_leg - 1] = norm(find_proj(p2, n) - fe_rot)# find_dist(fe_rot, n))
                        error[t, joint] = norm(p3 - (p_rot + point_old))
                        if joint%5 < 2:
                            plane_error[t, joint/points_per_leg - 1] = norm(find_proj(p3-p2, n) - p_rot)
                            #print('plane error', plane_error[t, joint/points_per_leg -1])
                        #print(name_vector[joint] + ' error', error[t, joint])

                    if verify_plot: #and joint%5 < 2:
                        # point = find_proj(p3-p2, n)
                        # x = [0, point[0]] + point_old[0]#+ py[0] + p1[0]
                        # y = [0, point[1]] + point_old[1]#py[1] + p1[1]
                        # z = [0, point[2]] + point_old[2]#py[2] + p1[2]
                        # ax.plot(x, y, z)

                        x = [0, p_rot[0]] + point_old[0] #+ yr[0] + p1[0]
                        y = [0, p_rot[1]] + point_old[1] #yr[1] + p1[1]
                        z = [0, p_rot[2]] + point_old[2] #yr[2] + p1[2]
                        ax.plot(x, y, z, colors[0])

                        if joint == 28:
                            #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  
                            for i in range(no_legs):
                                plt.gca().set_color_cycle(colors[6])
                                d = data[t,i*points_per_leg:((i+1)*points_per_leg), :]
                                x = d[:,0]
                                y = d[:,1]
                                z = d[:,2]
                                ax.scatter(x,y,z)
                    
                    
                            for i in range(no_legs):
                                #plt.gca().set_color_cycle(colors[6])
                                d = data[t,i*points_per_leg:((i+1)*points_per_leg), :]
                                x = d[:,0]
                                y = d[:,1]
                                z = d[:,2]
                                ax.plot(x,y,z, '--', color = colors[6])
                            plt.legend(['Data', 'Recreation using joint anlges'], fontsize = 11)#['rotated coxa', 'rotated femur', 'rotated tibia', 'rotated tarsus', 'data'])
                            
                            plt.show()
                            # x = [0, yr[0]] + p1[0]
                            # y = [0, yr[1]] + p1[1]
                            # z = [0, yr[2]] + p1[2]
                            # ax.plot(x, y, z, color = colors[1])

                            # x = [0, p_rot[0]] + p2[0] #+ yr[0] + p1[0]
                            # y = [0, p_rot[1]] + p2[1] #yr[1] + p1[1]
                            # z = [0, p_rot[2]] + p2[2] #yr[2] + p1[2]
                            # ax.plot(x, y, z, color = colors[5])
                        # if joint%5 == 3:
                        #     plt.show()
                    point_old = p_rot + point_old

                        

                eul = np.pi - eul # The joint angle is considered as 0 when the leg is straight

                datamat_j[name_vector[joint]][t+1] = eul

    datamat_j = np.unwrap(datamat_j.iloc[1:, :], axis = 0)
    
    # Correction if the file jumps by 2pi at some point
    if ld_correction is not None:
        #plot_large_difference = True
        datamat_j = correct_ld(datamat_j, ld_correction = ld_correction, plot_large_difference = plot_large_difference, fps = 1000/time_step, extended_name_vector = extended_name_vector)
            
    # Correcting if a joint's mean position is further than pi from 0
    for joint in range(datamat_j.shape[1]):
        if abs(np.mean(datamat_j[:, joint])) > np.pi:
            datamat_j[:, joint] = datamat_j[:, joint] - np.sign(np.mean(datamat_j[:, joint]))*2*np.pi

    datamat_j = pd.DataFrame(datamat_j, columns = extended_name_vector)

    # Print joint angles before interpolation
    if joint_angle_filename is not None:
        print('Saving joint angles to file...')
        datamat_j.to_csv(joint_angle_filename, index=False)

    if diff_filename is not None:
        print('Pickling diff dictionary...')
        import pickle

        f = open(diff_filename, 'w')   
        pickle.dump(diff, f)  
        f.close()      

    if plane_dist_filename is not None:
        print('Pickling plane distance dictionary...')
        import pickle
        f = open(plane_dist_filename, 'w')   
        pickle.dump(plane_dist, f)  
        f.close()  

    if segment_length_filename is not None:
        print('Saving leg segment lengths...')
        np.savetxt(segment_length_filename, segment_lengths)

    if plane_angle_name is not None:
        print('Pickling plane angles...')
        import pickle
        f = open(plane_angle_name, 'w')   
        pickle.dump(plane_ang, f)  
        f.close() 
    
    if error_name is not None:
        print('Pickling error estimate...')
        import pickle
        error_names = []

        for i in range(no_legs):
            error_names = np.append(error_names, np.append(name_vector[(points_per_leg*i + 1):(points_per_leg*i + 5)], name_vector[(points_per_leg*i)]))
        error = pd.DataFrame(error, columns = error_names)
        names = [side + leg for side in side_names for leg in leg_names]
        plane_error = pd.DataFrame(plane_error, columns = names)
        f = open(error_name, 'w')
        pickle.dump({'error': error, 'plane error': plane_error}, f)
        f.close()
    
    if interpolate:
        print('Interpolating...')
        datamat_j = interpolate_time(datamat_j, time_step = time_step, webots_factor = webots_factor, webots_time_step = webots_time_step)
        if wt_get_to_position > 0:
            datamat_j = into_position(datamat_j, wt_get_to_position = wt_get_to_position, webots_time_step = webots_time_step)
    
    datamat_j = datamat_j.reset_index(drop=True)


#     print('Done')
    return datamat_j


### Data manipulation functions
def get_name_vector(joint_names = ['THORAX_COXA', 'COXA_FEMUR', 'FEMUR_TIBIA', 'TIBIA_TARSUS'], 
                  coxa_names = ['LD', 'T_1', 'T_2'], side_names = ['L', 'R'], 
                  leg_names = ['F', 'M', 'H'], 
                  last_point_name = 'CLAW'
                  ):
    # This function creates a vector of names of the joints
    name_vector = []
    extended_name_vector = []
    segment_points = np.append(joint_names, last_point_name)
    for side in side_names:
        for leg in leg_names:
            for segment_point in segment_points:
                name_vector = np.append(name_vector, side + leg + '_' + segment_point)
                if not last_point_name in segment_point:
                    if not joint_names[0] in segment_point:
                        extended_name_vector = np.append(extended_name_vector, side + leg + '_' + segment_point)
                    else:
                        for coxa_name in coxa_names:
                            extended_name_vector = np.append(extended_name_vector, side + leg + '_' + segment_point + '_' + coxa_name)

    return name_vector, extended_name_vector

def axes_swap(points3d):
    # Moving the points to be approximately around zero, swapping and mirroring some axes
    points3d -= np.median(points3d.reshape(-1, 3), axis=0)
    pts_t = points3d.copy()
    # tmp = pts_t[:,:,1].copy()
    # pts_t[:,:,1] = pts_t[:,:,2].copy()
    # pts_t[:,:,2] = tmp
    # tmp = pts_t[:,:,2].copy()
    # pts_t[:,:,2] = pts_t[:,:,0].copy()
    # pts_t[:,:,0] = tmp
    #pts_t[:,:,2] *=-1
    pts_t[:,:,1] *=-1
    return pts_t

def correct_ld(datamat_j, ld_correction = np.pi/2, fps = 100, plot_large_difference = False, extended_name_vector = None):
    ''' 
        This function goes through the data and corrects when an angle jumps by 2pi during the course of a file.
    '''
    if plot_large_difference:
        import matplotlib.pyplot as plt
    
    for joint in range(datamat_j.shape[1]):
        if plot_large_difference:
            t = np.arange(0, len(datamat_j)/float(fps), 1./fps)
            plt.figure()
            plt.plot(t, datamat_j[:,joint]*180/np.pi)
            change = False
        
        resolved = False
        count = 0
        while not resolved:
            d = np.diff(datamat_j[:,joint])
            big_diff = abs(d) > ld_correction # Find big jumps in the angles
            a = np.arange(0,len(d), 1)
            idx = a[big_diff]

            if count == 0 and plot_large_difference:
                plt.plot(t[idx], datamat_j[idx, joint], '*')
                count += 1

            if len(idx) > 0:
                mean_before_first = np.mean(datamat_j[1:idx[0], joint]) # Find the mean before the first big jump
                too_far_away = abs(datamat_j[:, joint] - mean_before_first) > ld_correction*2 # Find the points that are too far away from that point
                if sum(too_far_away) > 1: # Correct them by 2pi if there are more than one outlier
                    datamat_j[too_far_away, joint] = datamat_j[too_far_away, joint] - 2*np.pi*np.sign(datamat_j[too_far_away, joint])
                    if plot_large_difference:
                        change = True
                else:
                    resolved = True
            else:
                resolved = True

        if plot_large_difference:
            if change:
                plt.plot(t, datamat_j[:,joint]*180/np.pi)
                #plt.title(extended_name_vector[joint])
                plt.xlabel('Time (s)', fontsize = 13)
                plt.ylabel(r'Angle ($\degree$)', fontsize = 13)
                plt.legend(['Original signal', 'Points of large difference', 'Corrected signal'], fontsize = 13)
                plt.show()
            plt.close()

    return datamat_j

def estimate_segment_length(data, no_legs = 6, points_per_leg = 5):
    # Estimate leg segment length for each leg
    segment_lengths = np.zeros([data.shape[1]])
    count = 0
    for leg in range(no_legs):
        for joint in range(points_per_leg-1):
            segment = data[:,leg*points_per_leg + joint, :] - data[:,leg*points_per_leg + joint + 1, :]
            
            segment_norms = np.array([norm(v) for v in segment])
            segment_length = np.mean(segment_norms, axis = 0)
            segment_lengths[count] = segment_length
            count += 1
    return segment_lengths

def find_median_point(data, segment, no_legs = 6, points_per_leg = 5):
    # This function finds all the points of a particular joint (segment) and returns an array of their median position.
    v = np.arange(0,no_legs,1)
    segment_points = data[:,v*points_per_leg + segment,:]
    median_points = np.median(segment_points, axis = 0)
    return median_points

def fix_coxa(data, segment = 0, no_legs = 6, points_per_leg = 5):
    # This function fixes the coxa (joint 1) position in the data.
    median_points = find_median_point(data,  segment = 0, no_legs = no_legs, points_per_leg = points_per_leg)
    v1 = median_points[0] - median_points[1]
    v2 = median_points[2] - median_points[1]
    a1 = angle_between(v1, v2)
    v1 = median_points[3] - median_points[4]
    v2 = median_points[5] - median_points[4]
    a2 = angle_between(v1, v2)
    print('COXA ANGLE DIFFERENCE', a1-a2)
    v = np.arange(0,no_legs,1)
    data[:,v*points_per_leg,:] = np.repeat(median_points[np.newaxis, :, :], len(data), axis = 0)  
    return data


### Functions for finding the rotation matrix in the nearest plane
def find_plane_normal(P):
    # Find the normal vector to the plane closest to the N points in P. 
    # P is a 3XN matrix where the lines are the 3D coordinates 
    # of the points with.
    u, s, v = np.linalg.svd(P)
    return u[:,-1]

def plane(x, y, params):
    a = params[0]
    b = params[1]
    c = params[2]
    z = a*x + b*y + c
    return z

def error(params, points):
    result = 0
    for (x,y,z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff**2
    return result

def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]

def get_plane_normal(points):
    '''
    Get the normal to the best plane through "points". Points is an array of 3d points.
    '''
    fun = functools.partial(error, points=points)
    params0 = [0, 0, 0]
    res = scipy.optimize.minimize(fun, params0)

    a = res.x[0]
    b = res.x[1]
    c = res.x[2]

    xs, ys, zs = zip(*points)
    normal = np.array(cross([1,0,a], [0,1,b]))
    point  = np.array([0.0, 0.0, c])
    
    return normal, point

def find_fly_R(data, g_ref, f_ref):
    '''
    This function finds the fly coordinate system wrt the lab frame 
        g_ref - gravity reference
        f_ref - fly forward reference
    '''

    if 'coxa' in f_ref or 'coxa' in g_ref:
        if len(data.shape) == 3:
            coxa_points = copy.deepcopy(data[0, np.arange(0,30,5), :])
        elif len(data.shape) == 2:
            coxa_points = copy.deepcopy(data[np.arange(0,30,5), :])

    # Calculating y (gravity) direction
    if g_ref == 'y':
        y = np.array([0,1,0])
    elif g_ref == 'coxa':
        ref_coxa, point = get_plane_normal(coxa_points)
        y = - ref_coxa
    else:
        print('ERROR: gravity reference ' + str(g_ref) + ' unknown.')

    # Calculating x (forward) direction

    if f_ref == 'HMcoxa':
        x_temp = [coxa_points[1, :] - coxa_points[2, :], coxa_points[4, :] - coxa_points[5, :]]
        z = [np.cross(x_temp[0],y), np.cross(x_temp[1],y)] #[np.cross(y, x_temp[0]), np.cross(y, x_temp[1])]#
        x = [np.cross(y,z[0]), np.cross(y,z[1])] # [np.cross(z[0],y), np.cross(z[1],y)] #
        y = y/norm(y)
        R = [[], []]
        for i in range(2):
            x[i] = x[i]/norm(x[i])
            z[i] = z[i]/norm(z[i])
            R[i] = np.zeros((3,3))
            R[i][:,0] = x[i]
            R[i][:,1] = y
            R[i][:,2] = z[i]  
            
    else:

        if f_ref == 'x':
            x_temp = np.array([1,0,0])
        elif f_ref == 'coxa':
            x_temp = np.mean(coxa_points[[0,3], :], axis = 0) - np.mean(coxa_points[[2, 5], :], axis = 0) # Front points - hind points
        else:
            print('ERROR: gravity reference ' + str(g_ref) + ' unknown.')

        # Calculating z (left to right) direction
        z = np.cross(x_temp,y) # np.cross(y, x_temp)#
        x = np.cross(y,z)#z, y)#

        x = x/norm(x)
        y = y/norm(y)
        z = z/norm(z)
        
        R = np.zeros((3,3))
        R[:,0] = x
        R[:,1] = y
        R[:,2] = z   
    
    return R


def find_proj(v,n):
    # Find the projection of v onto the plane normal to n.
    return v - np.dot(v, n)/(norm(n)**2)*n

def find_dist(v,n):
    # Find the distance of v from the plane normal to n.
    return np.dot(v, n)/norm(n)

def find_rotation_matrix(n, v1, v2):
    # Find the rotation matrix corresponding to a z-axis along the normal vector n, the y-axis along the 
    # projection of v1 to the plane defined by n and the x-axis orthogonal to both pointing in the direction
    # of v2.
    y = v1#find_proj(v1,n)
    x_temp = v2#find_proj(v2, n)
    z = np.cross(x_temp, y)
    x = np.cross(y, z)
    if norm(x) > 0:
        x = x/norm(x)
    if norm(y) > 0:
        y = y/norm(y)
    if norm(z) > 0:
        z = z/norm(z)
    R = np.zeros([3,3])
    R[:,0] = x
    R[:,1] = y
    R[:,2] = z
    return R


### Functions for finding the 1DOF angles
def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


### Functions for getting the file ready for Webots
def interpolate_time(datamat_j, time_step = 50, webots_factor = 0.1, webots_time_step = 2):
    '''
    This function interpolates the data to match the timestep in Webots.
    Parameters:
        time_step - time step between datapoints in real data (in ms)
        webots_factor - the factor by which Webots multiplies time
        webots_time_step - the time step in Webots (as it is inputted)
    '''
    total_time = len(datamat_j)*time_step
    wts = webots_time_step*webots_factor # The real Webots time step in ms
    new_no_points = total_time/wts
    t_new = np.arange(0, total_time, wts)
    t_old = np.arange(0, total_time, time_step)
    new_data = np.zeros([int(new_no_points), datamat_j.shape[1]])

    for i in range(datamat_j.shape[1]):
        new_data[:, i] = np.interp(t_new, t_old, datamat_j.iloc[:,i])
        
    datamat_j = pd.DataFrame(new_data, columns = datamat_j.columns)
    datamat_j['TIME'] = t_new
    
    return datamat_j

def into_position(datamat_j, wt_get_to_position = 1000, webots_time_step = 2):
    '''
    Duplicate the first set of angle values to give the simulation time to get into 
    a normal first position before starting to move.
    Parameters:
        wt_get_to_position - the time (in Webots time) needed to get safely into position. Default is 1000.
        webots_time_step - the time step in Webots (as it is inputted).
    '''
    
    no_frames = int(wt_get_to_position/webots_time_step) # Number of frames to get to basic position
    first_frame = np.array(datamat_j.iloc[1,:])
    first_frame = pd.DataFrame([first_frame], columns = datamat_j.columns)
    f_f = copy.deepcopy(first_frame)
    for i in range(no_frames-1):
        first_frame = first_frame.append(f_f)
    datamat_j = first_frame.append(datamat_j)


    return datamat_j

def filter_batch(pts, filter_indices=None, config=None, freq=None):
    '''Smooth the positional data (from Joao)'''
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
