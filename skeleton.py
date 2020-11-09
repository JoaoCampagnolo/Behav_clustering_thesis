
from enum import Enum

import numpy as np

num_cameras = 7


class Tracked(Enum):
    BODY_COXA = 0
    COXA_FEMUR = 1
    FEMUR_TIBIA = 2
    TIBIA_TARSUS = 3
    TARSUS_TIP = 4
    ANTENNA = 5
    STRIPE = 6


tracked_points = [Tracked.BODY_COXA, Tracked.COXA_FEMUR, Tracked.FEMUR_TIBIA, Tracked.TIBIA_TARSUS, Tracked.TARSUS_TIP,
                  Tracked.BODY_COXA, Tracked.COXA_FEMUR, Tracked.FEMUR_TIBIA, Tracked.TIBIA_TARSUS, Tracked.TARSUS_TIP,
                  Tracked.BODY_COXA, Tracked.COXA_FEMUR, Tracked.FEMUR_TIBIA, Tracked.TIBIA_TARSUS, Tracked.TARSUS_TIP,
                  Tracked.ANTENNA,
                  Tracked.STRIPE, Tracked.STRIPE, Tracked.STRIPE,
                  Tracked.BODY_COXA, Tracked.COXA_FEMUR, Tracked.FEMUR_TIBIA, Tracked.TIBIA_TARSUS, Tracked.TARSUS_TIP,
                  Tracked.BODY_COXA, Tracked.COXA_FEMUR, Tracked.FEMUR_TIBIA, Tracked.TIBIA_TARSUS, Tracked.TARSUS_TIP,
                  Tracked.BODY_COXA, Tracked.COXA_FEMUR, Tracked.FEMUR_TIBIA, Tracked.TIBIA_TARSUS, Tracked.TARSUS_TIP,
                  Tracked.ANTENNA,
                  Tracked.STRIPE, Tracked.STRIPE, Tracked.STRIPE]

limb_id = [0, 0, 0, 0, 0,
           1, 1, 1, 1, 1,
           2, 2, 2, 2, 2,
           3,
           4, 4, 4,
           5, 5, 5, 5, 5,
           6, 6, 6, 6, 6,
           7, 7, 7, 7, 7,
           8,
           9, 9, 9]

__limb_visible_left = [True, True, True, True, True,
                       False, False, False, False, False] 

__limb_visible_right = [False, False, False, False, False,
                        True, True, True, True, True]

# fix this
bones = [[0, 1], [1, 2], [2, 3], [3, 4],
         [5, 6], [6, 7], [7, 8], [8, 9],
         [10, 11], [11, 12], [12, 13], [13, 14],
         [16, 17], [17, 18],
         [19, 20], [20, 21], [21, 22], [22, 23],
         [24, 25], [25, 26], [26, 27], [27, 28],
         [29, 30], [30, 31], [31, 32], [32, 33],
         [35, 36], [36, 37]]

# bones3d = [[15, 34], [15, 16], [34, 16]]
bones3d = [[15, 34]]

colors = [(255, 0, 0),
          (0, 0, 255),
          (0, 255, 0),
          (150, 200, 200),
          (255, 165, 0),
          (255, 255, 0),
          (255, 0, 255),
          (0, 255, 255),
          (150, 200, 200),
          (255, 165, 0)]

num_joints = len(tracked_points)
num_limbs = len(set(limb_id))


def is_body_coxa(joint_id):
    return tracked_points[joint_id] == Tracked.BODY_COXA


def is_coxa_femur(joint_id):
    return tracked_points[joint_id] == Tracked.COXA_FEMUR


def is_femur_tibia(joint_id):
    return tracked_points[joint_id] == Tracked.FEMUR_TIBIA


def is_tibia_tarsus(joint_id):
    return tracked_points[joint_id] == Tracked.TIBIA_TARSUS


def is_antenna(joint_id):
    return tracked_points[joint_id] == Tracked.ANTENNA


def is_stripe(joint_id):
    return tracked_points[joint_id] == Tracked.STRIPE


def is_tarsus_tip(joint_id):
    return tracked_points[joint_id] == Tracked.TARSUS_TIP


def get_limb_id(joint_id):
    return limb_id[joint_id]


def is_joint_visible_left(joint_id):
    return __limb_visible_left[get_limb_id(joint_id)]


def is_joint_visible_right(joint_id):
    return __limb_visible_right[get_limb_id(joint_id)]


def is_limb_visible_left(limb_id):
    return __limb_visible_left[limb_id]


def is_limb_visible_right(limb_id):
    return __limb_visible_right[limb_id]


def camera_see_limb(camera_id, limb_id):
    if camera_id < 3:
        return is_limb_visible_left(limb_id)
    elif camera_id == 3:
        return is_limb_visible_mid(limb_id)
    elif camera_id > 3:
        return is_limb_visible_right(limb_id)
    else:
        raise NotImplementedError


def camera_see_joint(camera_id, joint_id):
    if camera_id in [2, 4]:  # they cannot see the stripes
        return camera_see_limb(camera_id, limb_id[joint_id]) and not (tracked_points[joint_id] == Tracked.STRIPE)
    elif camera_id == 3:
        return camera_see_limb(camera_id, limb_id[joint_id]) and tracked_points[joint_id] != Tracked.BODY_COXA \
               and tracked_points[joint_id] != Tracked.COXA_FEMUR
    else:
        return camera_see_limb(camera_id, limb_id[joint_id])


bone_param = np.ones((num_joints, 2), dtype=float)
bone_param[:, 0] = 0.85
bone_param[:, 1] = 0.2
for joint_id in range(num_joints):
    if is_body_coxa(joint_id) or is_stripe(joint_id) or is_antenna(joint_id):
        bone_param[joint_id, 1] = 10000  # no bone

ignore_joint_id = [joint_id for joint_id in
                   range(num_joints) if
                   is_body_coxa(joint_id) or is_coxa_femur(joint_id) or is_antenna(joint_id)]

ignore_joint_id_wo_stripe = [joint_id for joint_id in
                             range(num_joints) if
                             is_body_coxa(joint_id) or is_coxa_femur(joint_id) or is_antenna(joint_id)]
pictorial_joint_list = [j for j in range(num_joints)]


skeleton_1 = [3,2,1,4,5,6,6,6,7,8]
skeleton_2 = [2,1,3,4,5,6,6,6,7,8]
skeleton_3 = [1,2,3,4,5,6,6,6,7,8]
skeleton_4 = [1,2,3,4,8,1,2,3,7,8]

