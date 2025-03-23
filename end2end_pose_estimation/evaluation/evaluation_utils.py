import numpy as np
import os.path as path
from easydict import EasyDict as edict
import cv2 as cv
import torch
#from external.cocoapi.PythonAPI.pycocotools.coco import COCO
#import sys
#sys.path.append('/external/cocoapi/PythonAPI/')
#from external.cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval
from numpy import linalg as LA

from constrained_pose_estimation.optimization.hungarian import Hungarian
import pickle


def preprocess(bgr_image, scale=1.0, stride=8, pad_value=128):
    out_img = cv.resize(bgr_image, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    out_img, pad = pad_right_down_corner(out_img, int(stride), int(pad_value))
    out_img = np.transpose(np.float32(out_img[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
    return out_img, pad


def pad_right_down_corner(img, stride, pad_value):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img.copy()
    if len(img_padded.shape) < 3:
        img_padded = np.repeat(img_padded[:, :, np.newaxis], 1, axis=2)

    pad_up = np.tile(img_padded[0:1, :, :] * 0 + pad_value, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + pad_value, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + pad_value, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + pad_value, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad




sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
vars = (sigmas * 2)**2

#from external.cocoapi.PythonAPI.pycocotools.coco import COCO
#coco_gt = COCO('/media/datasets/pose_estimation/MSCOCO_2017/annotations_trainval2017/annotations/person_keypoints_val2017.json')
def calculate_scores(img_name, detected_poses):
    full_name = '/media/pose_estimation/Experiments/coco_precomputed_features/coco/val_gt_poses/' + img_name + '.pickle'

    with open(full_name, 'rb') as handle:
        gt_pose_obj = pickle.load(handle)

    gt_poses = gt_pose_obj.poses
    person_scales = gt_pose_obj.scales
    pairwise_scores = calculate_pairwise_oks_similarity(gt_poses, detected_poses, person_scales)

    min_scores = np.min(pairwise_scores, axis=0) # or maybe use the assignment to deduce the assignment!

    return min_scores
    #assignment = calc_assignment(pairwise_scores)

    #sorted, indices = torch.sort(assignment[:, 0])


    #J = detected_poses[0].shape[0]
    #for k in range(len(detected_poses)):
    #    nan_ind = np.argwhere(np.isnan(poses[k][:][:]))
    #    detected_poses[k][nan_ind] = 0
    #    poses[k][nan_ind] = 0
    #    dist_sqr = (detected_poses[k]-poses[k])**2
    #    dist_sqr /= 2*scale**2*sigma**2
    #    exp_values = np.exp(-dist_sqr/(2*scale**2*var))
    #    effective_len = nan_ind.shape[0]
    #    if(effective_len > 0):
    #        exp_values = (np.sum(exp_values)-nan_ind.shape[0])/(J-nan_ind.shape[0])
    #        scores[k] = exp_values
    #    else:
    #        scores[k] = 0
    #print(scores)
    #return scores


def calculate_pairwise_oks_similarity(dt_poses, gt_poses, person_scales):
    nu_joints = gt_poses.shape[1]
    nu_persons =  gt_poses.shape[0]
    dt_poses = dt_poses.view(nu_persons, nu_joints+1, dt_poses.shape[1], dt_poses.shape[2])
    oks_similarity = np.zeros((len(gt_poses), len(dt_poses)))
    for k in range(len(gt_poses)):
        for k_hat in range(len(dt_poses)):
            factor = 0
            for j in range(gt_poses[0].shape[0]):
                current_sim = calc_single_oks(gt_poses[k][j], dt_poses[k_hat][j][0:2], vars[j], person_scales[k])
                print ('****sim' + str(current_sim))
                oks_similarity[k, k_hat] += current_sim if current_sim>0 else 0
                if(current_sim>=0):
                    factor += 1
            if(factor >0):
                oks_similarity[k, k_hat] /= factor
    return oks_similarity

hungarian = Hungarian(is_profit_matrix=True)
def calc_assignment(similarity_scores):
    hungarian.calculate(similarity_scores)
    res = hungarian.get_results()
    return res

# calculate the oks distance per one joint, gt_joint is the ground truth location (a coordinate pair) of joint j
def calc_single_oks(gt_joint, dt_joint, var, scale):
    # what should we do during similarity calculation if the joints are occluded
    nan_ind = np.argwhere(np.isnan(gt_joint))
    if len(nan_ind)>1:
        return -1
    dist_sqr = (LA.norm(gt_joint - dt_joint))**2
    dist_sqr = dist_sqr/(2*scale**2*var)
    exp_value = np.exp(-dist_sqr)
    return exp_value

