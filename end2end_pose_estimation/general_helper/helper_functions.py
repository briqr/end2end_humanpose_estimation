import pickle
import numpy as np
import torch
import constrained_pose_estimation.evaluation.evaluation_utils as eu
import cv2 as cv
from easydict import EasyDict as edict
import yaml
import json
def get_file_list(img_list_path):
    with open(img_list_path) as f:
        img_ids = f.readlines()
    img_ids = [x.strip() for x in img_ids]
    return img_ids

def get_file_list_zero_stripped(img_list_path):
    with open(img_list_path) as f:
        img_ids = f.readlines()
    img_ids = [x.strip() for x in img_ids]
    img_ids = [x.rstrip('.jpg') for x in img_ids]
    img_ids = [int(x) for x in img_ids]
    #img_ids = [x.lstrip('0') for x in img_ids]
    return img_ids


def parse_params(params_path):
    with open(params_path, 'r') as stream:
        params = edict(yaml.load(stream))
    return params

def get_gt_poses(full_name):
    #full_name = '/media/pose_estimation/Experiments/coco_precomputed_features/coco/val/' + img_name +'.pickle'
    with open(full_name, 'rb') as handle:
        gt_pose_obj = pickle.load(handle)
    return gt_pose_obj


def save_poses(result_path, all_poses, im_ids):
    with open(result_path, 'w') as outfile:
        id = 0
        for current_pose in all_poses:
            for k in range(current_pose.shape[0]):
                pose_obj = dict()
                keypoints = current_pose[k,0:17,:]
                keypoints_confidence = np.zeros((17, 3))
                keypoints_confidence[:,0:2] = keypoints
                pose_obj['keypoints'] = keypoints_confidence.flatten().tolist()

                pose_obj['image_id'] = int(im_ids[id].lstrip('0'))
                pose_obj['category_id'] = 1
                pose_obj['score'] = 1
                json.dump(pose_obj, outfile)
            print ('im id' + str(id))
            id += 1


#img_ids = np.load('/media/pose_estimation/image_names.npy')
def get_poses(res):
    stride = 8
    start = stride/2-0.5
    nu_persons = len(res)
    J = res[0].shape[0]
    all_poses = np.zeros((nu_persons, res[0].shape[0], 3)) #k*J*2 coordinates
    for k in range(nu_persons):
        poses = torch.zeros(J, 3).int()
        for j in range(J):
            ind = np.argmax(res[k][j])
            ind = np.unravel_index(ind, res[0][0].shape)
            ind1 = ind[0]*stride + start
            ind2 = ind[1]*stride + start
            ind = [ind1, ind2]
            tuple = np.concatenate((np.asarray(ind).astype(int), [0])) # the zero is the visibility indicator, which is an expected number in the result file even though it is not being used
            poses[j] = torch.from_numpy(tuple)
        all_poses[k] = poses
    return all_poses

# don't resize the result
def get_poses_of_resized(res):
    nu_persons = len(res)
    J = res[0].shape[0]
    all_poses = np.zeros((nu_persons, res[0].shape[0], 3)) #k*J*2 coordinates
    for k in range(nu_persons):
        poses = torch.zeros(J, 3).int()
        for j in range(J):
            ind = np.argmax(res[k][j])
            ind = np.unravel_index(ind, res[0][0].shape)
            tuple = np.concatenate((np.asarray(ind).astype(int), [0])) # the zero is the visibility indicator, which is an expected number in the result file even though it is not being used
            poses[j] = torch.from_numpy(tuple)
        all_poses[k] = poses
    return all_poses


#heatmaps: the heatmaps separated per person
def resize_heatmap(bgr_image, heatmaps, is_cuda=False):
    resized_heatmaps = []
    input_image, pad = eu.preprocess(bgr_image, 1, 8, 8)
    input_image = input_image[:, [2, 1, 0], :, :]
    nu_persons = len(heatmaps)
    for k in range(nu_persons):
        heatmap = heatmaps[k]
        if is_cuda:
            heatmap = heatmap.detach().cpu().numpy()
        heatmap = np.transpose(heatmap, (1, 2, 0))
        heatmap = cv.resize(heatmap, (0, 0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
        heatmap = heatmap[:input_image.shape[2] - pad[2], :input_image.shape[3] - pad[3], :]
        heatmap = cv.resize(heatmap, (bgr_image.shape[1], bgr_image.shape[0]), interpolation=cv.INTER_CUBIC)
        resized_heatmaps.append(np.transpose(heatmap, (2, 0, 1)))


    return resized_heatmaps


def load_model(checkpoint_path, model):
    print('Loading checkpoint %s', checkpoint_path)
    model_dict = model.state_dict()
    saved_state_dict = torch.load(checkpoint_path)
    for name, param in saved_state_dict.items():
        print(name)
        if name in model_dict.keys():
            print(name)
            model_dict[name].copy_(param)
    model.load_state_dict(model_dict)


def load_pretrained(model):
    checkpoint_path = '/media/datasets/pose_estimation/PoseTracking_Data/V2/models/COCO/posetrack_order_six_stages/epoch_71.ckpt'
    saved_state_dict = torch.load(checkpoint_path)['state_dict']

    model_dict = model.state_dict()  # the current model

    for name, param in model_dict.items():
        if name in model_dict.keys():
            if('7' in name):
                print ('skipping ' + name)
                continue
            new_name = name[0:9] + '_6' + name[9:len(name)]
            existing_param = saved_state_dict[new_name]
            print(name)
            model_dict[name].copy_(existing_param)

