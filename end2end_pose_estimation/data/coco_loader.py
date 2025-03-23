# class for loading the coco data-set features extracted from the trained PAF network
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from constrained_pose_estimation.general_helper import helper_functions as hf
import constrained_pose_estimation.data.data_keys as data_keys
import random
from random import randint
from external.cocoapi.PythonAPI.pycocotools.coco import COCO
import sys
sys.path.append('/external/cocoapi/PythonAPI/')
from external.cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval


class coco_features_loader(Dataset):
    #unsure if mask_path is needed
    def __init__(self, data_dir, file_list_name, mask_path=None, batch_size=1, transform=None, is_lazy=True, shuffle=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir  # the path of the features directory
        self.batch_size = batch_size
        self.file_list_name = file_list_name
        self.transform = transform
        self.last_index = 0
        self.epoch = -1

        if 'val' in data_dir:
            annotation_file = '/media/datasets/pose_estimation/MSCOCO_2017/annotations_trainval2017/annotations/person_keypoints_val2017.json'
            coco = COCO(annotation_file)
            # get all images containing persons
            catIds = coco.getCatIds(catNms=['person'])
            self.img_ids = coco.getImgIds(catIds=catIds)
            #self.img_ids = hf.get_file_list(file_list_name)  # get the list of the dataset image ids

        else:
            #self.img_ids = hf.get_file_list(file_list_name)  # get the list of the dataset image ids
            annotation_file = '/media/datasets/pose_estimation/MSCOCO_2017/annotations_trainval2017/annotations/person_keypoints_train2017.json'
            coco = COCO(annotation_file)
            # get all images containing persons
            catIds = coco.getCatIds(catNms=['person'])
            self.img_ids = coco.getImgIds(catIds=catIds)
        if shuffle:
            import random
            random.shuffle(self.img_ids)
        self.frames_features = None
        if not is_lazy:
            self.frames_features = self.load_frame_features(file_list_name)


    def reset(self):
        max_epochs = 3 #move to params
        self.last_index = 0
        self.epoch = (self.epoch+1)%max_epochs
    #load all frames in one bulk
    def load_frame_features(self, file_list_name):
        frames_features = []
        count = 0
        for img_id in self.img_ids:
            frame_features = self.load_next_sample_features(img_id)
            if frame_features is None:
                continue
            frames_features.append(frame_features)
            #if count == 500:
            #    break
            count += 1
        return frames_features


    def load_next_sample_features(self, img_id):
        #img_name = img_id.split('.')[0]
        img_id = str(img_id)
        img_name = img_id.rjust(12-len(img_id)+len(img_id), '0')
        #print(img_name)
        #if 'val' in self.data_dir: # for validation, take the features of non augmented images
        #    path = self.data_dir.split('epoch')[0]
        #elif 'epoch_0' in self.data_dir or 'epoch_3' in self.data_dir: # if the user explicitly specified the path
        path = self.data_dir
        #else: # for training, take augmented samples from different epochs
        #    path = self.data_dir%(self.epoch)
        ext = data_keys.feature_ext
        data = dict()
        features_path = '%s%s%s%s' % (path, data_keys.features_prefix, img_name, ext)
        if not os.path.isfile(features_path): #or not os.path.isfile('%s%s%s%s' % (path, data_keys.separate_heatmap_gt_prefix, img_name, ext)):
            return None
        if not 'val' in path:
            #pass
            current_gt = np.load('%s%s%s%s' % (path, data_keys.separate_heatmap_gt_prefix, img_name, ext))
            # current_paf_gt = np.load('%s%s%s%s' % (path, data_keys.separate_paf_gt_prefix, img_name, ext))
            # current_paf_gt = torch.from_numpy(current_paf_gt).permute(3, 2, 0, 1)
            # data[data_keys.paf_gt_key] = current_paf_gt
            # mask_path = '%s%s%s%s' % (path, data_keys.heatmap_mask_prefix, img_name, ext)
            # if not os.path.isfile(mask_path):
            #     return None
            # current_mask = np.load(mask_path)
            # data[data_keys.mask_path_key] = torch.from_numpy(current_mask)
            current_gt = torch.from_numpy(current_gt).permute(0, 3, 1, 2)
            data[data_keys.heatmap_gt_key] = current_gt
        else: # for the validation, this is just to get k(number of persons)
             pass
             #gt_path = '/media/pose_estimation/Experiments/coco_precomputed_features/coco/val_gt_heatmaps/'
             #gt_name = '%s%s%s%s' % (gt_path, data_keys.separate_heatmap_gt_prefix, img_name, ext)
             #current_gt = np.load(gt_name)



        current_paf_pred = np.load('%s%s%s%s' % (path, data_keys.paf_pred_prefix, img_name, ext))
        current_heatmap_pred = np.load('%s%s%s%s' % (path, data_keys.heatmap_pred_prefix, img_name, ext))
        current_feature = np.load(features_path)
        #if 'val' in path:
        data[data_keys.paf_pred_key] = current_paf_pred[-1]
        data[data_keys.heatmap_pred_key] = current_heatmap_pred[-1]
        #else:
        #data[data_keys.paf_pred_key] = torch.from_numpy(current_paf_pred)
        #data[data_keys.heatmap_pred_key] = torch.from_numpy(current_heatmap_pred)
        data[data_keys.features_key] = torch.from_numpy(current_feature)
        data[data_keys.img_id] = img_name
        return data

    def __getitem__(self, idx):
        sample = self.frames_features[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_next_item(self):
        if self.last_index >= len(self.img_ids)-1:
            return None
        self.last_index += 1
        if self.frames_features is None: #lazy loading
            item = None
            while item is None:
                item = self.load_next_sample_features(self.img_ids[self.last_index-1])
                self.last_index += 1
                if self.last_index >= len(self.img_ids) - 1:
                    return None
            return item
        else:
            return self[self.last_index-1]


