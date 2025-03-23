"""
 * posetrack18
 * Created on 1/10/19
 * Author: doering
"""

import numpy as np
from scipy.misc import imread
import os
import cv2 as cv

from pycocotools.coco import COCO

# from dataset_readers.infos.edges import get_posetrack_edges
# we can use this dataset as image dataset or as video dataset


class DatasetReader:

    def __init__(self,
                 data_dir,
                 annotation_dir,
                 type='video_dataset'):

        self.type = type

        self.data_dir = data_dir
        self.annotation_dir = annotation_dir

        self.sequence_names = self.__get_sequence_names__()
        self.num_sequences = len(self.sequence_names)

        self.num_parts = 15  # the dataset has 2 pseudo joints

        self.valid_idxs = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        self.part_labels = ['nose', 'upper_neck', 'head_top', 'left_shoulder',
                            'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                            'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                            'right_ankle']

        self.flipRef = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]

        # self.edges = get_posetrack_edges()

        # self.pairRef = [
        #     [2, 0],
        #     [0, 1],
        #     [1, 3], # neck ls
        #     [1, 4], # neck rs
        #     [3, 5], # ls le
        #     [4, 6], # rs re
        #     [3, 9], # ls, lh
        #     [4, 10], # rs, rh
        #     [5, 7], # le lw
        #     [6, 8], # re rw
        #     [9, 11], # lh lk
        #     [10, 12], # rh rk
        #     [11, 13], # lk la
        #     [12, 14] # rk ra
        # ]

    def __get_sequence_names__(self):
        return [os.path.splitext(anno_file)[0] for anno_file in  os.listdir(self.annotation_dir)]


class ImgDatasetReader(DatasetReader):

    def __init__(self,
                 data_dir,
                 annotation_dir,
                 annotated_only=False):

        super(ImgDatasetReader, self).__init__(data_dir, annotation_dir, type='image_dataset')
        self.annotated_only = annotated_only

        self.__init_dataset__()

    def __init_dataset__(self):

        self.img_anns = []
        self.img_info = []
        self.ignore_regions = []

        for seq_idx in range(self.num_sequences):
            annotation_path = os.path.join(
                self.annotation_dir,
                '{}.json'.format(self.sequence_names[seq_idx])
            )

            api = COCO(annotation_path)

            tmp_img_ids = api.getImgIds()
            tmp_img_info = api.loadImgs(tmp_img_ids)

            # self.img_info += tmp_img_info

            for img_idx, img_id in enumerate(tmp_img_ids):
                tmp_ann_ids = api.getAnnIds(img_id)
                img_info = tmp_img_info[img_idx]

                if self.annotated_only and not img_info['is_labeled']:
                    continue

                tmp_anns = api.loadAnns(tmp_ann_ids)
                self.img_info.append(img_info)

                if "ignore_regions_x" in img_info:
                    frame_regions = []
                    for region_idx in range(len( img_info['ignore_regions_x'])):
                        region = []
                        for point_idx in range(len(img_info['ignore_regions_x'][region_idx])):
                            region.append(
                                    [
                                        img_info['ignore_regions_x'][region_idx][point_idx],
                                        img_info['ignore_regions_y'][region_idx][point_idx]
                                    ]
                            )
                        frame_regions.append(region)

                    self.ignore_regions.append(frame_regions)
                else:
                    self.ignore_regions.append([])

                self.img_anns.append(tmp_anns)

        self.num_examples = len(self.img_anns)

    def __build_image_path__(self, sample_idx):
        img_info = self.img_info[sample_idx]

        path = os.path.join(self.data_dir, img_info['file_name'])

        return path

    def load_image(self, sample_idx):
        return imread(self.__build_image_path__(sample_idx), mode='RGB')

    def num_objects(self, sample_idx):
        anns = self.img_anns[sample_idx]

        return len(anns)

    def get_mask(self, sample_idx, height, width):

        ignore_region = self.ignore_regions[sample_idx]

        m = np.ones((height, width))
        if len(ignore_region) > 0:
            for region in ignore_region:
                if len(region) > 0:
                    m = cv.fillPoly(m, np.asarray([region]), 0)
        return m

    def get_keypoints(self, sample_idx, ann=None):

        anns = self.img_anns[sample_idx]

        num_persons = self.num_objects(sample_idx)
        kpts = np.zeros((num_persons, self.num_parts + 2, 3))

        for i in range(num_persons):
            kpts[i] = np.asarray(anns[i]['keypoints']).reshape([-1, 3])

        # kpts = kpts[:, self.valid_idxs]
        return kpts

    def get_anns(self, sample_idx):
        return None

    def get_joint_mask(self, sample_idx):

        joint_mask = np.ones(self.num_parts)

        return joint_mask


class VideoDatasetReader(DatasetReader):

    def __init__(self,
                 data_dir,
                 annotation_dir,
                 annotated_only=False):

        super(VideoDatasetReader, self).__init__(data_dir,
                                                 annotation_dir,
                                                 type='video_dataset')
        self.annotated_only = annotated_only

        self.__init_dataset__()

    def __init_dataset__(self):
        self.sequence_anns = []

        tot_frames = 0
        self.index = {}

        for seq_idx in range(self.num_sequences):
            seq_ignore_regions = []

            annotation_path = os.path.join(
                self.annotation_dir,
                '{}.json'.format(self.sequence_names[seq_idx])
            )

            sequence_img_infos = []
            sequence_anns = []

            api = COCO(annotation_path)

            tmp_img_ids = api.getImgIds()
            tmp_img_info = api.loadImgs(tmp_img_ids)

            # self.img_info += tmp_img_info
            fr_count = 0
            for img_idx, img_id in enumerate(tmp_img_ids):
                tmp_ann_ids = api.getAnnIds(img_id)
                img_info = tmp_img_info[img_idx]
                frame_regions = []

                if self.annotated_only and not img_info['is_labeled']:
                    continue

                tmp_anns = api.loadAnns(tmp_ann_ids)
                n_ann = len(tmp_anns)

                tmp_anns = [ann for ann in tmp_anns if ('bbox' in ann.keys() and len(ann['bbox']) > 0)]
                nn_ann = len(tmp_anns)

                if not n_ann == nn_ann:
                    print('kicked out ', (n_ann - nn_ann), 'for frame', img_idx, 'in seq', seq_idx)
                sequence_img_infos.append(img_info)

                if "ignore_regions_x" in img_info:
                    frame_regions = []
                    for region_idx in range(len( img_info['ignore_regions_x'])):
                        region = []
                        for point_idx in range(len(img_info['ignore_regions_x'][region_idx])):
                            region.append(
                                    [
                                        img_info['ignore_regions_x'][region_idx][point_idx],
                                        img_info['ignore_regions_y'][region_idx][point_idx]
                                    ]
                            )
                        frame_regions.append(region)

                    seq_ignore_regions.append(frame_regions)
                else:
                    seq_ignore_regions.append([])

                sequence_anns.append(tmp_anns)
                self.index[tot_frames] = [seq_idx, fr_count]
                tot_frames += 1
                fr_count += 1

            self.sequence_anns.append(
                {
                    'img_infos': sequence_img_infos,
                    'img_anns': sequence_anns,
                    'ignore_regions': seq_ignore_regions,
                    'num_frames': len(sequence_img_infos)
                }
            )

        self.num_examples = tot_frames

    def __build_image_paths__(self, sequence_idx, frames=None):
        img_anns = self.sequence_anns[sequence_idx]['img_infos']

        if frames is None:
            filenames = [img_info['file_name'] for img_info in img_anns]
        else:
            filenames = [img_anns[img_idx]['file_name'] for img_idx in frames]

        paths = [os.path.join(self.data_dir, filename) for filename in filenames]

        return paths

    def load_images(self, sample_idx, frames=None):
        paths = self.__build_image_paths__(sample_idx, frames)
        return np.asarray([imread(path, mode='RGB') for path in paths])

    def num_objects(self, sequence_idx, frames=None):
        anns = self.sequence_anns[sequence_idx]['img_anns']

        if frames is None:
            num_objects = [len(ann) for ann in anns]
        else:
            num_objects = [len(anns[idx]) for idx in frames]

        return num_objects

    def get_mask(self, sequence_idx, height, width, frames=None):

        sequence_ignore_regions = self.sequence_anns[sequence_idx]['ignore_regions']
        ignore_regions = []

        if frames is None:
            frames = [idx for idx in range(len(sequence_ignore_regions))]

        for fr_idx in frames:
            m = np.ones((height, width))
            if len(sequence_ignore_regions[fr_idx]) > 0:
                for region in sequence_ignore_regions[fr_idx]:
                    if len(region) > 0:
                        m = cv.fillPoly(m, np.asarray([region]), 0)

            ignore_regions.append(m)

        return np.asarray(ignore_regions)

    def get_keypoints(self, sequence_idx, ann=None, frames=None):

        anns = self.sequence_anns[sequence_idx]['img_anns']
        frame_kpts = []

        if frames is None:
            frames = [idx for idx in range(len(anns))]

        num_persons = self.num_objects(sequence_idx, frames=frames)

        for idx, fr_idx in enumerate(frames):

            kpts = np.zeros((num_persons[idx], self.num_parts + 2, 3))

            for i in range(num_persons[idx]):
                kpts[i] = np.asarray(anns[fr_idx][i]['keypoints']).reshape([-1, 3])

            kpts = kpts[:, self.valid_idxs]

            frame_kpts.append(kpts)

        return frame_kpts

    def get_bounding_boxes(self, sequence_idx, frames=None):

        sequence_bbs = []

        sequence_anns = self.sequence_anns[sequence_idx]['img_anns']

        for fr_idx in frames:
            img_ann = sequence_anns[fr_idx]

            bbs = [ann['bbox'] for ann in img_ann]
            sequence_bbs.append(bbs)

        #
        return sequence_bbs

    def get_anns(self, sequence_idx, frames=None):
        return None

    def get_joint_mask(self, sequence_idx):

        joint_mask = np.ones(self.num_parts)

        return joint_mask
