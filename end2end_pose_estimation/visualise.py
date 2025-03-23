import cv2 as cv
from pycocotools.coco import COCO
from pycocotools import mask
import os
import numpy as np


colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
def visualise_joints(img_name, poses, flip=False):
# visualize: code taken from https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/testing/python/demo.ipynb

    oriImg = cv.imread(img_name)
    canvas = cv.imread(img_name) # B,G,R order

    for k in range(len(poses)):
        canvas = cv.imread(img_name)  # B,G,R order
        for j in range(len(poses[0])):
            if np.isnan(poses[k][j]).any():# or np.all(poses[k][j][x]==0 for x in range(2)): #or poses[k][j][0] > oriImg.shape[1] or poses[k][j][0]>oriImg.shape[1]:
                continue
            x, y = poses[k][j][0:2]
            if flip:
                cv.circle(canvas, (int(y),int(x)), 4, colors[j], thickness=-1)
            else:
                cv.circle(canvas, (int(x), int(y)), 4, colors[j], thickness=-1)

        to_plot = cv.addWeighted(oriImg, 0.3, canvas, 0.7, 0)
        cv.imshow('joints', to_plot[:,:,[2,1,0]])
        #fig = matplotlib.pyplot.gcf()
        #fig.set_size_inches(size_1, size_2)
        cv.waitKey(0)
        a = 0
    cv.destroyAllWindows()
    #plt.waitforbuttonpress(0)
        #plt.close()


data_dir = '/media/datasets_local/crowdpose/images'
ann_path = '/media/datasets_local/crowdpose/annotations/crowdpose_train.json'

coco = COCO(ann_path)
img_ids = coco.getImgIds()
for idx in range(10):
    ann_ids = coco.getAnnIds(imgIds=img_ids[idx])
    tmp_ann = coco.loadAnns(ann_ids)
    for ann in tmp_ann:
        print(ann)
    img_info = coco.loadImgs(img_ids[idx])[0]
    path = img_info['file_name']
    im_name = os.path.join(data_dir, path)

    current_ann = np.asarray(tmp_ann[0]['keypoints']).reshape(14, 3)
    visualise_joints(im_name, np.expand_dims(current_ann[13:14], 0))