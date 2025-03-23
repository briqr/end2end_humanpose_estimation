import tqdm
import os
import numpy as np
from data.coco_pose import ref
import pickle
train_im_path = '/media/datasets/pose_estimation/MSCOCO_2017/images/train2017/'
# the path of the validation of images for the coco dataset
val_im_path = '/media/datasets/pose_estimation/MSCOCO_2017/images/val2017/'
def build(idxes, ref):
    path, gts, info = [], [], []
    for idx in tqdm.tqdm( idxes, total = len(idxes) ):
        ann_ids = ref.coco.getAnnIds(imgIds = idx)
        ann = ref.coco.loadAnns(ann_ids)
        gts.append(ann)

        img_info = ref.coco.loadImgs(idx)[0]


        _path = train_im_path + img_info['file_name'] #img_info['file_name'].split('_')[1] + '/' + img_info['file_name']
        path.append(os.path.join(ref.data_dir, _path))
        assert os.path.exists(path[-1])
        info.append(img_info)
    return {
        'path': path,
        'anns': gts,
        'idxes': idxes,
        'info': info
    }

def main():
    ref.init()
    with open(ref.ref_dir + '/valid_id', 'r') as f:
        valid_id = list(map(lambda x:int(x.strip()), f.readlines()))
    pickle.dump(build(valid_id, ref), open(ref.ref_dir + '/validation.pkl', 'wb'))

if __name__ == '__main__':
    main()
