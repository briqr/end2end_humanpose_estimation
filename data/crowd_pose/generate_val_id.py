from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':


    annotation_file = '/media/datasets/briq/datasets/crowdpose/annotations/crowdpose_val.json'



    coco_gt = COCO(annotation_file)

    catIds = coco_gt.getCatIds(catNms=['person'])
    imgIds = coco_gt.getImgIds(catIds=catIds)
    annType = 'keypoints'
    # assert dataset.num_frames == len(imgIds)



    cocoEval = COCOeval(coco_gt, None, annType)
    cocoEval.params.imgIds = imgIds
    with open('data/coco_pose/valid_id', 'w') as f:
        for imId in imgIds:
            f.write("%s\n" % imId)
    f.close()