import cv2
import torch
import tqdm
import os
import numpy as np
import pickle
import random
import numpy as np
from data.coco_pose.ref import ref_dir
from utils.misc import get_transform, kpt_affine, resize
from utils.group import HeatmapParser
from visualise import *

#valid_filepath = ref_dir + '/OCHumantest.pkl' #OCHumanvalidation
valid_filepath = ref_dir + '/validation.pkl'
from crowdposetools.coco import COCO
from crowdposetools.cocoeval import COCOeval

parser = HeatmapParser(detection_val=0.015)


def refine(det, tag, keypoints):
    """
    Given initial keypoint predictions, we identify missing joints
    """
    if len(tag.shape) == 3:
        tag = tag[:, :, :, None]

    tags = []
    for i in range(keypoints.shape[0]):
        if keypoints[i, 2] > 0:
            y, x = keypoints[i][:2].astype(np.int32)
            tags.append(tag[i, x, y])

    prev_tag = np.mean(tags, axis=0)
    ans = []

    for i in range(keypoints.shape[0]):
        tmp = det[i, :, :]
        tt = (((tag[i, :, :] - prev_tag[None, None, :]) ** 2).sum(axis=2) ** 0.5)
        tmp2 = tmp - np.round(tt)

        x, y = np.unravel_index(np.argmax(tmp2), tmp.shape)
        xx = x
        yy = y
        val = tmp[x, y]
        x += 0.5
        y += 0.5

        if tmp[xx, min(yy + 1, det.shape[1] - 1)] > tmp[xx, max(yy - 1, 0)]:
            y += 0.25
        else:
            y -= 0.25

        if tmp[min(xx + 1, det.shape[0] - 1), yy] > tmp[max(0, xx - 1), yy]:
            x += 0.25
        else:
            x -= 0.25

        x, y = np.array([y, x])
        ans.append((x, y, val))
    ans = np.array(ans)

    if ans is not None:
        for i in range(14):
            #print('key point score',  keypoints[i,2])
            if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                #print('ans larger than 0')
                keypoints[i, :2] = ans[i, :2]
                keypoints[i, 2] = 1
            #else:
            #    print ('ans not larger than 0!!!!!!')

    return keypoints



def array2dict(tmp):
    return {
        'det': tmp[0][:, :, :14],  # take the heatmaps of all hourglass modules
        'tag': tmp[0][:, -1, 14:14*2]
        # tmp[0].shape=b*4*68*128*128, take the tags of the last hourglass module only
    }

def multiperson(img, func, mode, image_name=None):
    """
      1. Resize the image to different scales and pass each scale through the network
      2. Merge the outputs across scales and find people by HeatmapParser
      3. Find the missing joints of the people with a second pass of the heatmaps
      """
    mode = 'm'
    mirrored = False
    if mode == 'm':
        scales = [2, 1., 0.5]
        multi = True
    else:
        scales = [2, 1.]  # in the single mode, scale 2 is only used for reference when inverting back to the original scale
        multi = False

    required_dim = (256, 256)

    height, width = img.shape[0:2]
    center = (width / 2, height / 2)
    dets, tags = None, []

    for idx, i in enumerate(scales):
        scale = max(height, width) / 200

        inp_res = int((i * 512 + 63) // 64 * 64)
        res = (inp_res, inp_res)
        mat_ = get_transform(center, scale, res)[:2]
        if idx == 0:
            ref_mat_ = mat_.copy()
            if not multi:  # if this is not multi scale, skip scale 2, from which we only take the transform
                continue
        inp = cv2.warpAffine(img, mat_, res) / 255
        result = func([inp], image_name)

        num_outputs = 4  # number of hourglass modules
        result = result['preds'][0][:, 0:num_outputs]  # the result of the AE network
        result = result[np.newaxis, :]

        tmp1 = array2dict(result)

        if mirrored:  # use mirrored version only in multiscale
            result2 = func([inp[:, ::-1]], image_name)  # for the mirrored version
            tmp2 = array2dict(result2)  # for the mirrored version

        tmp = {}
        for ii in tmp1:
            if mirrored:
                tmp[ii] = np.concatenate((tmp1[ii], tmp2[ii]), axis=0)  # for the mirrored version
            else:
                tmp[ii] = tmp1[ii]  # for the non mirrored version

        det = tmp['det'][0, -1]
        if mirrored:
            det += tmp['det'][1, -1, :, :, ::-1][flipRef]

        if det.max() > 10:
            continue
        if dets is None:
            dets = resize(det, required_dim)
        else:
            dets = dets + resize(det, required_dim)

        if abs(i - 1) < 0.5:  # use tags of scale 1 only
            if mirrored:
                tags += [resize(tmp['tag'][0], required_dim),
                         resize(tmp['tag'][1, :, :, ::-1][flipRef], required_dim)]  # for the non mirrored version
            else:
                tags = [resize(tmp['tag'][0], required_dim)]

    if dets is None or len(tags) == 0:
        return [], []

    tags = np.concatenate([i[:, :, :, None] for i in tags], axis=3)
    if multi:
        dets = dets / len(scales) / 2  # /2 is for the added mirrored version
    elif mirrored:
        dets /= 2
    dets = np.minimum(dets, 1)
    grouped = parser.parse(np.float32([dets]), np.float32([tags]))[0]

    scores = [i[:, 2].mean() for i in grouped]

    should_refine = True  # todo eliminate refine
    if should_refine:
        for i in range(len(grouped)):
            grouped[i] = refine(dets, tags, grouped[i])

    if len(grouped) > 0:
        # print(grouped[:,:,:2])
        mat = np.linalg.pinv(np.array(ref_mat_).tolist() + [[0, 0, 1]])[:2]  # the inverse of the affine transform
        grouped[:, :, :2] = kpt_affine(grouped[:, :, :2] * 4, mat)
    # print(grouped[:,:,:2])
    return grouped, scores


def process_detection_heatmaps(result, num_outputs):
    result = result[0][:, 0:num_outputs]
    result = result[np.newaxis, :]
    det = result[0][:, :, :14]
    if det is None:
        return []
    # dets = dets / len(scales)  # /2 #/2 is for the added mirrored version
    det[det > 1] = 1
    return det



def coco_eval(prefix, dt, gt, is_single=False):
    """
    Evaluate the result with COCO API
    """

    for _, i in enumerate(sum(dt, [])):
         i['id'] = _ + 1

    image_ids = []
    import copy
    gt = copy.deepcopy(gt)

    dic = pickle.load(open(valid_filepath, 'rb'))
    paths, anns, idxes, info = [dic[i] for i in ['path', 'anns', 'idxes', 'info']]

    widths = {}
    heights = {}
    for idx, (a, b) in enumerate(zip(gt, dt)):
        if len(a) > 0:
            for i in b:
                i['image_id'] = a[0]['image_id']
            image_ids.append(a[0]['image_id'])
        if info[idx] is not None:
            widths[a[0]['image_id']] = info[idx]['width']
            heights[a[0]['image_id']] = info[idx]['height']
        else:
            widths[a[0]['image_id']] = 0
            heights[a[0]['image_id']] = 0
    image_ids = set(image_ids)

    import json
    cat = [{'supercategory': 'person', 'id': 1, 'name': 'person',
            'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                         [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]],
            'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                          'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']}]
    with open(prefix + '/gt.json', 'w') as f:
        json.dump({'annotations': sum(gt, []),
                   'images': [{'id': i, 'width': widths[i], 'height': heights[i]} for i in image_ids],
                   'categories': cat}, f)

    with open(prefix + '/dt.json', 'w') as f:
        json.dump(sum(dt, []), f)

    coco = COCO(prefix + '/gt.json')
    coco_dets = coco.loadRes(prefix + '/dt.json')
    coco_eval = COCOeval(coco, coco_dets, "keypoints")
    image_ids = list(image_ids)
    coco_eval.params.imgIds = image_ids
    coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if is_single:
        for i in range(len(image_ids)):
            print('img id', image_ids[i])
            coco_eval.params.imgIds = image_ids[i:i+1]
            coco_eval.params.catIds = [1]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()


    return coco_eval.stats



def genDtByPred(pred, image_id=0):
    """
    Generate the json-style data for the output
    """
    ans = []
    for i in pred:
        val = pred[i] if type(pred) == dict else i
        if val[:, 2].max() > 0:
            tmp = {'image_id': int(image_id), "category_id": 1, "keypoints": [], "score": float(val[:, 2].mean())}
            p = val[val[:, 2] > 0][:, :2].mean(axis=0)
            for j in val:
                if j[2] > 0.:
                    tmp["keypoints"] += [float(j[0]), float(j[1]), 1]
                else:
                    tmp["keypoints"] += [float(p[0]), float(p[1]), 1]
            ans.append(tmp)
    return ans


def get_img(inp_res=512):
    """
    Load validation images
    """
    if os.path.exists(valid_filepath) is False:
        from utils.build_valid import main
        main()

    dic = pickle.load(open(valid_filepath, 'rb'))

    paths, anns, idxes, info = [dic[i] for i in ['path', 'anns', 'idxes', 'info']]

    total = len(paths)
    tr = tqdm.tqdm(range(0, total), total=total)
    for i in tr:
        img = cv2.imread(paths[i])[:, :, ::-1]
        yield anns[i], img, paths[i]


def main():
    my_seed = 11
    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from train import init
    func, config = init()
    mode = config['opt'].mode

    def runner(imgs, image_name):
        # return func(0, config, 'inference', imgs=torch.Tensor(np.float32(imgs)), image_name=image_name)['preds']
        return func(0, config, 'inference', imgs=torch.Tensor(np.float32(imgs)), image_name=image_name)

    def do(img, img_name):
        if True:
            ans, scores = multiperson(img, runner, mode, img_name)
            if len(ans) > 0:
                ans = ans[:, :, :3]

            pred = genDtByPred(ans)

            for i, score in zip(pred, scores):
                i['score'] = float(score)
            return pred
        else:
            multiperson(img, runner, mode, img_name)  # for my inference

    gts = []
    preds = []

    idx = 0

    for anns, img, img_name in get_img(inp_res=-1):
        # print('gt keypoints')
        # for ann in anns:
        #    print(np.asarray(ann['keypoints']).reshape(17,3)[:,0:2])
        idx += 1
        gts.append(anns)
        dt = do(img, img_name)
        preds.append(dt)
        if True:
            to_disp = np.zeros((len(dt), 14, 3))
            k = 0
            for p in dt:
                to_disp[k] = (np.asarray(p['keypoints'])).reshape(14, 3)
                # print('to_disp')
                # print (to_disp[k,:,0:2])
                k += 1
            visualise_joints(img_name, to_disp, False)


        if idx == 1:
            break

    prefix = os.path.join('exp', config['opt'].exp)
    coco_eval(prefix, preds, gts)


if __name__ == '__main__':
    with torch.no_grad():
        main()
