import cv2
import sys
import os
import torch
import numpy as np
from utils.misc import get_transform, kpt_affine
import torch.utils.data
from multiprocessing import dummy


class GenerateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        return 0
        # hms = np.zeros(shape = (self.num_parts, self.output_res, self.output_res), dtype = np.float32)
        # sigma = self.sigma
        # k = 0
        # for p in keypoints:
        #     for idx, pt in enumerate(p):
        #         if pt[2]>0:
        #             x, y = int(pt[0]), int(pt[1])
        #             if x<0 or y<0 or x>=self.output_res or y>=self.output_res:
        #                 #print('not in', x, y)
        #                 continue
        #             ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
        #             br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)
        #
        #             c,d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
        #             a,b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]
        #
        #             cc,dd = max(0, ul[0]), min(br[0], self.output_res)
        #             aa,bb = max(0, ul[1]), min(br[1], self.output_res)
        #             #hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
        #             hms[idx, aa:bb, cc:dd] = self.g[a:b, c:d]
        #     k += 1
        # return hms


class GenerateSeparateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape=(len(keypoints), self.num_parts, self.output_res, self.output_res), dtype=np.float32)
        sigma = self.sigma
        k = 0
        for p in keypoints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                        # print('not in', x, y)
                        continue
                    ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1)
                    br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2)

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    # hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
                    hms[k, idx, aa:bb, cc:dd] = self.g[a:b, c:d]
            k += 1
        return hms


class KeypointsRef():
    def __init__(self, max_num_people, num_parts):
        self.max_num_people = max_num_people
        self.num_parts = num_parts

    def __call__(self, keypoints, output_res):
        visible_nodes = np.zeros((self.max_num_people, self.num_parts, 2))
        for i in range(len(keypoints)):
            tot = 0
            for idx, pt in enumerate(keypoints[i]):
                x, y = int(pt[0]), int(pt[1])
                if pt[2] > 0 and x >= 0 and y >= 0 and x < output_res and y < output_res:
                    visible_nodes[i][tot] = (idx * output_res * output_res + y * output_res + x, 1)
                    tot += 1
        return visible_nodes


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, ds, index, split=None):
        self.input_res = config['train']['input_res']
        self.output_res = config['train']['output_res']

        self.generateHeatmap = GenerateHeatmap(config['train']['output_res'], config['inference']['num_parts'])
        self.separateGenerateHeatmap = GenerateSeparateHeatmap(config['train']['output_res'],
                                                               config['inference']['num_parts'])
        self.keypointsRef = KeypointsRef(config['train']['max_num_people'], config['inference']['num_parts'])
        self.ds = ds
        self.index = index
        self.split = split

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.loadImage(self.index[idx % len(self.index)])

    def loadImage(self, idx):
        ds = self.ds

        if self.split is None:
            inp, im_name = ds.load_image(idx)
            mask = ds.get_mask(idx).astype(np.float32)
            ann = ds.get_anns(idx)
            keypoints = ds.get_keypoints(idx, ann)
        else:
            inp = ds.load_image(idx)
            im_name = ds.__build_image_path__(idx)
            height, width = inp.shape[:2]

            mask = ds.get_mask(idx, height, width).astype(np.float32)
            keypoints = ds.get_keypoints(idx)

        keypoints2 = [i for i in keypoints if np.sum(i[:, 2] > 0) > 1]

        height, width = inp.shape[0:2]
        center = np.array((width / 2, height / 2))
        scale = max(height, width) / 200

        inp_res = self.input_res
        res = (inp_res, inp_res)

        aug_rot = (np.random.random() * 2 - 1) * 30.
        aug_scale = np.random.random() * (1.25 - 0.75) + 0.75
        scale *= aug_scale

        if self.split is None:
            areas = [ann[i]['area'] / scale for i in range(len(ann))]
        else:
            areas = []

        dx = np.random.randint(-40 * scale, 40 * scale) / center[0]
        dy = np.random.randint(-40 * scale, 40 * scale) / center[1]
        center[0] += dx * center[0]
        center[1] += dy * center[1]

        mat_mask = get_transform(center, scale, (self.output_res, self.output_res), aug_rot)[:2]
        mask = cv2.warpAffine((mask * 255).astype(np.uint8), mat_mask, (self.output_res, self.output_res)) / 255
        mask = (mask > 0.5).astype(np.float32)

        mat = get_transform(center, scale, res, aug_rot)[:2]
        inp = cv2.warpAffine(inp, mat, res).astype(np.float32) / 255
        keypoints[:, :, 0:2] = kpt_affine(keypoints[:, :, 0:2], mat_mask)

        if np.random.randint(2) == 0:
            inp = inp[:, ::-1]
            mask = mask[:, ::-1]
            keypoints = keypoints[:, ds.flipRef]
            keypoints[:, :, 0] = self.output_res - keypoints[:, :, 0]

        heatmaps = self.generateHeatmap(keypoints)
        separate_heatmaps = self.separateGenerateHeatmap(keypoints)
        keypoints = self.keypointsRef(keypoints, self.output_res)
        # return self.preprocess(inp).astype(np.float32), mask.astype(np.float32), keypoints.astype(np.int32), heatmaps.astype(np.float32), separate_heatmaps.astype(np.float32), im_name
        return self.preprocess(inp).astype(np.float32), mask.astype(np.float32), keypoints.astype(
            np.int32), heatmaps, separate_heatmaps.astype(np.float32), areas, im_name

    def preprocess(self, data):
        # random hue and saturation
        data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV);
        delta = (np.random.random() * 2 - 1) * 0.2
        data[:, :, 0] = np.mod(data[:, :, 0] + (delta * 360 + 360.), 360.)

        delta_sature = np.random.random() + 0.5
        data[:, :, 1] *= delta_sature
        data[:, :, 1] = np.maximum(np.minimum(data[:, :, 1], 1), 0)
        data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)

        # adjust brightness
        delta = (np.random.random() * 2 - 1) * 0.3
        data += delta

        # adjust contrast
        mean = data.mean(axis=2, keepdims=True)
        data = (data - mean) * (np.random.random() + 0.5) + mean
        data = np.minimum(np.maximum(data, 0), 1)
        # cv2.imwrite('x.jpg', (data*255).astype(np.uint8))
        return data


def prepare_dataloader(config):
    data_reader_module = config['dataset']

    if config['dataset_info']['name'] == 'posetrack':
        ds_train = data_reader_module.ImgDatasetReader(config['dataset_info']['posetrack_data_path'],
                                                       config['dataset_info']['posetrack_train_path'],
                                                       annotated_only=True)

        ds_val = data_reader_module.ImgDatasetReader(config['dataset_info']['posetrack_data_path'],
                                                     config['dataset_info']['posetrack_val_path'],
                                                     annotated_only=True)


    else:
        raise NotImplementedError("Unknown dataset name")

    return ds_train, ds_val


def init(config):

    """ Ugly, but deadline is close """

    batchsize = config['train']['batchsize']
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_path)

    my_seed = 11
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)

    if 'dataset' in config:
        ds_train, ds_val = prepare_dataloader(config)
        train = [i for i in range(ds_train.num_examples)]
        valid = [i for i in range(ds_val.num_examples)]

        dataset = {key: Dataset(config, ds, data, key) for key, data, ds in zip(['train', 'valid'],
                                                                                [train, valid],
                                                                                [ds_train, ds_val])}

        loaders = {}
        for key in dataset:
            if key == 'valid':
                bs = 1
            elif key == 'train':
                bs = batchsize
            else:
                raise NotImplementedError("Unknown key")

            loaders[key] = torch.utils.data.DataLoader(dataset[key], batch_size=bs, shuffle=True,
                                                       num_workers=config['train']['num_workers'], pin_memory=False)

    else:
        import ref as ds
        ds.init()

        train, valid = ds.setup_val_split()
        dataset = {key: Dataset(config, ds, data) for key, data in zip(['train', 'valid'], [train, valid])}

        use_data_loader = config['train']['use_data_loader']

        loaders = {}
        for key in dataset:
            loaders[key] = torch.utils.data.DataLoader(dataset[key], batch_size=batchsize, shuffle=True,
                                                       num_workers=config['train']['num_workers'], pin_memory=False)

    def gen(phase):
        batchsize = config['train']['batchsize']
        batchnum = config['train']['{}_iters'.format(phase)]
        loader = loaders[phase].__iter__()
        for i in range(batchnum):
            imgs, masks, keypoints, heatmaps, separate_heatmaps, areas, im_name = next(loader)
            yield {
                'imgs': imgs,
                'masks': masks,
                'heatmaps': heatmaps,
                'keypoints': keypoints,
                'separate_heatmaps': separate_heatmaps,
                'areas': areas,
                'im_name': im_name
            }

    return lambda key: gen(key)


def init_old(config):
    batchsize = config['train']['batchsize']
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_path)
    import ref as ds
    ds.init()

    train, valid = ds.setup_val_split()
    dataset = {key: Dataset(config, ds, data) for key, data in zip(['train', 'valid'], [train, valid])}

    use_data_loader = config['train']['use_data_loader']
    my_seed = 11
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)
    loaders = {}
    for key in dataset:
        loaders[key] = torch.utils.data.DataLoader(dataset[key], batch_size=batchsize, shuffle=True,
                                                   num_workers=config['train']['num_workers'], pin_memory=False)

    def gen(phase):
        batchsize = config['train']['batchsize']
        batchnum = config['train']['{}_iters'.format(phase)]
        loader = loaders[phase].__iter__()
        for i in range(batchnum):
            imgs, masks, keypoints, heatmaps, separate_heatmaps, areas, im_name = next(loader)
            yield {
                'imgs': imgs,
                'masks': masks,
                'heatmaps': heatmaps,
                'keypoints': keypoints,
                'separate_heatmaps': separate_heatmaps,
                'areas': areas,
                'im_name': im_name
            }

    return lambda key: gen(key)
