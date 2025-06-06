"""
__config__ contains the options for training and testing
Basically all of the variables related to training are put in __config__['train'] 
"""
import torch
import numpy as np
from torch import nn
import os
from torch.nn import DataParallel
from utils.misc import make_input, make_output, importNet
from constrained_pose_estimation.train_constrained_rnn import trainer as rnn_train
from data.coco_pose.dp import GenerateSeparateHeatmap

__config__ = {
    'data_provider': 'data.coco_pose.dp',
    'network': 'models.posenet.PoseNet',
    'inference': {
        'nstack': 4,
        'inp_dim': 256,
        'oup_dim': 68,
        'num_parts': 17,
        'increase': 128,
        'keys': ['imgs']
    },

    'train': {
        'batchsize': 1,
        'input_res': 512,
        'output_res': 128,
        'train_iters': 57000,
        'valid_iters': 2,
        'learning_rate': 1e-3,  # 2e-4, decreasing to 1e-5 since this is finetuning.
        'num_loss': 4,

        'loss': [
            ['push_loss', 1e-3],
            ['pull_loss', 1e-3],
            ['detection_loss', 1],
        ],

        'max_num_people': 30,
        'num_workers': 2,
        'use_data_loader': True,
    },
}


class Trainer(nn.Module):
    """
    The wrapper module that will behave differetly for training or testing
    inference_keys specify the inputs for inference
    """

    def __init__(self, model, inference_keys, calc_loss):
        super(Trainer, self).__init__()
        self.model = model
        self.keys = inference_keys
        self.calc_loss = calc_loss

    def forward(self, imgs, **inputs):
        inps = {}
        labels = {}

        for i in inputs:
            if i in self.keys:
                inps[i] = inputs[i]
            else:
                labels[i] = inputs[i]
        # TODO rania:
        if False:  # not self.training:
            return self.model(imgs, **inps)
        else:
            res = self.model(imgs, **inps)
            if type(res) != list and type(res) != tuple:
                res = [res]
            return list(res) + list(self.calc_loss(*res, **labels))


def make_network(configs):
    PoseNet = importNet(configs['network'])
    train_cfg = configs['train']
    config = configs['inference']

    poseNet = PoseNet(**config)

    forward_net = DataParallel(poseNet.cuda())

    def calc_loss(*args, **kwargs):
        return poseNet.calc_loss(*args, **kwargs)

    config['net'] = Trainer(forward_net, configs['inference']['keys'], calc_loss)
    config['heatmap_generator'] = GenerateSeparateHeatmap(128, config['num_parts'])
    train_cfg['optimizer'] = torch.optim.Adam(config['net'].parameters(), 1e-5)  # train_cfg['learning_rate'])
    config['rnn_net'] = rnn_train(list(config['net'].parameters()))

    exp_path = os.path.join('exp', configs['opt'].exp)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    logger = open(os.path.join(exp_path, 'log'), 'a+')

    def make_train(batch_id, config, phase, **inputs):
        for i in inputs:
            if i != 'im_name' and i != 'separate_heatmaps' and i != 'areas':
                # print('******************', i)
                inputs[i] = make_input(inputs[i])

        net = config['inference']['net']
        rnn_trainer = config['inference']['rnn_net']
        config['batch_id'] = batch_id
        # print ('****************phase', phase)
        if phase == 'train':
            net = net.train()
        else:
            net = net.eval()

        if phase != 'inference':
            result = net(inputs['imgs'], **{i: inputs[i] for i in inputs if i != 'imgs'})

            ae_loss = None
            if isinstance(result, list):
                result, push_loss, pull_loss = result
                ae_loss = push_loss + pull_loss

            result.retain_grad()
            a = list(net.parameters())[0].clone()
            rnn_trainer.forward_sample(result[0], inputs['separate_heatmaps'], inputs['areas'], inputs['masks'],
                                       image_name=inputs['im_name'], is_validation=phase == 'valid', ae_loss=ae_loss)

 #           train_cfg['optimizer'].zero_grad()
#            train_cfg['optimizer'].step()
            b = list(net.parameters())[0].clone()
            print(torch.equal(a.data, b.data), 'are params equal***************************')
            if False:
                num_loss = len(config['train']['loss'])

                ## I use the last outputs as the loss
                ## the weights of the loss are controlled by config['train']['loss']
                losses = {i[0]: result[-num_loss + idx] * i[1] for idx, i in enumerate(config['train']['loss'])}

                loss = 0
                toprint = '\n{}: '.format(batch_id)
                for i in losses:
                    loss = loss + torch.mean(losses[i])

                    my_loss = make_output(losses[i])
                    my_loss = my_loss.mean(axis=0)

                    if my_loss.size == 1:
                        toprint += ' {}: {}'.format(i, format(my_loss.mean(), '.8f'))
                    else:
                        toprint += '\n{}'.format(i)
                        for j in my_loss:
                            toprint += ' {}'.format(format(j.mean(), '.8f'))

                logger.write(toprint)
                logger.flush()

                if batch_id == 200000:
                    ## decrease the learning rate after 200000 iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 1e-5

                if phase == 'train':
                    optimizer = train_cfg['optimizer']
                    optimizer.zero_grad()
                    loss.backward()
                    print('is grad*******************************', list(model.parameters())[0].grad)
                return None
            # else:
            #     out = {}
            #     net = net.eval()
            #     result = net(**inputs)
            #     if type(result)!=list and type(result)!=tuple:
            #         result = [result]
            #     out['preds'] = [make_output(i) for i in result]
            #     return out

    return make_train
