import sys
import time
import os
import tqdm
from os.path import dirname

import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.enabled = True

import pickle
import torch
import importlib
import argparse
import random
import numpy as np


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='pose', help='task to be trained')
    parser.add_argument('-c', '--continue_exp', type=str, help='continue exp')
    parser.add_argument('-e', '--exp', type=str, default='pose', help='experiments name')
    parser.add_argument('-m', '--mode', type=str, default='single', help='scale mode')
    parser.add_argument('-pt', '--pose_track', action='store_true')
    args = parser.parse_args()
    return args


def reload(config):
    """
    load or initialize model's parameters by config from config['opt'].continue_exp
    config['train']['epoch'] records the epoch num
    config['inference']['net'] is the model
    """
    opt = config['opt']

    if opt.continue_exp:
        resume = '/media/datasets/briq/pose_estimation/models/associative_embedding' #os.path.join('exp', opt.continue_exp)
        resume_file = os.path.join(resume, 'AEbackprop_allhm_epoch5.pth')

        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume_file)

            config['inference']['net'].load_state_dict(checkpoint['state_dict'])
            config['train']['optimizer'].load_state_dict(checkpoint['optimizer'])
            config['train']['epoch'] = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            exit(0)

    if 'epoch' not in config['train']:
        config['train']['epoch'] = 0


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    from pytorch/examples
    """
    basename = dirname(filename)
    if not os.path.exists(basename):
        os.makedirs(basename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save(config):
    resume = os.path.join('exp', config['opt'].exp)
    resume_file = os.path.join(resume, 'checkpoint.pth.tar')
    config['inference']['rnn_net'].save()
    save_checkpoint({
        'state_dict': config['inference']['net'].state_dict(),
        'optimizer': config['train']['optimizer'].state_dict(),
        'epoch': config['train']['epoch'],
    }, False,
        filename='/media/datasets/briq/pose_estimation/models/associative_embedding/AEbackprop_epoch_alltags_nomask_mask_epoch'+str(config['train']['epoch']))
    print('=> save checkpoint')


def train(train_func, data_func, config, post_epoch=None):
    while True:
        print('epoch: ', config['train']['epoch'])
        if 'epoch_num' in config['train']:
            if config['train']['epoch'] > config['train']['epoch_num']:
                break

        for phase in ['train', 'valid']:
            num_step = config['train']['{}_iters'.format(phase)]
            generator = data_func(phase)
            print('start', phase, config['opt'].exp)

            show_range = range(num_step)
            show_range = tqdm.tqdm(show_range, total=num_step, ascii=True)
            batch_id = num_step * config['train']['epoch']

            for i in show_range:
                try:
                    datas = next(generator)
                except Exception as e:
                    print('stop iteration exception!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', e)
                    break
                outs = train_func(batch_id + i, config, phase, **datas)
            if phase == 'train':
                save(config)
        config['inference']['rnn_net'].reset_validation_epoch()
        config['train']['epoch'] += 1


def init():
    """
    import configurations specified by opt.task

    task.__config__ contains the variables that control the training and testing
    make_network builds a function which can do forward and backward propagation

    please check task/base.py
    """
    opt = parse_command_line()
    task = importlib.import_module('task.' + opt.task)
    exp_path = os.path.join('exp', opt.exp)

    config = task.__config__
    try:
        os.makedirs(exp_path)
    except FileExistsError:
        pass

    config['opt'] = opt
    config['data_provider'] = importlib.import_module(config['data_provider'])

    if opt.pose_track:

        config['dataset_info'] = {}
        config['dataset_info']['name'] = 'posetrack'
        config['dataset_info']['posetrack_data_path'] = '/media/datasets/pose_estimation/PoseTrack_Challenge_2018/posetrack_data/'
        config['dataset_info']['posetrack_train_path'] = '/media/datasets/pose_estimation/PoseTrack_Challenge_2018/posetrack_data/annotations/train/'
        config['dataset_info']['posetrack_val_path'] =  '/media/datasets/pose_estimation/PoseTrack_Challenge_2018/posetrack_data/annotations/val/'

        config['dataset'] = importlib.import_module('dataset_readers.posetrack18')

    func = task.make_network(config)
    reload(config)
    return func, config


def main():
    func, config = init()
    data_func = config['data_provider'].init(config)
    train(func, data_func, config)


import multiprocessing as mp

if __name__ == '__main__':
    my_seed = 11
    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()
