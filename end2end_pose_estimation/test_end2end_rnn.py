#This file is invoked from the test_fused rather than being a standalone inference class, i.e., the features are generated on the fly

import torch
import torch.optim
import numpy as np
from scipy.sparse.sputils import upcast
from torch import nn
from torch.autograd import Variable
import constrained_pose_estimation.optimization.calculation_util as calc_util
import constrained_pose_estimation.evaluation.evaluation_utils as eu
from constrained_pose_estimation.general_helper import helper_functions as hf
from constrained_pose_estimation.modules.RNN_model import *

from constrained_pose_estimation.general_helper import helper_functions as hf
import constrained_pose_estimation.visualisation.visual_util as vu


class rnn_tester:

    def __init__(self):
        params_file = 'constrained_pose_estimation/config/test_rnn_params.yaml'
        self.params = hf.parse_params(params_file)
        self.model = RPP(self.params)
        checkpoint_path = self.params.checkpoint# % (self.params.max_k)
        if 'stop' in checkpoint_path:
            self.stopping = True
        else:
            self.stopping = True
        #self.stopping = False
        print('****************stopping:' + str(self.stopping))
        hf.load_model(checkpoint_path, self.model)
        self.all_poses = []
        self.all_scores = []
        self.all_im_ids = []
        self.max_k = self.params.max_k
        self.model.cuda()


        # is_training indicates whether we are evaluating on the training set or the validation set, needs to be consistent with is_training in train_fused.py
        val = 'val'  # or train

        if val == 'val':
            self.img_path = self.params.val_im_path
        else:
            self.img_path = self.params.train_im_path

    def run_inference(self, total_features, transform_mat, img_name=None):

        print(img_name)
        num_outputs = self.params.num_hourglass_modules
        #total_features = total_features[:, 0:34, :, :]  # exclude the visual features vf
        #total_features = total_features[:,17:68,:,:] #exclude the Hms
        total_features_in = torch.cat(([total_features[i] for i in range(num_outputs)]), dim=0)
        #total_features_in = torch.cat([total_features[0], total_features[3]], dim=0)

        input = Variable(total_features_in.unsqueeze(0))

        out_heatmaps = []  # the joint heatmaps per person
        out_stops = []  # the stop probabilities
        stop_next = False
        #nu_persons = current_heat_gt.shape[0]
        T = self.params.max_k
        hidden = None
        print('****************************----------------------------------*************************************')
        for t in range(T):
            #out_heatmap, confidence, hidden = self.model(input, hidden)  # this is if there's a stop probability trained
            #print(confidence)
            prev_hm = torch.empty(1, 17, 128, 128).uniform_(0, 0.1).cuda()
            for iter in range(3):
                input_ = torch.cat((input, prev_hm), dim=1)
                # print('input_.shape*****', input_.shape)
                out_heatmap, confidence, hidden = self.model(input_, hidden)
                # print('passed first iteration ******')
                # print (out_heatmap.shape, 'heatmap*** shape')
                prev_hm = out_heatmap.clone()
            if self.stopping:
                if confidence < 0.4 and len(out_heatmaps) > 0:
                    count_after_stop += 1
                    if count_after_stop == 3:
                        break
            else:
                max_val = torch.max(out_heatmap[0])
                print('max_val %.3f' % max_val)
                if max_val < 0.01 and len(out_heatmaps) > 0:  # this is if there is no stop probability, get rid of the second condition
                    break
            out_heatmaps.append(out_heatmap[0])

                #out_stops.append(out_stop)

        predicted_k = len(out_heatmaps)  # the number of instances
        #print('gt number of persons, predicted:', true_k, ', ', predicted_k)
        print('predicted number of persons: ', predicted_k)


        #im_full_name =  img_name#'/media/datasets/pose_estimation/MSCOCO_2017/images/train2017/000000554625.jpg' #self.img_path + img_name + '.jpg'
        dt_poses, scores = calc_util.get_dt_poses(img_name, out_heatmaps, transform_mat)
        self.all_poses.append(dt_poses)
        self.all_scores.append(scores)
        self.all_im_ids.append(img_name.split('/')[-1].split('.')[0])
        if False:
            #vu.visualise_heatmaps(None, out_heatmaps)
            vu.visualise_heatmaps_per_person(img_name, out_heatmaps, None, title='result', is_rnn=True, hm_combined_features=total_features[0, 0:17])
            vu.visualise_joints(img_name, dt_poses, False)
            #vu.visualise_joints_from_heatmaps(img_name, out_heatmaps, transform_mat, flip=True)
            #vu.visualise_connections_res(img_name, dt_poses, flip=False)
            # for k in range(3):
            #     print('----now difference')
            #     print (total_features[k,0:17]- total_features[k+1,0:17])
            #     print('*** now heatmaps')
            #     print (total_features[k,0:17])
            #     #vu.visualise_heatmaps_per_person(img_name, out_heatmaps, None, title='result', is_rnn=True,
                #                                hm_combined_features=total_features[k, 0:17])


            #vu.visualise_connections_res(im_full_name, dt_poses, True)
            #vu.visualise_heatmaps_per_person_supression(img_name, out_heatmaps, title='result supp', is_rnn=True, hm_combined_features=total_features[0, 0:17])
            #vu.visualise_heatmaps_per_person(img_name, out_heatmaps, None, title='result', is_rnn=True,
            #                                hm_combined_features=current_heatmap[0])
            #vu.visualise_joints_from_heatmaps_with_suppression(im_full_name, out_heatmaps, flip=True)
            return out_heatmaps
    def save_poses(self):

        hf.save_poses(self.params.result_path, self.all_poses, self.all_scores, self.all_im_ids)


