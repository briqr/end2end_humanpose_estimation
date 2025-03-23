# This trainer class is used when training in a fused manner, i.e., the input features from the training samples are generated online by the underlying network.exp
import torch
import torch.optim
from torch.autograd import Variable
import numpy as np
import constrained_pose_estimation.optimization.calculation_util as calc_util
import constrained_pose_estimation.evaluation.evaluation_utils as eu
from constrained_pose_estimation.general_helper import helper_functions as hf
from constrained_pose_estimation.modules.RNN_model import *
from constrained_pose_estimation.Global_Constants import symmetric_joints as SymmetricJoints
import constrained_pose_estimation.visualisation.visual_util as vu
from torch.optim.lr_scheduler import ReduceLROnPlateau


class trainer:

    def __init__(self, AE_params):
        params_file = 'constrained_pose_estimation/config/train_rnn_params.yaml'
        params = hf.parse_params(params_file)

        self.criterion = [nn.MSELoss(size_average=False), nn.BCEWithLogitsLoss(size_average=False)]
        self.params = params
        self.model = RPP(params)
        if params.checkpoint:
            checkpoint_path = self.params.checkpoint  # % (self.params.max_k)
            hf.load_model(checkpoint_path, self.model)
            print('loaded checkpoint path', checkpoint_path)

        self.max_k = self.params.max_k
        self.model.cuda()
        self.model.train()
        self.stopping = True

        weight_decay = float(self.params.weight_decay)
        momentum = float(self.params.weight_decay)
        lr = float(self.params.lr)
        parameter_list = list(self.model.parameters()) + list(AE_params)
        self.max_k = self.params.max_k

        # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, parameter_list), lr=lr,
        #                              weight_decay=weight_decay)

        self.optimizer = torch.optim.Adam(
            [
                {'params': list(self.model.parameters()), "lr": lr},
                {'params': list(AE_params), "lr": 1e-4}
                # for finetuning on posetrack we need a higher learning rate in the beginning
            ],
            lr=lr,
            weight_decay=weight_decay)

        # self.ae_optimizer = AE_optimizer
        # self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, parameter_list),
        #                            lr=lr, weight_decay=weight_decay,
        #                            momentum=momentum)

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=0, verbose=True,
                                              threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        self.J = self.params.num_joints
        self.epoch = 0
        self.sample_i = 0
        self.validation_loss = 0

    def calc_loss_intensity(self, heatmap_out, heatmap_gt, confidence_out, predicted_k, true_k, heatmap_mask):
        total_loss = 0
        if predicted_k > 0.5:
            heatmap_out = torch.cat(heatmap_out, 0).view(len(heatmap_out), heatmap_out[0].shape[0],
                                                         heatmap_out[0].shape[1],
                                                         heatmap_out[0].shape[2])
            if True:  # self.epoch > 3:
                heatmap_loss = self.criterion[0](heatmap_out * heatmap_mask, heatmap_mask * heatmap_gt.float())
            else:
                heatmap_loss = self.criterion[0](heatmap_out, heatmap_gt.float())

            total_loss = heatmap_loss

        if self.stopping:  # whether to have a stopping prob in the output as well
            confidence_out = torch.cat(confidence_out, 0)  # make it an array rather than a list
            len_out = len(confidence_out)
            confidence_gt = torch.zeros(len_out)  #
            if true_k > len_out:
                confidence_gt[0: len_out] = 1
            else:
                confidence_gt[0: true_k] = 1
            stopping_loss = (self.criterion[1](confidence_out.cuda(), confidence_gt.cuda()))
            total_loss += (stopping_loss)
        return total_loss

    def save(self, is_epoch=True):
        if is_epoch:
            self.epoch += 1
            self.sample_i = 0
            save_name = '%s_epoch%s.pth' % (self.params.save_path, str(self.epoch))
            print('taking snapshot... ' + save_name)
            torch.save(self.model.state_dict(), save_name)
            torch.cuda.empty_cache()
            if self.epoch == 2:
                for group_idx, param_group in enumerate(self.optimizer.param_groups):
                    if group_idx == 0:
                        param_group['lr'] = 2e-5
                    elif group_idx == 1:
                        param_group['lr'] = 5e-5
                print('updated lr to 2e-5 after third epoch')
            if self.epoch == 3:
                for group_idx, param_group in enumerate(self.optimizer.param_groups):
                    if group_idx == 0:
                        param_group['lr'] = 1e-5
                    elif group_idx == 1:
                        param_group['lr'] = 1e-5

                print('updated lr to 1e-5 after fouth epoch')


        else:
            save_name = 'sample_%s%s_epoch%s.pth' % (self.params.save_path, str(self.sample_i), str(self.epoch))
            print('taking intermediate snapshot... ' + save_name)
            torch.save(self.model.state_dict(), save_name)

    def set_validation_mode(self, is_validation):
        pass

    # if is_validation:
    #    self.model.eval()
    # else:
    #    self.model.train()

    def reset_validation_epoch(self):
        print('*********validation loss', self.validation_loss)
        print('*-*-*-*-*-*-*-*-*-*-*-*- reset validation epoch   *-*-*-*-*-*-*-**-*-*-*- \n')
        for param_group in self.optimizer.param_groups:
            print('before lr scheduler step ********----------- learning rate', param_group['lr'])
        #        self.lr_scheduler.step(self.validation_loss)
        for param_group in self.optimizer.param_groups:
            print('after lr scheduler step *******------------ learning rate', param_group['lr'])
        self.validation_loss = 0

    def forward_sample(self, total_features, current_heat_gt, areas, current_mask=None, image_name=None,
                       is_validation=False, ae_loss=None):
        #        self.sample_i += 1
        print(image_name)
        im_full_name = image_name  # self.params.train_im_path + image_name + '.jpg'
        num_outputs = self.params.num_hourglass_modules  # the number of glass modules  outputs
        if 'avghm' in self.params.save_path:
            heatmap_features = torch.mean(total_features[:, 0:17, :, :], dim=0)  # averaged hms
        elif 'maxhm' in self.params.save_path:
            heatmap_features = torch.max(total_features[:, 0:17, :, :], dim=0)[0]  # max hm
            # heatmap_features = torch.cat(heatmap_features, 0).view(len(heatmap_features), heatmap_features[0].shape[0], heatmap_features[0].shape[1])
        else:
            heatmap_features = torch.cat(([total_features[i, 0:17, :, :] for i in range(num_outputs)]),
                                         dim=0)  # all hm features
        vf_features = torch.cat(([total_features[i, 34:68, :, :] for i in range(num_outputs)]),
                                dim=0)  # all vf features
        # tag_features = torch.cat(([total_features[i, 17:34,:,:] for i in range(num_outputs)]), dim=0) # all tag features
        tag_features = total_features[-1, 17:34, :, :]
        total_features_in = torch.cat([heatmap_features, tag_features, vf_features],
                                      dim=0)  # take the tags of the last hourglass module only.
        input = total_features_in.unsqueeze(0)

        current_heat_gt = current_heat_gt[0]  # self.upsampler(current_heat_gt[0])
        input.requires_grad_()
        current_heat_gt = current_heat_gt.cuda()
        current_heat_gt_temp = []
        for p in range(len(current_heat_gt)):
            if current_heat_gt[p].sum() > 0.001:  # hmm, well, i think I need to save something for the negative samples
                current_heat_gt_temp.append(current_heat_gt[p])
        if len(current_heat_gt_temp) < 1:
            current_heat_gt_temp = current_heat_gt  # this is a negative example, let's use it
            true_k = 0
        else:
            true_k = len(current_heat_gt_temp)
        if len(current_heat_gt_temp) < 1:
            return
        new_shape = current_heat_gt[0].shape
        try:
            current_heat_gt = torch.cat(current_heat_gt_temp, 0).view(len(current_heat_gt_temp), new_shape[0],
                                                                      new_shape[1], new_shape[2])
        except:
            print('torch cat error')
            return

        out_heatmaps = []  # the joint heatmaps per person
        out_confidence = []  # the stop probabilities
        stop_next = False
        true_k = current_heat_gt.shape[0]
        hidden = None
        # mask = 0 #torch.zeros(1, current_heat_gt.shape[1], current_heat_gt.shape[2], current_heat_gt.shape[3]).cuda()
        T = self.params.max_k
        dim = self.params.out_res
        additional_runs = 0  # run at least until n+2 so the network learns when to stop
        predicted_k = 0
        conf_thresh = 0
        for t in range(T):
            if stop_next:  # and len(out_heatmaps) >= true_k:
                additional_runs += 1
                if additional_runs == 2:
                    # out_heatmaps.append(out_heatmaps[0])
                    # out_confidence.append(confidence)
                    break
            prev_hm = torch.zeros(1, self.params.num_joints, dim, dim).uniform_(0, 0.1).cuda()
            for iter in range(3):
                input_ = torch.cat((input, prev_hm), dim=1)
                out_heatmap, confidence, hidden = self.model(input_, hidden)
                prev_hm = out_heatmap.clone()
            if not stop_next:  # once stop_next is True, the network shouldn't try to find more valid persons
                stop_next = confidence <= conf_thresh  # let's call the stop prob confidence  # out_stop>0.5 # or I could probably rely on the values of heatmaps output
            if not stop_next:
                if confidence > conf_thresh:
                    predicted_k += 1
            out_heatmaps.append(out_heatmap[0])
            out_confidence.append(confidence)
        len_out = len(out_heatmaps)

        # if predicted_k < 0.5:
        #    predicted_k = len_out
        min_k = true_k if true_k < predicted_k else predicted_k

        print('gt number of persons, predicted:', true_k, ', ', predicted_k)

        # pair wise distance between each pair of a gt and output
        if true_k > 1 and predicted_k > 0.5:
            pairwise_dist = calc_util.calc_pairwise_dist_nonbatch(out_heatmaps[0:min_k], current_heat_gt, False)
            assignment = calc_util.calc_assignment_non_batch(pairwise_dist[0].detach().cpu().numpy())
            if assignment is None:
                print('assignment is none')
                return

            sorted_, indices = torch.sort(assignment[:, 1])
            current_heat_gt = current_heat_gt[assignment[indices, 0]]

        if true_k > 0.5:
            if min_k > 0.5:
                current_heat_gt = current_heat_gt[0:min_k]
                out_heatmaps = out_heatmaps[0:min_k]
            heatmap_loss = self.calc_loss_intensity(out_heatmaps, current_heat_gt, out_confidence, predicted_k, true_k,
                                                    current_mask)  # the output length is at least true_k+2

            if ae_loss is not None:
                # only take last loss 
                heatmap_loss += ae_loss[-1] * 100

            if not is_validation:  # TODO, should be made a class field
                self.optimizer.zero_grad()
                heatmap_loss.backward()
                self.optimizer.step()
                # self.ae_optimizer.step()
                self.sample_i += 1
                print('training epoch:%d, batch: %d, toal_loss: %.4f' % (self.epoch, self.sample_i, heatmap_loss))
            else:
                self.validation_loss += heatmap_loss.detach()
