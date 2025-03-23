# includes ordering experiments, such as learned, random, or confidence based
import torch
import torch.optim
from torch.autograd import Variable
import numpy as np
import constrained_pose_estimation.optimization.calculation_util as calc_util
import constrained_pose_estimation.evaluation.evaluation_utils as eu
from constrained_pose_estimation.general_helper import helper_functions as hf
from constrained_pose_estimation.modules.RNN_model_single_person import *
from constrained_pose_estimation.modules.RNN_model import *
from constrained_pose_estimation.Global_Constants import  symmetric_joints as SymmetricJoints
import constrained_pose_estimation.visualisation.visual_util as vu
from torch.optim.lr_scheduler import ReduceLROnPlateau
from constrained_pose_estimation.modules.STN import *

inepoch = 2000
class trainer:

    def __init__(self, AE_params):
        params_file = 'constrained_pose_estimation/config/train_rnn_params.yaml'
        params = hf.parse_params(params_file)
        self.criterion = [nn.MSELoss(size_average=False), nn.BCEWithLogitsLoss(size_average=False), nn.L1Loss(size_average=False)]

        self.params = params
        self.model = RPP(params)
        self.model_single = RPP_single(params)  # single person RNN model on top of STN
        if params.checkpoint:
            checkpoint_path = self.params.checkpoint #% (self.params.max_k)
            hf.load_model(checkpoint_path, self.model)
            print('loaded checkpoint path', checkpoint_path)


        self.max_k = self.params.max_k
        self.model.cuda()
        self.model_single.cuda()
        self.model.train()
        self.model_single.train()
        self.stopping = True

        weight_decay = float(self.params.weight_decay)
        momentum = float(self.params.weight_decay)
        lr = float(self.params.lr)
        self.max_k = self.params.max_k
        self.stn = STN().cuda()
        current_parameter_list = list(self.model.parameters()) + list(self.model_single.parameters()) 
        #current_parameter_list = list(self.model.parameters()) + list(AE_params)
        #current_parameter_list = list(self.model.parameters()) + list(self.stn.parameters())


#        self.optimizer = torch.optim.Adam([ {'params': filter(lambda p: p.requires_grad, current_parameter_list)}, {'params': filter(lambda p: p.requires_grad, AE_params), 'lr': 1e-6}], lr=lr, weight_decay=weight_decay)
        self.optimizer = torch.optim.Adam([ {'params': filter(lambda p: p.requires_grad, current_parameter_list)}], lr=lr, weight_decay=weight_decay)
#        self.single_model_optimizer = torch.optim.Adam([ {'params': filter(lambda p: p.requires_grad, current_parameter_list)}], lr=lr, weight_decay=weight_decay)

#        self.stn_optimizer = torch.optim.Adam(self.stn.parameters(), lr=1e-4/128, weight_decay=weight_decay)



        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=0, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        self.J = self.params.num_joints
        self.epoch = 0
        self.sample_i = 0
        self.validation_loss = 0
        self.cumulative_loss = 0



    def calc_loss_intensity(self, heatmap_out, heatmap_gt, confidence_out, predicted_k, true_k, heatmap_mask, bb_out, bb_gt):

        total_loss = 0
        if predicted_k > 0.5:
            #heatmap_out = torch.cat(heatmap_out, 0).view(len(heatmap_out), heatmap_out[0].shape[0], heatmap_out[0].shape[1],
            #                                         heatmap_out[0].shape[2])
            heatmap_loss = self.criterion[0](heatmap_out*heatmap_mask, heatmap_gt.float()*heatmap_mask) #/len(heatmap_out)*3
            total_loss = heatmap_loss

        if self.stopping:  # whether to have a stopping prob in the output as well
#            confidence_out = torch.cat(confidence_out, 0) # make it an array rather than a list
            len_out = len(confidence_out)
            confidence_gt = torch.zeros(len_out) #
            if true_k > len_out:
                confidence_gt[0: len_out] = 1
            else:
                confidence_gt[0: true_k] = 1
            min_len = len(confidence_gt) #min(4, len(confidence_gt))
            stopping_loss = (self.criterion[1](confidence_out[0:min_len].cuda(), confidence_gt[0:min_len].cuda())) #/4 #/ len(confidence_gt)*4
            total_loss += (stopping_loss)

            #bb_term = 0
            #dim = 1
            #bb_gt = bb_gt.view(len(bb_gt), 4).float().cuda()
            #bb_gt[bb_gt<0] = 0#

            #bb_out = torch.cat(bb_out, 0).view(len(bb_out), 4)
            #bb_term = self.criterion[2](bb_out, bb_gt/dim)
            #total_loss += bb_term


        return total_loss



    def save(self, is_epoch=True):
        save_name = '%s_rnn1_epoch%s.pth' % (self.params.save_path, str(self.epoch))
        rnn2_save_name = '%s_rnn2_epoch%s.pth' % (self.params.save_path, str(self.epoch))
        #stn_save_name = '%s_stn_epoch%s.pth' % (self.params.save_path, str(self.epoch))
        if not is_epoch:
            seq = self.sample_i//inepoch
            save_name += str(seq)
            rnn2_save_name += str(seq)
            #stn_save_name += str(seq)
        torch.save(self.model.state_dict(), save_name)
        #torch.save(self.stn.state_dict(), stn_save_name)
        torch.save(self.model_single.state_dict(), rnn2_save_name)
        print('taking snapshot... ' + save_name, rnn2_save_name)


        if is_epoch:
            self.epoch += 1
            self.sample_i = 0
            torch.cuda.empty_cache()
            if self.epoch==3: # or self.epoch==4:
                for group_idx, param_group in enumerate(self.optimizer.param_groups):
                    if group_idx ==0:
                        param_group['lr'] /= 10 #param_group['lr']/10
                    print('updated lr to after third epoch', param_group['lr'])
            #for param_group in self.stn_optimizer.param_groups:
            #            param_group['lr'] /= 10
#            if self.epoch==3:
#                for param_group in self.optimizer.param_groups:
#                    if group_idx == 0:
#                        param_group['lr'] = 1e-6
#                print('updated lr to 1e-6 after fouth epoch')


        print('cumulative loss at epoch', self.epoch, self.cumulative_loss)
        self.cumulative_loss = 0



    def set_validation_mode(self, is_validation):
        pass
       #if is_validation:
       #    self.model.eval()
       #else:
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


    def forward_sample(self, total_features, current_heat_gt, current_mask, bboxes=None, image_name=None,is_validation=False, ae_loss=None):

        print (image_name)


        num_outputs = self.params.num_hourglass_modules # the number of glass modules  outputs

        vf_features = torch.cat(([total_features[i, 28:28+34,:,:] for i in range(num_outputs)]), dim=0) # all vf features

        #total_features_in = vf_features

        total_features_in = vf_features
        #total_features_in = heatmap_features

        #

        numsamples = 10
        #sh = total_features_in.shape
#        is_stn = True
        input = total_features_in.unsqueeze(0)

        current_heat_gt = current_heat_gt[0] #self.upsampler(current_heat_gt[0])
        input.requires_grad_()
        current_heat_gt = current_heat_gt.cuda()
        current_heat_gt_temp = []
        for p in range(len(current_heat_gt)):
            if current_heat_gt[p].sum()>0.001: 
                current_heat_gt_temp.append(current_heat_gt[p])
        if len(current_heat_gt_temp) < 1:
            return
            current_heat_gt_temp = current_heat_gt 
            true_k = 0
        else:
            true_k = len(current_heat_gt_temp)
        if len(current_heat_gt_temp) < 1:
            return
        new_shape = current_heat_gt[0].shape
        try:
            current_heat_gt = torch.cat(current_heat_gt_temp, 0).view(len(current_heat_gt_temp), new_shape[0], new_shape[1], new_shape[2])
        except:
            print ('torch cat error', current_heat_gt)
            return
    
        out_heatmaps = []  # the joint heatmaps per person
        out_confidence = []  # the stop probabilities
        out_bboxes = []
        stop_next = False
        true_k = current_heat_gt.shape[0]
        hidden = None
        T = self.params.max_k
        dim = self.params.out_res
        additional_runs = 0 #run at least until n+2 so the network learns when to stop
        predicted_k = 0
        conf_thresh = -1 
        for t in range(T):
            if stop_next and len(out_heatmaps) > true_k:
                additional_runs += 1
                if additional_runs == 1:
                    break
            prev_hm =torch.zeros(1, self.params.num_joints, dim, dim).uniform_(0, 0.1).cuda()
            for iter in range(2): #self.params.model_num_iters):   
                input_ = torch.cat( (input, prev_hm), dim=1)
                out_heatmap, confidence, hidden = self.model(input_, hidden)
                prev_hm = out_heatmap.clone()
            if not stop_next: #once stop_next is True, the network shouldn't try to find more valid persons
                 stop_next = confidence <= conf_thresh #let's call the stop prob confidence  # out_stop>0.5 # or I could probably rely on the values of heatmaps output
            if not stop_next:
                if confidence > conf_thresh:
                    predicted_k += 1

            out_heatmaps.append(out_heatmap[0])
            out_confidence.append(confidence)
        len_out = len(out_heatmaps)
        
        #if predicted_k < 0.5:
        #    predicted_k = len_out
#        min_k = true_k if true_k < len_out else len_out-1 # force true_k
        min_k = true_k if true_k < predicted_k else predicted_k     #don't force true k
        print('gt number of persons, predicted:', true_k, ', ', predicted_k)

        # pair wise distance between each pair of a gt and output
        if predicted_k>0.5: 
            out_heatmaps2 = []
            hidden = None
            perm = np.random.permutation(min_k)
            #for s in range(min_k):
            out_confidence = torch.cat(out_confidence, 0)
            #sorted_, indices = torch.sort(out_confidence[0:min_k], descending=True)
            for s in range(0, min_k):
                #p = perm[s]  # for a random permutation
                p = s # for learned ordering
                #p = indices[s] # for ordering based on confidence
                vf_hm = torch.cat( (input, out_heatmaps[p].unsqueeze(0)) , dim=1)
                out_hm2, hidden = self.model_single(vf_hm, hidden)
                out_heatmaps2.append(out_hm2[0])
            if true_k > 1: 
                pairwise_dist = calc_util.calc_pairwise_dist_nonbatch(out_heatmaps[0:min_k], current_heat_gt, False)
                assignment = calc_util.calc_assignment_non_batch(pairwise_dist[0].detach().cpu().numpy())
                if assignment is None:
                    print('assignment is none')
                    return
                sorted_, indices = torch.sort(assignment[:, 1])
                current_heat_gt = current_heat_gt[assignment[indices, 0]]
        current_loss = 0
        if true_k > 0.5:
            if min_k  > 0.5:
                current_heat_gt = current_heat_gt[0:min_k]
                out_heatmaps = out_heatmaps[0:min_k]
                out_heatmaps2 = out_heatmaps2[0:min_k]
                out_heatmaps2 = torch.cat(out_heatmaps2, 0).view(len(out_heatmaps2), out_heatmaps2[0].shape[0], out_heatmaps2[0].shape[1], out_heatmaps2[0].shape[2])

                out_heatmaps = torch.cat(out_heatmaps, 0).view(len(out_heatmaps), out_heatmaps[0].shape[0], out_heatmaps[0].shape[1], out_heatmaps[0].shape[2])
                current_loss = self.calc_loss_intensity(out_heatmaps, current_heat_gt, out_confidence, predicted_k, true_k, current_mask, out_bboxes, bboxes) # the output length is at least true_k+2
            # loss of the second RNN:
#                current_heat_gt = current_heat_gt[perm] # enforce the permutation ordering in the second RNN when random is used
                heatmap_loss2 = self.criterion[0](out_heatmaps2*current_mask, current_heat_gt.float()*current_mask) #/len(heatmap_out)*3
                current_loss += heatmap_loss2

            if current_loss > 0 and not is_validation: #TODO, should be made a class field
                self.cumulative_loss += current_loss.item()
                self.sample_i += 1
                if False: #(self.sample_i-1)%8==0:
                    self.optimizer.zero_grad()
                if ae_loss is not None:
                    print('ae loss, rnn loss', ae_loss, current_loss)
                    print('training epoch:%d, batch: %d, ae, rnn, toal_loss: %.4f %.4f %.4f' % (self.epoch, self.sample_i, ae_loss, current_loss, current_loss))
                    current_loss += ae_loss
                else:
                    print('training epoch:%d, batch: %d, loss: %.4f' % (self.epoch, self.sample_i, current_loss))

                if False: # self.sample_i > numsamples+1000:
                    vu.visualise_heatmaps_gt_per_person(image_name[0], out_heatmaps, current_heat_gt)
                self.optimizer.zero_grad()
                current_loss.backward()
#                if False and self.sample_i % 128 == 0:
#                    print('******************--------------------------updating stn gradient')
#                    self.stn_optimizer.step()
#                    self.stn_optimizer.zero_grad()
                self.optimizer.step()
                if self.sample_i%inepoch==0:
                    self.save(False)
            #else:
                #self.validation_loss += current_loss.detach()


