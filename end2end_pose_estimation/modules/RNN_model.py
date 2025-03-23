import torch
from end2end_pose_estimation.modules.clstm import ConvLSTMCell
import torch.nn as nn
from end2end_pose_estimation.layers.projection_rnn import Projection

import sys
sys.path.append("..")


class RPP(nn.Module): # recurrent person pose

    def __init__(self, params):
        super(RPP,self).__init__()
        features_dim = params.features_dim
        J = params.num_joints
        num_output = params.num_hourglass_modules
        features_dim = params.features_dim
        if 'avghm' in params.save_path or 'maxhm' in params.save_path:
            self.hidden_size = num_output*(features_dim+J)+2*J
        else:
            self.hidden_size = num_output*(J+features_dim) + 2*J # J for HM, J for tags, 2J for VF, an additional 1 or J for the iterative feedback, depending on whether this is the regression value or HM
        self.kernel_size = 3
        output_dim = J # the number of heatmaps outputted in one iterations
        padding = 0 if self.kernel_size == 1 else 1

        self.dropout = 0
        self.dropout_stop = 0
        self.skip_mode = 'none'

        # convlstms have decreasing dimension as width and height increase
        skip_dims_out = [self.hidden_size, self.hidden_size//2,
                        self.hidden_size//4] #, self.hidden_size//8]
                        #self.hidden_size//16]

        # initialize layers for each deconv stage
        self.clstm_list = nn.ModuleList()
        # 5 is the number of deconv steps that we need to reach image size in the output
        for i in range(len(skip_dims_out)):
            if i == 0:
                clstm_in_dim = self.hidden_size
            else:
                clstm_in_dim = skip_dims_out[i-1]
                if self.skip_mode == 'concat':
                    clstm_in_dim*=2
            clstm_i = ConvLSTMCell(True, clstm_in_dim, skip_dims_out[i],self.kernel_size, padding = padding)
            self.clstm_list.append(clstm_i)
        self.conv_out = nn.Conv2d(skip_dims_out[-1], output_dim, self.kernel_size, padding = padding)


        fc_dim = 0
        for sk in skip_dims_out:
            fc_dim+=sk
        self.fc_stop = nn.Linear(fc_dim,1)
        self.bn = nn.BatchNorm2d(self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.project = Projection()
        self.bias_term = nn.Parameter(torch.zeros(1, J, 128, 128))

    #@profile
    def forward(self, features, prev_hidden_list):
        clstm_in = self.bn(features)
        side_feats = []
        hidden_list = []

        for i in range(len(self.clstm_list)): #+1) +1 is for upsampling, which I might need after things works

            # hidden states will be initialized the first time forward is called
            if prev_hidden_list is None:
                state = self.clstm_list[i](clstm_in, None)
            else:
                # else we take the ones from the previous step for the forward pass
                state = self.clstm_list[i](clstm_in, prev_hidden_list[i])
            hidden_list.append(state)
            hidden = state[0]

            if self.dropout > 0:
                hidden = nn.Dropout2d(self.dropout)(hidden)

            side_feats.append(nn.MaxPool2d(clstm_in.size()[2:])(hidden))

            clstm_in = hidden

        out_heatmap = self.conv_out(clstm_in)


        # stopping criterion branch
        side_feats = torch.cat(side_feats, 1).squeeze()

        if self.dropout_stop > 0:
            stop_feats = nn.Dropout(self.dropout_stop)(side_feats)
        else:
            stop_feats = side_feats

        stop_probs = self.fc_stop(stop_feats)
        #stop_probs = self.sigmoid(stop_probs)

        return out_heatmap, stop_probs, hidden_list
        #return out_heatmap, hidden_list #

