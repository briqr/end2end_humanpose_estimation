from constrained_pose_estimation.my_imports import *

# the softargmax matched for single heatmaps (used for calculating the broken symmetry penalty terms)
class SoftArgmax(nn.Module):


    def __init__(self):
        super(SoftArgmax, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input tensor (N, C, H, W)
        :return:
        """

        N = x.size(0) #number of persons
        C = x.size(1) # the relevant heatmaps with the symmetry
        H = x.size(2) # the heatmap dims
        W = x.size(3)

        # collapse
        res = torch.zeros(N, C, 2)
        for k in range(N):
            tmp = x[k].view(C, -1)
            weights = self.softmax(tmp.double())

            range_ = torch.arange(H * W).cuda()

            indices = range_.unsqueeze(0).expand(weights.size())

            semi_indices = (weights * indices.double()).sum(dim=-1)
            indices_x = semi_indices / H
            indices_y = semi_indices % W
            res[k,:,0] = indices_x#.float()
            res[k,:,1] = indices_y#.float()
        print(res)
        return res