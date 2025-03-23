from constrained_pose_estimation.my_imports import *

class SoftArgmax(nn.Module):


    def __init__(self):
        super(SoftArgmax, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input tensor (N, C, H, W)
        :return:
        """

        N = x.size(0)
        nu_persons = x.shape[1]
        C = x.size(2)
        H = x.size(3)
        W = x.size(4)

        # collapse
        res = torch.zeros(N, nu_persons, C,2)
        for k in range(nu_persons):
            temp = x[:,k].view(N, C, -1)
            weights = self.softmax(temp)

            range_ = torch.arange(H * W).cuda()

            indices = range_.unsqueeze(0).unsqueeze(0).expand(weights.size())

            semi_indices = (weights * indices.float()).sum(dim=-1)
            indices_x = semi_indices / H
            indices_y = semi_indices % W
            res[:,k,:,0] = indices_x.float()/46.0
            res[:,k,:,1] = indices_y.float()/46.0

        return res