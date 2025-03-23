from constrained_pose_estimation.my_imports import *

class SoftArgmax(nn.Module):


    def __init__(self):
        super(SoftArgmax, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, x):
        """

        :param x: input tensor (N, C, H, W)
        :return:
        """

        #N = x.size(0)
        #C = x.size(1)
        H = x.shape[0]
        W = x.shape[1]
        alpha = 2
        x = alpha*x
        # collapse
        temp = x.view(H*W)#.view(N, C, -1)
        weights = self.softmax(temp)

        range = torch.arange(H*W) #, device=x.get_device()

        indices = range#.unsqueeze(0).expand(weights.size())

        semi_indices = (weights.float() * indices.float()).sum()
        #print(semi_indices)
        indices_x = semi_indices / H
        indices_y = semi_indices % W
        #print(int(indices_x), int(indices_y))
        return indices_x, indices_y




    # def __init__(self):
    #     super(SoftArgmax, self).__init__()
    #
    #
    # def forward(self, x):
    #     nu_persons = x.shape[1]
    #     nu_joints = x.shape[2]
    #     output = torch.zeros(nu_persons, nu_joints) # the joints locations for every person, for every joint
    #     for s in range (x.shape[0]): #sample
    #         for k in range(x.shape[1]): #person
    #             for j in range(x.shape[2]): # joint
    #                 hm = x[s][k][j]
    #                 dim1 = hm.shape[0]
    #                 dim2 = hm.shape[1]
    #                 W = torch.zeros(dim1, dim2, 2) # two for coordinate (x,y)
    #                 x_grid = torch.arange(1, dim1+1).expand
    #                 y_grid = torch.arange(1, dim2+1)
    #                 W[:,:,0] = x_grid/dim1
    #                 W[:,:,1] = y_grid/dim2
    #
    #
    #
    #     return W
    #
    #
