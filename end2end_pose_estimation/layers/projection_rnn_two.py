from constrained_pose_estimation.my_imports import *

class Projection(nn.Module):

    num_invoked = 0
    def __init__(self):
        super(Projection, self).__init__()
        self.softmax_op = nn.Softmax(dim=-1)
        self.softmax_dim_hm = nn.Softmax(dim=0)

        self.alpha=10

    def forward(self, x):
        self.num_invoked += 1
        if self.num_invoked%500 ==0 and self.alpha>1:
            self.alpha/=1.02
        #z = self.applySimplexProjection(x)
        z = self.multiplyByPeaked(x)
        return z

    def applySimplexProjection(self, x):
        #temp = x
        batch_size = x.shape[0]
        nu_joints = 19
        nu_persons = x.shape[1]//nu_joints
        dim1 = x.shape[2]
        dim2 = x.shape[3]
        x = x.reshape(batch_size, nu_persons, nu_joints, dim1, dim2)
        alpha = self.alpha
        total_result = torch.tensor(x)
        for i in range(x.shape[0]):
            for p in range(x.shape[1]):
                V_im = alpha*x[i][p]
                V_im = V_im.reshape(V_im.shape[0], dim1*dim2)
                #V_exp = torch.exp(V_im)
                #V_exp_sum = torch.sum(V_exp, dim=1)
                #V_exp_sum = V_exp_sum.unsqueeze(1).expand(nu_joints, dim1*dim2)
                #sm_man = V_exp/V_exp_sum
                V_im = self.softmax_op(V_im)
                print(V_im)
                V_im = V_im.reshape(V_im.shape[0], dim1, dim2)
                res = torch.tensor(V_im)
                heatmap_size = V_im.shape[1] * V_im.shape[2]
                for l in range(nu_joints - 1): # the last joint heatmap is background,maybe later i can apply a contraints using the exact number of peaks
                    peak_count = 1 # it can be zero or one, for instance, if the joint is occluded and not annotated
                    raw_sum = res[l, :, :].sum()
                    if (peak_count == 0):
                        continue
                    if (raw_sum < peak_count):
                        res[l, :, :] = res[l, :, :] + (peak_count - raw_sum) / heatmap_size
                    else:
                        Z, indices = torch.sort(res[l, :, :].reshape(heatmap_size), descending=True)
                        Z_cumsum = torch.cumsum(Z, dim=0)
                        Z_cumsum = Z_cumsum-peak_count
                        frac = (1.0 / torch.range(1, heatmap_size).cuda())  # .reshape(x.shape[2], x.shape[3])).cuda()
                        T = Z - frac * Z_cumsum
                        max_i = (T < 0).nonzero() # the < condition returns a boolean matrix
                        if (max_i.shape[0] < 1):
                            j = len(Z)-1
                        else:
                            j = (max_i[0])
                        theta = (torch.sum(Z[0:j])-peak_count)/float(j)

                        res[l, :, :] = res[l, :, :] - theta
                        res[l, res[l]<0] = 0
                        print('sum***')
                        print(res[l].sum())

            #print(total_result)
            total_result[i, ...] = res
        return total_result



    def applySinkhornProjection(self, x, poses, length_encoding):
        num_iters = 1
        num_labels = x.shape[1]
        total_result = torch.tensor(x)
        #for l in range(num_labels):  # go over each column for size constraints
        #    true_nu = (gt_im == l).sum().float()
        count_peaks = 3 #self.count_peaks(poses, length_encoding)
        for i in range(x.shape[0]):
            V_im = x[i]
            heatmap_size = V_im.shape[1] * V_im.shape[2]

            res = torch.tensor(V_im)  # np.zeros(V.shape)

            for iter in range(num_iters):
                for l in range(num_labels-1):  # go over each column for size constraints

                    raw_size = res[l,:,:].sum()
                    true_nu = count_peaks[i, l].cuda()
                    if(true_nu==0):
                        res[l,...] = 0
                        continue
                    if (raw_size==true_nu): # <= true_nu+1 and raw_size>=true_nu-1 and iter >=2):
                        continue
                    if (raw_size < 0.001):
                        ind_i = 23 #res[l].shape[0]/2
                        ind_j = 23 #res[l].shape[1]/2
                        offset = 10
                        res[l, ind_i,:] += 0.1
                        res[l, :, ind_j] += 0.1
                        raw_size = res[l,:,:].sum()
                    res[l,:,:] = (float)(true_nu) * res[l,:,:] / raw_size
                #for m in range(V_im.shape[1]):  # go over rows for the probability constraint
                #    for n in range(V_im.shape[2]):
                #        pass
                #        raw_prob_sum = res[:,m,n].sum() + 0.001
                #        current_val = res[:,m,n]
                #        res[:,m,n] *= 1.0/raw_prob_sum
            #if False:
            #    for l in range(num_labels):
            #        true_nu = 1#(gt_im == l).sum().float()
            #        print 'true, actual', l, true_nu.data.cpu(), res[l,...].sum().data.cpu()
            visualise = False
            if visualise:
                self.visualise_heatmaps(V_im, res)

            total_result[i,...] = res

        return total_result



    def multiplyByPeaked(self, x):

        batch_size = x.shape[0]
        nu_joints = 19
        #nu_persons = x.shape[1] // nu_joints
        dim1 = x.shape[2]
        dim2 = x.shape[3]
        x = x.reshape(batch_size, nu_joints, dim1, dim2)
        alpha = self.alpha
        total_result = torch.tensor(x)
        for i in range(x.shape[0]):
            V_im = alpha * x[i]
            V_im = V_im.reshape(V_im.shape[0], dim1 * dim2)
            V_im = self.softmax_op(V_im[0:nu_joints-1])
            V_im = V_im.reshape(V_im.shape[0], dim1, dim2)
            heatmap_size = V_im.shape[1] * V_im.shape[2]
            #for l in range(nu_joints - 1):  # the last joint heatmap is background,maybe later i can apply a contraints using the exact number of peaks
            total_result[i,0:nu_joints-1] =  V_im[0:nu_joints-1] * x[i,0:nu_joints-1]

        return total_result

