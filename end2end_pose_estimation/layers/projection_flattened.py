from constrained_pose_estimation.my_imports import *

class Projection(nn.Module):


    def __init__(self):
        super(Projection, self).__init__()
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        N = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)

        x_flat = x.view(N, C, -1)
        #x_flat = self.softmax(x_flat)
        z = self.applySimplexProjection(x_flat, H, W)
        return z

    def applySimplexProjection(self, x, H, W):
        print('Performing simplex projection')
        num_labels = x.shape[1]
        total_result = torch.tensor(x)
        total_result = total_result.view(x.shape[0], x.shape[1], H, W)
        for i in range(x.shape[0]):
            V_im = x[i]
            res = torch.tensor(V_im)
            heatmap_size = V_im.shape[1]
            for l in range(num_labels-1):
                #if(torch.isnan(poses[i][0][l]).any()):
                #    continue
                raw_sum = res[l,:].sum()
                pixel_count = 1
                if(raw_sum<pixel_count):
                    res[l,:] =  res[l,:] + (pixel_count-raw_sum)/heatmap_size
                else:
                    Z, indices = torch.sort(res[l,:], descending=True)
                    Z_cumsum = torch.cumsum(Z,dim=0)
                    frac = (1.0 / torch.range(1,heatmap_size).cuda())
                    T = Z - torch.dot(frac, Z_cumsum-pixel_count)
                    max_i = (T < 0).nonzero() #-1 for the last element larger than 0
                    if(max_i.shape[0]<1):
                        j = len(Z)-1
                    else:
                        j = max_i[0]-1
                    theta = Z[j] - T[j]
                    res[l, :] = res[l,:]-theta

            total_result[i, ...] = res.view(res.shape[0], H, W)
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

