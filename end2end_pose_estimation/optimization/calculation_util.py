from constrained_pose_estimation.my_imports import *

hungarian = Hungarian()
from constrained_pose_estimation.Global_Constants import limbSeq
import math

from constrained_pose_estimation.layers.soft_argmax import SoftArgmax
soft_argmax = SoftArgmax()
import cv2 as cv
import constrained_pose_estimation.general_helper.helper_functions as hf

def calc_pairwise_dist(hm_out, hm_gt, reshape=True):
    batch_size = hm_gt.shape[0]
    nu_persons = hm_gt.shape[1]
    nu_joints = hm_gt.shape[2]
    nu_persons_hat = hm_gt.shape[1]
    if reshape: #for the cnn output
        hm_out = hm_out.view(len(hm_out), nu_persons, nu_joints, hm_out.shape[2], hm_out.shape[3])
        nu_persons_hat = len(hm_out[0])
    dist = torch.zeros(batch_size, nu_persons, nu_persons_hat).cuda()
    for b in range(batch_size):
        for k in range(nu_persons):
            for k_hat in range(nu_persons_hat):
                for j in range(nu_joints):
                    dist[b][k][k_hat] += torch.norm(hm_gt[b][k][j].float()-hm_out[b][k_hat][j])
    return dist

#fast_version
def calc_pairwise_dist_nonbatch(hm_out, hm_gt, reshape=True):
    nu_persons = hm_gt.shape[0]
    nu_persons_hat = len(hm_out)
    nu_joints = hm_gt.shape[1]
    #hm_out = torch.cat(hm_out,0).view(nu_persons_hat, hm_out[0].shape[0], hm_out[0].shape[1], hm_out[0].shape[2])
    hm_gt = hm_gt.unsqueeze(1).expand(-1,nu_persons_hat, -1, -1, -1)
    hm_out = hm_out.unsqueeze(0).expand(nu_persons,-1, -1, -1, -1)
    hm_out = hm_out.contiguous().view(nu_persons_hat*nu_persons, nu_joints* hm_out.shape[3]* hm_out.shape[4])
    hm_gt = hm_gt.contiguous().view(nu_persons_hat * nu_persons, nu_joints* hm_gt.shape[3]* hm_gt.shape[4])
    dist = torch.norm(hm_gt-hm_out,dim=-1)
    dist = dist.reshape(nu_persons, nu_persons_hat).unsqueeze(0)
   
    return dist


def calc_pairwise_dist_nonbatch_slow(hm_out, hm_gt, reshape=True):
    nu_persons = hm_gt.shape[0]
    nu_persons_hat = len(hm_out)
    nu_joints = hm_gt.shape[1]
    if reshape: #for the cnn output
        hm_out = hm_out.view(nu_persons, nu_joints, hm_out.shape[1], hm_out.shape[2])
        nu_persons_hat = len(hm_out)
    #current_heat_gt = current_heat_gt.view(current_heat_gt.shape[0], J, max_k, current_heat_gt.shape[2]*current_heat_gt.shape[3])
    batch_size = 1
    dist = torch.zeros(batch_size, nu_persons, nu_persons_hat).cuda()

    for k in range(nu_persons):
        for k_hat in range(nu_persons_hat):
            #for j in range(nu_joints):
            dist[0][k][k_hat] = torch.norm(hm_gt[k].cuda().float()-hm_out[k_hat])
    return dist

def calc_assignment(pairwise_dist):
    #print(pairwise_dist.shape)
    nu_batches = pairwise_dist.shape[0]
    nu_persons = max(pairwise_dist.shape[1], pairwise_dist.shape[2])
    nu_joints = pairwise_dist.shape[2]
    assignments = torch.zeros(nu_batches, nu_persons, 2) # 2 for the assignment pair
    for b in range(nu_batches):
        hungarian.calculate(pairwise_dist[b])
        res = hungarian.get_results()
        if res is None:
            return None
        assignments[b] = torch.from_numpy(np.asarray(res))
    return assignments


def calc_assignment_non_batch(pairwise_dist):
    #print(pairwise_dist.shape)
    #nu_batches = pairwise_dist.shape[0]
    #nu_persons = max(pairwise_dist.shape[1], pairwise_dist.shape[2])
    #nu_joints = pairwise_dist.shape[2]
    #assignments = torch.zeros(nu_batches, nu_persons, 2) # 2 for the assignment pair
    #for b in range(nu_batches):

    hungarian.calculate(pairwise_dist)
    res = hungarian.get_results()
    if res is None:
        return None
    assignments = torch.from_numpy(np.asarray(res))
    return assignments


#probably an obsoluete function
def calc_pairwise_dist_obs(out, current_heat_gt):
    #out = out.view(out.shape[0], J, max_k, out.shape[2]*out.shape[3])
    #out = out.view(out.shape[0], out.shape[1], out.shape[2], out.shape[2] * out.shape[3])
    #current_heat_gt = current_heat_gt.view(current_heat_gt.shape[0], J, max_k, current_heat_gt.shape[2]*current_heat_gt.shape[3])

    nu_joints = out[0].shape[0]
    nu_persons = current_heat_gt[0].shape[0]
    current_heat_gt = current_heat_gt.permute(1, 0, 2, 3)
    dist = torch.zeros(nu_joints, nu_persons, nu_persons)
    for j in range(nu_joints):
        for k in range(nu_persons):
            for k_hat in range(nu_persons):
                if k>=len(out):
                    val = 0
                else:
                    val = out[k][j]
                dist[j][k][k_hat] = torch.norm(val-current_heat_gt[k_hat][j])
    return dist

#probably an obsoluete function
def calc_assignment_obs(pairwise_dist):

    nu_joints = pairwise_dist.shape[0]
    nu_persons = pairwise_dist.shape[1]
    assignments = torch.zeros(nu_joints, nu_persons, 2) # 2 for pair

    for j in range(nu_joints):
        hungarian.calculate(pairwise_dist[j])
        res = hungarian.get_results()
        if res is None:
            return None
        assignments[j] = torch.from_numpy(np.asarray(res))
    return assignments



# the arguments are the gt PAFs and Heatmaps
soft_argmax_noncuda = SoftArgmax()
def calculate_associations_scores_gt(paf, hm,is_cuda=False):
    nu_limbs = 19
    nu_joints = 18
    nu_persons = paf.shape[0]
    dim1 = paf.shape[2]
    dim2 = paf.shape[3]
    num_intermed_pts = 10
    threshold = 0.2
    final_scores = torch.zeros((nu_persons, nu_joints, dim1, dim2)) # will contain value zero for non candidates and integral of the relevant PAF scores
    if is_cuda:
        final_scores = final_scores.cuda()
    for k in range(nu_persons): #iterator over persons
        for l in range(nu_limbs):
            score_mid = paf[k,l:l+2] # the scores of person k, limb type l, x and y coordinate
            limb_source_ind = limbSeq[l][0]
            limb_target_ind = limbSeq[l][1]
            candA = np.unravel_index(np.argmax(hm[k][limb_source_ind]), hm[k][limb_source_ind].shape) #source
            candB = np.unravel_index(np.argmax(hm[k][limb_target_ind]), hm[k][limb_source_ind].shape) #source

            #candA = soft_argmax_noncuda(hm[k][limb_source_ind])
            #candB = soft_argmax_noncuda(hm[k][limb_target_ind])
            #print(candA, candB)
            nA = 1 #len(candA[0])
            nB = 1 #len(candB[0])
            if nA == 0 or nB == 0:
                continue
            for i in range(nA):
                for j in range(nB):
                    #currentACand = [candA[0], candA[1]]
                    #currentBCand = [candB[0], candB[1]]
                    vec = np.subtract(candB, candA)
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    vec = np.divide(vec, norm + 1e-5)

                    startend = list(zip(np.linspace(candA[0], candB[0], num=num_intermed_pts),
                                        np.linspace(candA[1], candB[1], num=num_intermed_pts)))



                    vec_x = np.array([score_mid[0, int(round(startend[I][0])), int(round(startend[I][1]))-1]
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[1, int(round(startend[I][0])), int(round(startend[I][1]))-1]
                                      for I in range(len(startend))])

                    mult = 1

                    score_midpts = np.multiply(np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1]), mult)
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) \
                                            + min(0.5 * paf[k][l].shape[0] / (norm + 1e-5) - 1, 0)
                    #if is_cuda:
                    #    score_with_dist_prior = score_with_dist_prior.cuda()
                    final_scores[k,limb_source_ind , candA] = score_with_dist_prior
                    final_scores[k, limb_target_ind, candB] = score_with_dist_prior
                    #print(score_with_dist_prior)
    return final_scores



def calculate_associations_scores(paf, hm):
    nu_limbs = 19
    nu_joints = 18
    nu_persons = paf.shape[0]
    dim1 = paf.shape[2]
    dim2 = paf.shape[3]
    num_intermed_pts = 10
    threshold = 0.2
    final_scores = torch.zeros((nu_persons, nu_joints, dim1, dim2)).cuda() # will contain value zero for non candidates and integral of the relevant PAF scores

    for k in range(nu_persons): #iterator over persons
        for l in range(nu_limbs):
            score_mid = paf[k,l:l+2] # the scores of person k, limb type l, x and y coordinate
            limb_source_ind = limbSeq[l][0]
            limb_target_ind = limbSeq[l][1]
            #candA = np.where(hm[k][limb_source_ind]>threshold) #source
            #candB = np.where(hm[k][limb_target_ind] > threshold) #target
            candA = soft_argmax(hm[k][limb_source_ind])
            candB = soft_argmax(hm[k][limb_target_ind])
            nA = 1#len(candA[0])
            nB = 1#len(candB[0])
            if nA == 0 or nB == 0:
                continue
            for i in range(nA):
                for j in range(nB):
                    currentACand = [candA[0], candA[1]]
                    currentBCand = [candB[0], candB[1]]
                    vec = np.subtract(currentBCand, currentACand)
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    vec = np.divide(vec, norm + 1e-5)

                    startend = list(zip(torch.linspace(candA[0].int(), candB[0].int(), steps=num_intermed_pts),
                                        torch.linspace(candA[1].int(), candB[1].int(), steps=num_intermed_pts)))


                    vec_x = torch.zeros(len(startend)).cuda()
                    vec_y = torch.zeros(len(startend)).cuda()
                    for I in range(len(startend)):
                        vec_x[I] = score_mid[0, startend[I][0].int(), startend[I][1].int()]
                        vec_y[I] = score_mid[1, startend[I][0].int(), startend[I][1].int()]
                        # vec_x[I] = score_mid[0, torch.round(startend[I][0]).int(), torch.round(startend[I][1]).int()]
                        # vec_y[I] = score_mid[1, torch.round(startend[I][0]).int(), torch.round(startend[I][1]).int()]

                    mult = 1

                    score_midpts = ((vec_x * vec[0])+ (vec_y * vec[1]) * mult)
                    score_with_dist_prior = torch.sum(score_midpts)/len(score_midpts)
                    addition = 0.5 * paf[k][l].shape[0] / (norm + 1e-5) - 1
                    if(addition<0):
                        score_with_dist_prior = score_with_dist_prior + addition

                    final_scores[k,limb_source_ind , currentACand] = score_with_dist_prior
                    final_scores[k, limb_target_ind, currentBCand] = score_with_dist_prior
                    #print(score_with_dist_prior)
    return final_scores



def get_dt_poses(img_name, heatmaps):
    oriImg = cv.imread(img_name)
    heatmaps = hf.resize_heatmap(oriImg, heatmaps, is_cuda=True)
    keypoints = np.zeros((len(heatmaps), heatmaps[0].shape[0]-1, 2))
    for k in range(len(heatmaps)):  # len(poses)):
        for j in range(len(heatmaps[0]) - 1):
            current_heatmap = heatmaps[k][j]
            keypoint = np.argmax(current_heatmap.flatten())
            keypoint = np.unravel_index(keypoint,current_heatmap.shape)
            keypoints[k,j] = keypoint[0], keypoint[1]
    return keypoints