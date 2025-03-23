from constrained_pose_estimation.my_imports import *
import math
from constrained_pose_estimation.Global_Constants import limbSeq
import constrained_pose_estimation.optimization.calculation_util as calc_util
class PafScoreCalculator(nn.Module):


    def __init__(self):
        super(PafScoreCalculator, self).__init__()

    def forward(self, x):
        """

        :param x: input tensor (N, C, H, W)
        :return:
        """
        threshold = 0.1
        N = x.shape[0]
        C = x.shape[1]
        H = x.shape[3]
        W = x.shape[4]
        max_k = 3
        num_joints = 19
        num_limbs = 19

        paf_scores = torch.zeros(N, max_k, num_joints-1, H, W)
        for n in range(N):
            joints_hms = x[n, :, 0:num_joints, :, :]
            pafs = x[n, :, num_joints:, :, :]
            current_scores = calc_util.calculate_associations_scores(pafs, joints_hms).cuda()
            paf_scores[n] = current_scores
        return paf_scores


#calculate association scores between the heatmaps of the same person

# def calculate_associations_scores(paf, hm):
#     nu_limbs = 19
#     nu_joints = 18
#     nu_persons = paf.shape[0]
#     dim1 = paf.shape[2]
#     dim2 = paf.shape[3]
#     num_intermed_pts = 10
#     threshold = 0.1
#     final_scores = torch.zeros((nu_persons, nu_joints, dim1, dim2)) # will contain value zero for non candidates and integral of the relevant PAF scores
#     for k in range(nu_persons): #iterator over persons
#         for l in range(nu_limbs):
#             score_mid = paf[k,l:l+2]
#             limb_source_ind = limbSeq[l][0]
#             limb_target_ind = limbSeq[l][1]
#             candA = np.where(hm[k][limb_source_ind]>threshold) #source
#             candB = np.where(hm[k][limb_target_ind] > threshold) #target
#             nA = len(candA[0])
#             nB = len(candB[0])
#             if nA == 0 or nB == 0:
#                 continue
#             for i in range(nA):
#                 for j in range(nB):
#                     vec = np.subtract(candB[j][:2], candA[i][:2])
#                     norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
#                     vec = np.divide(vec, norm + 1e-5)
#
#                     startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=num_intermed_pts),
#                                         np.linspace(candA[i][1], candB[j][1], num=num_intermed_pts)))
#
#                     vec_x = np.array([score_mid[int(0, round(startend[I][1])), int(round(startend[I][0]))]
#                                       for I in range(len(startend))])
#                     vec_y = np.array([score_mid[int(1, round(startend[I][1])), int(round(startend[I][0]))]
#                                       for I in range(len(startend))])
#
#                     mult = 1
#
#                     score_midpts = np.multiply(np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1]), mult)
#                     score_with_dist_prior = sum(score_midpts) / len(score_midpts) \
#                                             + min(0.5 * paf[k][l].shape[3] / (norm + 1e-5) - 1, 0)
#                     final_scores[k,limb_source_ind , candA[0:2]] = score_with_dist_prior
#                     final_scores[k, limb_target_ind, candB[0:2]] = score_with_dist_prior
#     #print ('training scores: ' + final_scores)
#     return final_scores
#
#             # connections_cands = []
#             # limb_intermed_coords[2, :] = paf_xy_coords_per_limb[limb_type][0]
#             #
#             # score_mid = paf[:, :, [x for x in limb_map[l]]]
#             # for i, joint_src in enumerate(above_thr_src_indices):
#             #
#             #     for j, joint_dst in enumerate(above_thr_target_indices):
#             #
#             #         limb_dir = joint_dst[:2] - joint_src[:2]
#             #         limb_dist = np.sqrt(np.sum(limb_dir**2)) + 1e-8
#             #         limb_dir = limb_dir / limb_dist
#             #
#             #         intermed_paf = paf[k,above_thr_src_indices[0, :],
#             #                                   limb_intermed_coords[1, :], limb_intermed_coords[2:4, :]].T
#             #
#             #         score_intermed_pts = intermed_paf.dot(limb_dir)
#             #         score_penalizing_long_dist = score_intermed_pts.mean(
#             #         ) + min(0.5 * paf.shape[0] / limb_dist - 1, 0)
#             #
#             #         startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=num_intermed_pts),
#             #                             np.linspace(candA[i][1], candB[j][1], num=num_intermed_pts)))
#             #
#             #         vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
#             #                           for I in range(len(startend))])
#             #         vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
#             #                           for I in range(len(startend))])
#             #
#             #         mult = 1
#             #
#             #         score_midpts = np.multiply(np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1]), mult)
#             #         score_with_dist_prior = sum(score_midpts) / len(score_midpts) \
#             #                                 + min(0.5 * img.shape[0] / (norm + 1e-5) - 1, 0)
        # return score_penalizing_long_dist