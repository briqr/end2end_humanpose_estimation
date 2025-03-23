from constrained_pose_estimation.my_imports import *
import matplotlib
import pylab as plt
import matplotlib.pyplot as pp
import cv2 as cv
import math
import constrained_pose_estimation.general_helper.helper_functions as hf

pp.ion()

#factor = 1.001 # factor by which we divide the maximum value
size_1 = 7
size_2 = 7
def visualise_heatmaps(orig, result):

    nu_persons = min(len(result), len(orig))
    for k in range(nu_persons): # iterating over the joints


        for j in range(result[0].shape[0]):
            fig = plt.figure()
            ax = fig.add_subplot('211')
            plt.title('heatmap features')
            plt.axis('off')
            res_after_np1 = result[k][j].detach().cpu().numpy()
            res_after_np1[res_after_np1 < 0] = 0

            orig_np = orig[k][j].detach().cpu().numpy()
            orig_np[orig_np < 0] = 0

            ax = fig.add_subplot('211')
            ax.imshow(orig_np)
            plt.title('gt, j=%d, person=%d' %(j,k+1))

            ax = fig.add_subplot('212')
            ax.imshow(res_after_np1)
            plt.title('result, j=%d, person=%d' % (j, k + 1))

            plt.axis('off')
            plt.show()
            plt.waitforbuttonpress(0)
            plt.close()

        if False:
            fig = plt.figure()
            ax = fig.add_subplot(str(nu_persons)+'11')
            ax.imshow(orig[j])
            plt.title('heatmap features')
            plt.axis('off')
            # when taking the argmax
            for k in range(nu_persons): #iterating over the joints
                res_after_np1 = result[k][j].detach().cpu().numpy()
                max_val = 0 #np.max(res_after_np1)/5
                res_after_np1[res_after_np1 < max_val] = 0
                ax = fig.add_subplot('72'+str(k+2))
                ax.imshow(res_after_np1)
                plt.title('argmax result, j=%d, person=%d' % (j, k + 1))
                plt.axis('off')
            plt.show()
            plt.waitforbuttonpress(0)
            plt.close()

def moveon(event):
    pp.close()


def visualise_heatmaps_per_person_gt(im_name, result):
    nu_persons = len(result) if len(result) <= 7 else 7
    nu_joints = result[0].shape[0] - 1  # the last heatmap is for background
    im = cv.imread(im_name) #aug_image.detach().cpu().numpy()[0].transpose([1,2,0])
    # fig = plt.figure(figsize=(size_1, size_2))
    # fig.suptitle(title)

    fig = pp.figure()
    cid = fig.canvas.mpl_connect('key_press_event', moveon)
    ax = fig.add_subplot('%d11' % (nu_persons + 2))
    ax.imshow(im)
    plt.title('image')
    plt.axis('off')

    for k in range(nu_persons):  # iterating over the persons
        total_res = np.zeros((nu_persons, result[0].shape[1], result[0].shape[2]))
        for j in range(nu_joints):
            # print('visualising the results for person ' + str(k))
            res_after_np1 = result[k][j]
            if res_after_np1.is_cuda:
                res_after_np1 = res_after_np1.detach().cpu().numpy()
            # res_after_np1[res_after_np1 < general_threshold] = 0
            total_res[k] = total_res[k] + res_after_np1
        ax = fig.add_subplot('%d1%d' % (nu_persons + 2, k + 3))
        ax.imshow(total_res[k])
        plt.title('person=%d' % (k + 1))
        plt.axis('off')

    plt.show()
    plt.waitforbuttonpress(0)  # this will wait for indefinite time
    plt.close()






def visualise_heatmaps_gt_per_person(im_name, result, gt):
    nu_persons = len(gt) if len(gt)<=7 else 7
    nu_joints = result[0].shape[0]-1 # the last heatmap is for background

    for k in range(nu_persons): # iterating over the persons
        total_res = np.zeros((nu_persons, result[0].shape[1], result[0].shape[2]))
        total_res_gt = np.zeros((nu_persons, result[0].shape[1], result[0].shape[2]))
        fig = pp.figure(figsize=(size_1, size_2))
        cid = fig.canvas.mpl_connect('key_press_event', moveon)
        ax = fig.add_subplot('311')
        im = cv.imread(im_name)
        ax.imshow(im)
        plt.title('image')
        plt.axis('off')
        for j in range(nu_joints):
            #print('visualising the results for person ' + str(k))
            res_after_np1 = result[k][j]
            res_gt_np = gt[k][j]
            if res_after_np1.is_cuda:
                res_after_np1 = res_after_np1.detach().cpu().numpy()
                res_gt_np = res_gt_np.detach().cpu().numpy()
            max_val = np.max(res_after_np1)/1.5
            res_after_np1[res_after_np1 < max_val] = 0
            #res_after_np1[res_after_np1 < general_threshold] = 0
            if k < len(result):
                total_res[k] = total_res[k] + res_after_np1
            total_res_gt[k] = total_res_gt[k] + res_gt_np
        print(total_res_gt.sum())
        ax = fig.add_subplot('312')
        ax.imshow(total_res_gt[k])
        plt.title('gt person=%d' % (k + 1))
        plt.axis('off')
        if k < len(result):
            ax = fig.add_subplot('313')
            ax.imshow(total_res[k])
            plt.title('out person=%d' %(k+1))
            plt.axis('off')



        plt.show()
        plt.waitforbuttonpress(0)  # this will wait for indefinite time
        plt.close()




def visualise_heatmaps_per_person(im_name, result, gt=None, title='result', is_rnn=True, hm_combined_features=None):
    nu_persons = len(result) if len(result)<=7 else 7
    nu_joints = result[0].shape[0]-1 # the last heatmap is for background
    im = cv.imread(im_name)
    #fig = plt.figure(figsize=(size_1, size_2))
    #fig.suptitle(title)

    fig = pp.figure(figsize=(size_1, size_2))
    cid = fig.canvas.mpl_connect('key_press_event', moveon)
    ax = fig.add_subplot('%d11' %(nu_persons + 2))
    ax.imshow(im)
    plt.title('image')
    plt.axis('off')
    if hm_combined_features is not None:
        hm_combined_features = hm_combined_features.detach().cpu().numpy()
        total_combined = np.zeros((hm_combined_features.shape[1], hm_combined_features.shape[2]))
        for j in range(nu_joints):
            # print('visualising the results for person ' + str(k))
            total_combined = total_combined + hm_combined_features[j]
        ax = fig.add_subplot('%d12' % (nu_persons + 2))
        ax.imshow(total_combined)
        plt.title('combined hm')
        plt.axis('off')
    for k in range(nu_persons): # iterating over the persons
        total_res = np.zeros((nu_persons, result[0].shape[1], result[0].shape[2]))
        for j in range(nu_joints):
            #print('visualising the results for person ' + str(k))
            res_after_np1 = result[k][j]
            if is_rnn and res_after_np1.is_cuda:
                res_after_np1 = res_after_np1.detach().cpu().numpy()
            max_val = np.max(res_after_np1)/1.5
            res_after_np1[res_after_np1 < max_val] = 0
            #res_after_np1[res_after_np1 < general_threshold] = 0
            total_res[k] = total_res[k] + res_after_np1
        ax = fig.add_subplot('%d1%d' % (nu_persons + 2, k + 3))
        ax.imshow(total_res[k])
        plt.title('person=%d' %(k+1))
        plt.axis('off')
    plt.savefig('/media/pose_estimation/constrained_results/val/' + im_name.split('/')[-1])
    plt.show()
    plt.waitforbuttonpress(0)  # this will wait for indefinite time
    plt.close()

# if gt is not None:
#     gt_i = k + 4
#     ax = fig.add_subplot('42' + str(gt_i))
#     ax.imshow(gt[k][j])
#     plt.title('gt, person=%d' % (k + 1))
#     plt.axis('off')
def visualise_heatmaps_per_person_supression(im_name, result, title='result suppression', is_rnn=False, hm_combined_features=None):
    nu_persons = len(result)
    nu_joints = result[0].shape[0] - 1  # the last heatmap is for background
    im = cv.imread(im_name)
    fig = plt.figure(figsize=(size_1, size_2))
    fig.suptitle(title)
    ax = fig.add_subplot('%d11'%(nu_persons+2))
    ax.imshow(im)
    plt.title('image')
    plt.axis('off')

    if hm_combined_features is not None:
        hm_combined_features = hm_combined_features.detach().cpu().numpy()
        total_combined = np.zeros((hm_combined_features.shape[1], hm_combined_features.shape[2]))
        for j in range(nu_joints):
            # print('visualising the results for person ' + str(k))
            total_combined = total_combined + hm_combined_features[j]
        ax = fig.add_subplot('%d12' % (nu_persons + 2))
        ax.imshow(total_combined)
        plt.title('combined hm')
        plt.axis('off')


    total_res = np.zeros((nu_persons, result[0].shape[1], result[0].shape[2]))
    if is_rnn:
        arr_result = torch.zeros(nu_persons, nu_joints + 1, result[0][0].shape[0], result[0][0].shape[1])
        for k in range(nu_persons):
            arr_result[k] = result[k]
        result = arr_result
    for j in range(nu_joints):
        res_after_np1 = result[:,j]
        if is_rnn:
            res_after_np1 = res_after_np1.detach().cpu().numpy()
        max_val = 0#np.max(res_after_np1)/1.1
        res_after_np1[res_after_np1 < max_val] = 0
        max_val, max_ind = np.max(res_after_np1,axis=0), np.argmax(res_after_np1,axis=0)
        for k in range(nu_persons):  # iterating over the persons
            k_max_ind = np.where(max_ind==k)
            total_res[k][k_max_ind] = total_res[k][k_max_ind] + max_val[k_max_ind]
    for k in range(nu_persons):  # iterating over the persons
        ax = fig.add_subplot('%d1%d' % (nu_persons + 2, k + 3))
        ax.imshow(total_res[k])
        plt.title('person=%d' % (k + 1))
        plt.axis('off')
    plt.show()



def visualise_limbs_per_person(im_name, result, title='PAF'):
    nu_persons = len(result)
    nu_limbs = result[0].shape[0]
    im = cv.imread(im_name)
    fig = plt.figure(figsize=(size_1, size_2))
    fig.suptitle(title)
    for k in range(nu_persons): # iterating over the persons
        ax = fig.add_subplot('711')
        ax.imshow(im)
        plt.title('image')
        plt.axis('off')
        total_res = np.zeros((nu_persons, result[0].shape[1], result[0].shape[2]))
        for j in range(nu_limbs):
            res_after_np1 = res_after_np1.detach().cpu().numpy()
            total_res[k] = total_res[k] + res_after_np1
        ax = fig.add_subplot('71'+str(k+2))
        ax.imshow(total_res[k])
        plt.title('result, person=%d' %(k+1))
        plt.axis('off')
    plt.show()



# find connection in the specified sequence, center 29 is in the position 15# find c
# These connections match the joints ordering in the original framework (PAF)
limbSeq_nonadapted = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
                      [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]

#THE SEQUENCE THAT MATCHES THE GT, based on the joints ordering in Uni-Bonn implementation
limbSeq = [ [17,6], [17, 5], [6,8], [8,10], [5,7], [7,9], [17,12], [12, 14], \
            [14,16], [17,11], [11,13], [13,15], [17,0], [0,1], [1,3],\
            [0,2], [2,4], [6, 3], [5,4]]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
cmap = matplotlib.cm.get_cmap('hsv')


def visualise_joints(img_name, poses, flip=True):
# visualize: code taken from https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/testing/python/demo.ipynb

    oriImg = cv.imread(img_name)
    canvas = cv.imread(img_name) # B,G,R order

    for j in range(len(poses[0])):
        J = 18
        rgba = np.array(cmap(1 - j/18. - 1./36))
        rgba[0:3] *= 255
        for k in range(1): #len(poses)):
            if np.isnan(poses[k][j]).any() or poses[k][j][0] > oriImg.shape[1] or poses[k][j][0]>oriImg.shape[1]:
                continue
            x, y = poses[k][j][0:2]
            if flip:
                cv.circle(canvas, (int(y),int(x)), 4, colors[j], thickness=-1)
            else:
                cv.circle(canvas, (int(x), int(y)), 4, colors[j], thickness=-1)

    to_plot = cv.addWeighted(oriImg, 0.3, canvas, 0.7, 0)
    plt.imshow(to_plot[:,:,[2,1,0]])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(size_1, size_2)
    plt.show()

def visualise_connections_res(img_name, poses, flip=True):
# visualize: code taken from https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/testing/python/demo.ipynb
    stickwidth = 4
    oriImg = cv.imread(img_name)
    canvas = cv.imread(img_name) # B,G,R order
    for j in range(poses[0].shape[0]-1):
        rgba = np.array(cmap(1 - j/18. - 1./36))
        rgba[0:3] *= 255
        for k in range(len(poses)):
            if np.isnan(poses[k][j]).any() or poses[k][j][0] > oriImg.shape[1] or poses[k][j][0]>oriImg.shape[1]:
                continue
            x, y = poses[k][j][0:2]
            if flip:
                cv.circle(canvas, (int(y),int(x)), 4, colors[j], thickness=-1)
            else:
                cv.circle(canvas, (int(x), int(y)), 4, colors[j], thickness=-1)
    for l in range(len(limbSeq)-2):
        #l = 0
        k = 0
        index = np.array(limbSeq[l])
        if -1 in index:
            continue
        cur_canvas = canvas.copy()
        J = poses[k][index.astype(int)]
        if flip:
            X = J[:,1]
            Y = J[:,0]
        else:
            X = J[:, 0]
            Y = J[:, 1]
        if np.isnan(Y).any() or np.isnan(X).any():
            continue
        mX = np.mean(X) # the mean of the two ends of the limb (the mean of the two joints)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
        polygon = cv.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv.fillConvexPoly(cur_canvas, polygon, colors[l])
        canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        plt.imshow(canvas[:, :, [2, 1, 0]])
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(size_1, size_2)
        plt.show()


stickwidth = 4
def visualise_limbs_from_heatmaps(img_name, heatmaps, flip=True):
    pass # probably the same as visualise connections

general_threshold = 0.1
def visualise_joints_from_heatmaps(img_name, heatmaps, flip=True):

    oriImg = cv.imread(img_name)
    heatmaps = hf.resize_heatmap(oriImg, heatmaps, is_cuda=True)

    num_vis = min(5, len(heatmaps))
    for k in range(num_vis) : #len(heatmaps)): #len(heatmaps)):  # len(poses)):
        oriImg = cv.imread(img_name)
        canvas = cv.imread(img_name)  # B,G,R order
        for j in range(len(heatmaps[0])-1):

            current_heatmap = heatmaps[k][j]
            J = 18
            #current_heatmap = current_heatmap.detach().cpu().numpy()
            max_thresh = np.max(current_heatmap.flatten())
            #multiple conditions for np where

            factor = 1.001
            candidates = np.where( (current_heatmap >= max_thresh / factor) & (current_heatmap > 0.05) )
            #candidates = np.where((current_heatmap >= max_thresh / factor) & (current_heatmap > general_threshold) | (current_heatmap >= max_thresh / 1.0001))
            for ind in range(len(candidates[0])):
                cand = candidates[0][ind], candidates[1][ind]
                cand = np.asarray(cand)
                rgba = np.array(cmap(1 - j/18. - 1./36))
                rgba[0:3] *= 255

                x, y = cand
                if flip:
                    cv.circle(canvas, (int(y),int(x)), 4, colors[j], thickness=-1)
                else:
                    cv.circle(canvas, (int(x), int(y)), 4, colors[j], thickness=-1)

        to_plot = cv.addWeighted(oriImg, 0.3, canvas, 0.7, 0)
        plt.imshow(to_plot[:,:,[2,1,0]])
        plt.title('wo_supp')

        #plt.savefig('/media/pose_estimation/constrained_results/' +str(k) +'_' + img_name.split('/')[-1])
        plt.show()
        plt.waitforbuttonpress(0)  # this will wait for indefinite time
        plt.close()


def visualise_joints_from_heatmaps_with_suppression(img_name, heatmaps, flip=True):

    oriImg = cv.imread(img_name)
    heatmaps = hf.resize_heatmap(oriImg, heatmaps, is_cuda=True)
    nu_joints = heatmaps[0].shape[0]-1 # exclude the background heatmap
    nu_persons = len(heatmaps)
    total_res = np.zeros((nu_persons, heatmaps[0].shape[1], heatmaps[0].shape[2]))
    heatmaps = np.asarray(heatmaps)
    for j in range(nu_joints):
        res_after_np1 = heatmaps[:,j,:,:]

        max_val = np.max(res_after_np1)/1.001
        res_after_np1[res_after_np1 < max_val] = 0
        res_after_np1[res_after_np1 < general_threshold] = 0
        max_val, max_ind = np.max(res_after_np1,axis=0), np.argmax(res_after_np1,axis=0)
        for k in range(nu_persons):  # iterating over the persons
            k_max_ind = np.where(max_ind==k) # the joints where k is the maximum index
            total_res[k][k_max_ind] = total_res[k][k_max_ind] + max_val[k_max_ind]


    num_vis = min(5, len(heatmaps))
    for k in range(num_vis) : #len(heatmaps)): #len(heatmaps)):  # len(poses)):
        oriImg = cv.imread(img_name)
        canvas = cv.imread(img_name)  # B,G,R order
        for j in range(nu_joints):
            current_heatmap = total_res[k]
            candidates = np.where(current_heatmap >0)
            candidates = np.asarray(candidates)
            for ind in range(len(candidates[0])):
                x, y = candidates[0:2, ind]
                rgba = np.array(cmap(1 - j/18. - 1./36))
                rgba[0:3] *= 255

                if flip:
                    cv.circle(canvas, (int(y),int(x)), 4, colors[4], thickness=-1)
                else:
                    cv.circle(canvas, (int(x), int(y)), 4, colors[4], thickness=-1)

        to_plot = cv.addWeighted(oriImg, 0.3, canvas, 0.7, 0)
        plt.imshow(to_plot[:,:,[2,1,0]])
        plt.title('supp')
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(size_1, size_2)
        plt.show()




def visualise_connections_gt(img_name, poses, flip=True):
# visualize: code taken from https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/testing/python/demo.ipynb
    stickwidth = 4
    oriImg = cv.imread(img_name)
    canvas = cv.imread(img_name) # B,G,R order
    for j in range(poses[0].shape[0]-1):
        rgba = np.array(cmap(1 - j/18. - 1./36))
        rgba[0:3] *= 255
        for k in range(len(poses)):
            if np.isnan(poses[k][j]).any() or poses[k][j][0] > oriImg.shape[1] or poses[k][j][0]>oriImg.shape[1]:
                continue
            x, y = poses[k][j][0:2]
            if flip:
                cv.circle(canvas, (int(y),int(x)), 4, colors[j], thickness=-1)
            else:
                cv.circle(canvas, (int(x), int(y)), 4, colors[j], thickness=-1)
    for l in range(len(limbSeq)-2):
        #l = 0
        k = 0
        index = np.array(limbSeq[l])
        if -1 in index:
            continue
        cur_canvas = canvas.copy()
        J = poses[k][index.astype(int)]
        X = J[:,0]
        Y = J[:,1]
        if np.isnan(Y).any() or np.isnan(X).any():
            continue
        mX = np.mean(X) # the mean of the two ends of the limb (the mean of the two joints)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
        polygon = cv.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv.fillConvexPoly(cur_canvas, polygon, colors[l])
        canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        plt.imshow(canvas[:, :, [2, 1, 0]])
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(size_1, size_2)
        plt.show()




def visualise_joints_resized(img_name, poses):
# visualize

    oriImg = cv.imread(img_name)
    canvas = cv.imread(img_name) # B,G,R order
    #oriImg = cv.resize(oriImg, (46, 46)) #,  interpolation=cv.INTER_CUBIC
    canvas = cv.resize(canvas, (46, 46))
    for j in range(poses[0].shape[0]-1):
        rgba = np.array(cmap(1 - j/18. - 1./36))
        rgba[0:3] *= 255
        for k in range(len(poses)):
            if np.isnan(poses[k][j]).any() or poses[k][j][0] > oriImg.shape[1] or poses[k][j][0]>oriImg.shape[1]:
                continue
            x, y = poses[k][j][0:2]
            cv.circle(canvas, (int(x),int(y)), 4, colors[j], thickness=-1)

    to_plot = cv.addWeighted(oriImg, 0.3, canvas, 0.7, 0)
    plt.imshow(to_plot[:,:,[2,1,0]])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(size_1, size_2)
    plt.show()
