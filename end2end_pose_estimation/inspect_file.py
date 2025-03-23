import numpy as np
import pose_utils.Pose

poses_path = '/media/pose_estimation/Experiments/Original_Coco_Setup/poses/000000248616.npy'

#poses_path = '/media/pose_estimation/Experiments/coco_precomputed_features/coco/heatmap_pred_000000313198.npy'

pose = np.load(poses_path)
for l in range(18):
    print(np.sum(pose[0].score)*100/(l+1))

print('*****')
print (pose[0].score)
print(pose[0].subset_score)
print(pose[0].joints)
a = [[1,2], [3,4]]
min_0 = np.min(a, axis=0)
min_1 = np.min(a, axis=1)

print (min_0)
print (min_1)