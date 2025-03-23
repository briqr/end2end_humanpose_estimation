
from enum import IntEnum


class CocoJoints(IntEnum):
    NOSE = 0
    L_EYE = 1
    R_EYE = 2
    L_EAR = 3
    R_EAR = 4
    L_SHOULDER = 5
    R_SHOULDER = 6
    L_ELBOW = 7
    R_ELBOW = 8
    L_WRIST = 9
    R_WRIST = 10
    L_HIP = 11
    R_HIP = 12
    L_KNEE = 13
    R_KNEE = 14
    L_ANKLE = 15
    R_ANKLE = 16
    NECK = 17
    BACKGROUND=18


#
limbSeq = [[17, 6], [17, 5], [6, 8], [8, 10], [5, 7], [7, 9], [17, 12], [12, 14], \
           [14, 16], [17, 11], [11, 13], [13, 15], [17, 0], [0, 1], [1, 3], \
           [0, 2], [2, 4], [6, 3], [5, 4]]
#the limb seq using enum
limbSeqConnections = [ [CocoJoints.NECK, CocoJoints.R_SHOULDER ], [CocoJoints.NECK, CocoJoints.L_SHOULDER ] ]



# the joints that are related by symmetry
symmetric_joints = [[CocoJoints.L_WRIST, CocoJoints.L_ELBOW, CocoJoints.R_WRIST, CocoJoints.R_ELBOW],
                    [CocoJoints.L_ELBOW, CocoJoints.L_SHOULDER, CocoJoints.R_ELBOW, CocoJoints.R_SHOULDER],
                    [CocoJoints.NOSE, CocoJoints.L_EAR, CocoJoints.NOSE, CocoJoints.R_EAR],
                    [CocoJoints.L_HIP, CocoJoints.L_KNEE, CocoJoints.R_HIP, CocoJoints.R_KNEE],
                    [CocoJoints.L_KNEE, CocoJoints.L_ANKLE, CocoJoints.R_KNEE, CocoJoints.R_ANKLE],
                    [CocoJoints.L_EYE, CocoJoints.NOSE, CocoJoints.R_EYE, CocoJoints.NOSE]]

    # The limbs joints connections
    # Coco_Connections= [ [NECK,R_SHOULDER], [NECK, L_SHOULDER], [R_SHOULDER,R_ELBOW],\
    #                 [R_ELBOW,R_WRIST], [L_SHOULDER, L_ELBOW], [L_ELBOW,L_WRIST],\
    #                 [NECK,R_HIP], [R_HIP, R_KNEE], [R_KNEE,R_ANKLE],\
    #                [NECK,L_HIP],[L_HIP,L_KNEE],[L_KNEE,L_ANKLE],\
    #                [NECK, NOSE], [NOSE,L_EYE], [L_EYE, L_EAR],\
    #                [NOSE, R_EYE], [R_EYE, R_EAR], [R_SHOULDER,L_EAR], \
    #               [L_SHOULDER, R_EAR] ]

