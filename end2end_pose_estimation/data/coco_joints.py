#not sure, I either keep this class or dump its contents into Global_Constants
from enum import Enum

# the joint ordering matching the cnn output
class CocoJoints(Enum):
    NOSE=0
    L_EYE=1
    R_EYE=2
    L_EAR=3
    R_EAR=4
    L_SHOULDER=5
    R_SHOULDER=6
    L_ELBOW=7
    R_ELBOW=8
    L_WRIST=9
    R_WRIST=10
    L_HIP=11
    R_HIP=12
    L_KNEE=13
    R_KNEE=14
    L_ANKLE=15
    R_ANKLE=16
    NECK=17
    BACKGROUND=18

    # The limbs joints connections
#Coco_Connections= [ [NECK,R_SHOULDER], [NECK, L_SHOULDER], [R_SHOULDER,R_ELBOW],\
#                 [R_ELBOW,R_WRIST], [L_SHOULDER, L_ELBOW], [L_ELBOW,L_WRIST],\
#                 [NECK,R_HIP], [R_HIP, R_KNEE], [R_KNEE,R_ANKLE],\
#                [NECK,L_HIP],[L_HIP,L_KNEE],[L_KNEE,L_ANKLE],\
#                [NECK, NOSE], [NOSE,L_EYE], [L_EYE, L_EAR],\
#                [NOSE, R_EYE], [R_EYE, R_EAR], [R_SHOULDER,L_EAR], \
#               [L_SHOULDER, R_EAR] ]

