class Pose(object):
    """
    This class represents a single pose in an image.
    """
    def __init__(self, id, joints=None, score=None):
        self.category = 1
        self.id = id
        self.joints = joints
        self.score = score
