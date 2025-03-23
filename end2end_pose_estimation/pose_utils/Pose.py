class Pose(object):
    """
    This class represents a single pose in an image.
    """
    def __init__(self, joints=None, score=None, feature=None, id=None, subset_score=None):
        self.joints = joints
        self.score = score
        self.feature = feature
        self.id = id
        self.subset_score = subset_score
