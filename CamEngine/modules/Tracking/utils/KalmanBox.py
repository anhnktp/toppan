from filterpy.kalman import KalmanFilter
from helpers.bbox_utils import *


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0
    def __init__(self, bbox, timestamp):
        """
        Initialises a tracker using initial bounding box.
        timestamp is start time of track
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        # when initialize track, default local_id = -1 & when track has hit_streak >= min_hits, local_id is allocated
        self.id = -1
        self.basket_count = 0
        self.basket_time = timestamp        # First timestamp has basket
        self.ppl_dist = {}
        self.timestamp = timestamp
        self.history = []
        # self.hits = 0
        self.hit_streak = 0
        self.area = None
        # self.age = 0

    def update(self, bbox, min_hits):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        # self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        # new local_id then return True, otherwise return False
        if (self.id == -1) and (self.hit_streak > min_hits):    # ENTER
            KalmanBoxTracker.count += 1
            self.id = KalmanBoxTracker.count
            return True
        return False

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        # self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    def get_last_state(self, max_age):
        """
        Returns the last-maxage bounding box estimate before track is removed
        """
        return self.history[-max_age]