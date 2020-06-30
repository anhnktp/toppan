from filterpy.kalman import KalmanFilter
from helpers.bbox_utils import *
from shapely.geometry import Point
from modules.PostProcessing.Matching import Matching

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

        self.attention = False
        self.start_hp_time = None  # start headpose timestamp
        self.end_hp_time = None  # end headpose timestamp
        self.hp_max_age = 0
        self.cnt_frame_attention = 0
        self.duration_hp_list = []
        self.start_hp_list = []
        self.end_hp_list = []

        self.sig1_start_time = None
        self.sig1_end_time = None
        self.sig2_start_time = None
        self.sig2_end_time = None

        self.sig_start_bbox = bbox[:4]
 
    def update(self, bbox, min_hits, timestamp=None, reid_area=None, track_manager=None, img=None, is_reid=False):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        # self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        # new local_id then return True, otherwise return False
        if (self.id == -1) and (self.hit_streak > min_hits):
            # Check if bbox in reid are:
            if is_reid:
                center_tup = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                center_point = Point(center_tup)
                if reid_area.contains(center_point):
                    # dead_tracks = sorted(track_manager.get_dead_tracks(interruption_threshold=10).values(), key=lambda t: t.start_time)
                    dead_tracks = track_manager.get_dead_tracks(anchor_time=timestamp, interruption_threshold=10, reid_area=reid_area).values()
                    matched_id, end_point = Matching.match_real_time(bbox, timestamp, dead_tracks)
                    if matched_id is not None:
                        img_obj = {'data': img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])], 'timestamp': timestamp,
                                   'center': np.array(center_tup)}
                        track_manager.tracks[matched_id].recover_track(img_obj, track_manager._vectorizer)
                        self.id = matched_id
                        return True, end_point
            KalmanBoxTracker.count += 1
            self.id = KalmanBoxTracker.count
            return False, None
        return False, None

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

    def get_last_state(self, max_age=1):
        """
        Returns the last-maxage bounding box estimate before track is removed
        """
        return self.history[-max_age]