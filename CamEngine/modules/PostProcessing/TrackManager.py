from .ConcatTracker import ConcatTracker
from .Matching import Matching

class TrackManager(object):

    def __init__(self, vectorizer=None):
        self.tracks = {}
        self._vectorizer = vectorizer

    def update(self, img, trackers):
        for trk in trackers:
            if trk[-1] < 1: continue
            track_id = int(trk[-1])
            img_obj = {'data': img[int(trk[1]):int(trk[3]), int(trk[0]):int(trk[2])], 'timestamp': trk[-4]}
            if int(trk[-5]) == 1: event_name = 'ENTER'
            elif int(trk[-5]) == 2: event_name = 'EXIT'
            else: event_name = None
            track = self.tracks.get(track_id, None)
            if track is None:
                self.tracks[track_id] = ConcatTracker(track_id, img_obj, event_name=event_name, vectorizer=self._vectorizer)
            else:
                track.add_image(img_obj, event_name=event_name, vectorizer=self._vectorizer)

    def post_processing(self):
        # Do no process non-fragmented tracks
        self.remove_completed_track()

        # Extract features
        if self._vectorizer:
            for track_id, track in self.tracks.items():
                track.extract_features(self._vectorizer)

        # Start matching
        if self._vectorizer:
            # Matching by vectorization
            # enter_tracks = self.get_enter_tracks()
            # tail_tracks = sorted(self.get_tail_tracks().values(), key=lambda t: t.start_time)
            # # tail_tracks = sorted(track_manager.get_enter_tracks().values(), key=lambda t: t.start_time)
            # print('Enter tracks: {}'.format(enter_tracks.keys()))
            # unmatched_tracks = []
            #
            # for track in tail_tracks:
            #     track_id = track.track_id
            #     match_id = Matching.match(track, enter_tracks, interruption_threshold=10)
            #     if match_id:
            #         match_track = enter_tracks[match_id]
            #         match_track.concat_track(track)
            #     else:
            #         unmatched_tracks.append(track_id)
            # print('----- Matched tracks -----')
            # for track in enter_tracks.values():
            #     print('The track {} is concatenated with: {}'.format(track.track_id, ', '.join(
            #         map(str, track.concatenated_tracks))))
            # print('----- Unmatched tracks -----')
            # print(','.join(map(str, unmatched_tracks)))
            tracks = sorted(self.get_tracks().values(), key=lambda t: t.start_time)
            matched_tracks = Matching.match_by_visual(tracks, interruption_threshold=10, dist_metric='cosine')
            print('----- Matching results -----')
            for track_id, concated_tracks in matched_tracks.items():
                print(
                    'The track {} is concatenated with: {}'.format(track_id, ', '.join(map(str, concated_tracks))))
        else:
            # Matching by timestamp
            tracks = sorted(self.get_tracks().values(), key=lambda t: t.start_time)
            matched_tracks = Matching.match_by_timestamp(tracks, interruption_threshold=10)
            print('----- Matching results -----')
            for track_id, concated_tracks in matched_tracks.items():
                print(
                    'The track {} is concatenated with: {}'.format(track_id, ', '.join(map(str, concated_tracks))))

    def get_tracks(self):
        return self.tracks

    def remove_track(self, track_id):
        del self.tracks[track_id]

    def remove_completed_track(self):
        to_del = []
        for track_id, track in self.tracks.items():
            if track.is_completed_track(): to_del.append(track_id)
        for t in to_del:
            del self.tracks[t]

    def get_enter_tracks(self):
        enter_tracks = {}
        for track_id, track in self.tracks.items():
            if track.is_enter_track(): enter_tracks[track_id] = track
        return enter_tracks

    def get_tail_tracks(self):
        tail_tracks = {}
        for track_id, track in self.tracks.items():
            if track.is_tail_track(): tail_tracks[track_id] = track
        return tail_tracks

    def get_exit_tracks(self):
        exit_tracks = {}
        for track_id, track in self.tracks.items():
            if track.is_exit_track(): exit_tracks[track_id] = track
        return exit_tracks

    def get_completed_tracks(self):
        completed_tracks = {}
        for track_id, track in self.tracks.items():
            if track.is_completed_track(): completed_tracks[track_id] = track
        return completed_tracks