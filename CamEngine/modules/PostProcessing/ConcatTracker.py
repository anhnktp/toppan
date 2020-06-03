import torch

class ConcatTracker(object):

    def __init__(self, track_id, img_obj, event_name=None, vectorizer=None):
        self.start_time = img_obj['timestamp']
        self.end_time = img_obj['timestamp']
        self.track_id = track_id
        self._events = {}
        self.concatenated_tracks = []
        self._embeddings = []
        self.images = []
        if event_name and (event_name not in self._events):
            self._events[event_name] = img_obj['timestamp']
        if vectorizer:
            self.images.append(img_obj['data'])
            # self._embeddings.append(vectorizer.predict(img_obj['data']))

    def add_image(self, img_obj, event_name=None, vectorizer=None):
        self.end_time = img_obj['timestamp']
        if event_name and (event_name not in self._events):
            self._events[event_name] = img_obj['timestamp']
        if vectorizer:
            # self._embeddings.append(vectorizer.predict(img_obj['data']))
            self.images.append(img_obj['data'])

    def is_enter_track(self):
        return ('ENTER' in self._events) and ('EXIT' not in self._events)

    def is_tail_track(self):
        return (len(self._events) == 0) or ('EXIT' in self._events)

    def is_exit_track(self):
        return ('ENTER' not in self._events) and ('EXIT' in self._events)

    def is_completed_track(self):
        return ('ENTER' in self._events) and ('EXIT' in self._events)

    def get_events(self):
        return self._events

    def get_images(self, maxsize=-1):
        if maxsize > 0:
            return self.images[0:maxsize]
        return self.images

    def get_embeddings(self, maxsize=-1):
        if maxsize > 0:
            return torch.cat(self._embeddings, 0)[0:maxsize]
        return torch.cat(self._embeddings)

    def concat_track(self, track):
        self.start_time = min(self.start_time, track.start_time)
        self.end_time = max(self.end_time, track.end_time)
        self._events.update(track._events)
        self._embeddings.extend(track._embeddings)
        self.concatenated_tracks.append(track.track_id)

    def extract_features(self, vectorizer=None):
        if vectorizer:
            print('Starting extract features of {} images of track id {}...'.format(len(self.images), self.track_id))
            chunks = [self.images[i:i+300] for i in range(0, len(self.images), 300)]
            self.images = []
            for chunk in chunks:
                embeddings = vectorizer.predict(chunk)
                self._embeddings.append(embeddings)