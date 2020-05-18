"""
Created by sonnt at 1/2/20
"""
import pickle
import os
from .RedisQueue import RedisQueue

class SyncManager():

    def __init__(self, host='localhost', port=6379):
        self._queue = RedisQueue(name="local_id", max_size=int(os.getenv("REDIS_MAX_QUEUE_SIZE")), host=host, port=port)

    def to_redis(self, trackers, timestamp):
        value = {
            'trackers': trackers,
            'timestamp': timestamp
        }
        pickled_object = pickle.dumps(value)
        self._queue.put(pickled_object)

    def from_redis(self, timestamp):
        list_trackers = [pickle.loads(v) for v in self._queue.get() if v is not '\x00']
        for idx in range(len(list_trackers) - 1, -1, -1):
            if list_trackers[idx]['timestamp'] < timestamp:
                if idx == 0:
                    return list_trackers[idx]['trackers']
                if (timestamp - list_trackers[idx]['timestamp']) < (list_trackers[idx - 1]['timestamp'] - timestamp):
                    return list_trackers[idx]['trackers']
                else:
                    return list_trackers[idx - 1]['trackers']
        if len(list_trackers):
            return list_trackers[-1]['trackers']
        return None