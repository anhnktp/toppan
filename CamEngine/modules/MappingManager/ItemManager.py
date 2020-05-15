"""
Created by AnhVN at 12/2/20
"""
import pickle
import os
from multiprocessing import Process
from modules.RedisQueue import RedisQueue
from modules.EventManager import EventManager

class ItemManager(Process):

    def __init__(self, host='localhost', port=6379, ITEM_OFFSET_TIME=2):
        Process.__init__(self, daemon=True)
        self._queue = RedisQueue(name="item_event", max_size=int(os.getenv("REDIS_MAX_QUEUE_SIZE")), host=host, port=port)
        self._event = EventManager()
        self._dif_time = ITEM_OFFSET_TIME

    def to_redis(self, item_event, cam_type):
        value = {
            'item_event': item_event,
            'cam_type': cam_type
        }
        pickled_object = pickle.dumps(value)
        self._queue.put(pickled_object)

    def run(self):
        while True:
            if not self._queue.empty():
                data = pickle.loads(self._queue.get_front())
                self._event.save_event(data['item_event'])
