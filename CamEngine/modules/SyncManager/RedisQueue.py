"""
Created by sonnt at 1/9/20
"""
import redis


class RedisQueue(object):
    """Simple Queue with Redis Backend"""

    def __init__(self, name, namespace='queue', max_size=10, **redis_kwargs):
        """The default connection parameters are: host='localhost', port=6379, db=0"""

        self.__db = redis.Redis(**redis_kwargs)
        self.__max_size = max_size
        self.key = '%s:%s' % (namespace, name)

    def qsize(self):
        """Return the approximate size of the queue."""
        return self.__db.llen(self.key)

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        return self.qsize() == 0

    def put(self, item):
        """Put item into the queue."""
        self.__db.rpush(self.key, item)
        if self.qsize() > self.__max_size:
            self.__db.lpop(self.key)

    def get(self):
        return self.__db.lrange(self.key, 0, self.__max_size - 1)

    def get_front(self):
        return self.__db.lpop(self.key)

    def close(self):
        self.__db.srem(self.key)