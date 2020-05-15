class DataTemplate:

    # define document for persons collection
    @staticmethod
    def person_document(local_id, cropped_img_url, start_time, exit_time):
        """
        :param local_id: local_id of person
        :param image_url: url of cropped image
        :param enter_time: start timestamp of local_id
        :param exit_time:  end timestamp of local_id
        :return:
        """
        return {
            'local_id': local_id,
            'image_url': cropped_img_url,
            'enter_time': start_time,
            'exit_time': exit_time,
        }

    # --------------- Event definition -------------------- #

    @staticmethod
    def enter_event(local_id, timestamp, video_url_evidence, video_url_engine):
        return {
            'local_id': local_id,
            'timestamp': timestamp,
            'event_type': 'ENTER',
            'video_url_evidence': video_url_evidence,
            'video_url_engine': video_url_engine,
            'is_sent': False
        }

    @staticmethod
    def exit_event(local_id, timestamp, video_url_evidence, video_url_engine):
        return {
            'local_id': local_id,
            'timestamp': timestamp,
            'event_type': 'EXIT',
            'video_url_evidence': video_url_evidence,
            'video_url_engine': video_url_engine,
            'is_sent': False
        }

    @staticmethod
    def overwrap_event(local_id_pairs, timestamp, video_url_evidence, video_url_engine):
        return {
            'local_id': local_id_pairs,
            'timestamp': timestamp,
            'event_type': 'OVERWRAP',
            'video_url_evidence': video_url_evidence,
            'video_url_engine': video_url_engine,
            'is_sent': False
        }

    @staticmethod
    def item_event(local_ids, timestamp, video_url_evidence, video_url_engine):
        return {
            'local_id': local_ids,
            'timestamp': timestamp,
            'event_type': 'ITEM',
            'video_url_evidence': video_url_evidence,
            'video_url_engine': video_url_engine,
            'is_sent': False
        }

    @staticmethod
    def sacker_event(local_ids, timestamp, items, video_url_evidence, video_url_engine):
        return {
            'local_id': local_ids,
            'timestamp': timestamp,
            'event_type': 'SACKER',
            'items': items,
            'video_url_evidence': video_url_evidence,
            'video_url_engine': video_url_engine,
            'is_sent': False
        }

    @staticmethod
    def send_data(persons=None, events=None):
        return {
            'persons': persons,
            'events': events
        }

    @staticmethod
    def video_url(cam_id, cam_type):
        return {
            'cam_id': cam_id,
            'cam_type': cam_type,
            'video_url': None
        }