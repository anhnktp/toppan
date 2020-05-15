import pymongo
import json
import time
from helpers.settings import *

class ResultManager():

    def __init__(self, db_uri, db_name):
        """
        @The constructor
        @parameters
        @   uri: uri to mongo database (ex: mongodb://localhost:27017/)
        """
        self.db_uri = db_uri
        self.db_name = db_name

        client = pymongo.MongoClient(self.db_uri)
        ResultManager.DATABASE = client[self.db_name]

    # Insert a record into our database
    def insert(self, collection, data):
        ResultManager.DATABASE[collection].insert_one(data)

    # Update a first record
    def update_one(self, collection, data, query):
        ResultManager.DATABASE[collection].update_one(query, data)

    # Update all records
    def update_many(self, collection, data, query):
        ResultManager.DATABASE[collection].update_many(query, data)

    def find(self, collection):
        return ResultManager.DATABASE[collection].find()

    def filter_events_data(self, query):
        return ResultManager.DATABASE['events'].find(query)

    def filter_persons_data(self, query):
        return ResultManager.DATABASE['persons'].find(query)

    def delete(self, collection, id):
        query = {'_id': id}
        return ResultManager.DATABASE[collection].delete_one(query)

    def get_local_ids(self, local_ids_in_event, local_ids):
        for local_id in local_ids_in_event:
            if isinstance(local_id, str):
                if local_id not in local_ids:
                    local_ids.append(local_id)
            elif isinstance(local_id, list):
                for local_id1 in local_id:
                    if local_id1 not in local_ids:
                        local_ids.append(local_id1)

    def get_datas(self):

        # For person
        persons = []
        events = []
        event_ids = []

        # Get all events data were not sent to BPO
        query_for_event = {'is_sent': False}
        eventdb = self.filter_events_data(query_for_event)
        local_ids = []
        for event in eventdb:
            # Store event_id for update sending status
            event_ids.append(event['_id'])
            # Setting data for each event
            event_data = dict()
            # self.get_element_array(doc['local_id'])
            event_data['local_ids'] = event['local_id']
            event_data['event_type'] = event['event_type']
            event_data['video_url_evidence'] = event['video_url_evidence']
            event_data['video_url_engine'] = event['video_url_engine']
            event_data['timestamp'] = event['timestamp']
            event_data['event_id'] = str(event['_id'])
            if event['event_type'] == 'SACKER': event_data['items'] = event['items']
            # Append data for return value
            events.append(event_data)

            # Get local_ids to select data for person
            self.get_local_ids(event['local_id'], local_ids)
        if len(events) == 0: return None
        # Get all persons data based on local_ids
        query_for_person = {'local_id': {'$in': local_ids}}
        persondb = self.filter_persons_data(query_for_person)

        for person in persondb:
            person_data = dict()
            person_data['local_id'] = person['local_id']
            person_data['image_url'] = person['image_url']
            person_data['enter_time'] = person['enter_time']
            person_data['exit_time'] = person['exit_time']
            # Append data for return value
            persons.append(person_data)
        data = {
            'persons': persons,
            'events': events
        }

        # Update sending status for event data
        query_for_sending_status = {'_id': {'$in': event_ids}}

        data_update = {'$set': {'is_sent': True}}

        self.update_many('events', data_update, query_for_sending_status)
        if os.getenv('SAVE_JSON') == 'TRUE':
            cur_time = int(time.time())
            json_path = join(os.getenv('CROPPED_IMAGE_FOLDER'), str(cur_time) + '.json')
            with open(json_path, 'w') as outfile:
                json.dump(data, outfile, indent=4)
        return json.dumps(data)

    def get_latest_local_id(self):
        latest_person = list(ResultManager.DATABASE['persons'].find().sort([("enter_time", -1), ("local_id", -1)]).limit(1))
        if len(latest_person) > 0:
            return list(latest_person)[0]['local_id']
        else:
            return None
