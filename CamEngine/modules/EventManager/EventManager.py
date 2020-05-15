import requests
from threading import Timer
from helpers.settings import *
from modules.ResultManager import ResultManager

class EventManager():
    def __init__(self):
        self.result_manager = ResultManager(os.getenv('DB_URL'), os.getenv('DB_NAME'))

    def save_person(self, person):
        self.result_manager.insert('persons', person)

    def save_event(self, event_data):
        self.result_manager.insert('events', event_data)

    def find_event(self, query):
        return list(self.result_manager.filter_events_data(query))

    def update_person(self, query, person):
        person = {"$set": person}
        self.result_manager.update_one('persons', person, query)

    def update_event(self, query, event):
        event = {"$set": event}
        self.result_manager.update_one('events', event, query)

    def find_latest_local_id(self):
        return self.result_manager.get_latest_local_id()

    def send_to_bpo(self):
        Timer(interval=int(os.getenv('INTERVAL_TIME')), function=self.send_to_bpo).start()
        data = self.result_manager.get_datas()
        if data is not None:
            send_data = {'event_json': data}
            headers = {'User-Agent': 'Mozilla/5.0'}
            api_url = os.getenv('API_URL')
            response = requests.post(api_url, auth=('mujin','2019'), verify=False, headers=headers, data=send_data)
            bpo_logger.info(response.json())
            if response.json()['status'] == 'true':
                bpo_logger.info('Sent to BPO successfully')
            else:
                bpo_logger.error('Failed to send BPO')
