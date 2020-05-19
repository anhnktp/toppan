import requests
import cv2
import uuid
from threading import Timer
from ..ResultManager import ResultManager
from ..DataTemplate import DataTemplate
from helpers.time_utils import concat_local_id_time
from helpers.settings import *

class EventManager():
    def __init__(self, img_size):
        self.result_manager = ResultManager(os.getenv('DB_URL'), os.getenv('DB_NAME'))
        self.img_size = img_size

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
        last_local_id = self.result_manager.get_latest_local_id()
        if last_local_id is None:
            last_local_id = int(os.getenv('LAST_ID'))
        else:
            last_local_id = int(last_local_id.split('_')[0])
        return last_local_id

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

    def save_persons_toDB(self, trks_start, localIDs_end, img_ori, cur_time):

        # Insert new persons (ornew tracks) to db
        for trk_start in trks_start:
            local_id_time = concat_local_id_time(trk_start[-1], trk_start[-2])
            cropped_img_path = join(os.getenv('CROPPED_IMAGE_FOLDER'), '{}.jpg'.format(local_id_time))
            cropped_img = img_ori[int(trk_start[1]):int(trk_start[3]), int(trk_start[0]):int(trk_start[2])]
            cv2.imwrite(cropped_img_path, cropped_img)
            cropped_img_path = 'http://{}:{}/{}.jpg'.format(os.getenv('PUBLIC_IP'), os.getenv('CROPPED_IMAGE_PORT'),
                                                            local_id_time)
            person = DataTemplate.person_document(local_id_time, cropped_img_path, int(trk_start[-2]), None)
            self.save_person(person)

        # Update exit time of persons (or tracks) to db
        for local_id in localIDs_end:
            query = {'local_id': local_id}
            self.update_person(query, {'exit_time': cur_time})

    def save_events_toDB(self, vms, localIDs_entered, localIDs_exited, overwrap_cases, cur_time, engine_frames):

        # Save ENTER event to database
        cur_time_in_ms = cur_time * 1000
        if len(localIDs_entered) > 0:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            video_name = '{}.mp4'.format(str(uuid.uuid1()))
            video_path = join(os.getenv('CROPPED_IMAGE_FOLDER'), video_name)
            write_event = cv2.VideoWriter(video_path, fourcc, 5, self.img_size)
            for i in range(0, len(engine_frames)):
                write_event.write(engine_frames[i])
            write_event.release()

            # Cam entrance
            videl_url_cam_entrance = DataTemplate.video_url(os.getenv('ID_CAM_ENTRANCE'), 'CAM_ENTRANCE')
            video_url_engine = 'http://{}:{}/{}'.format(os.getenv('PUBLIC_IP'), os.getenv('CROPPED_IMAGE_PORT'),
                                                        video_name)
            video_url_evidence = vms.get_video_url([videl_url_cam_entrance], cur_time_in_ms - 5000,
                                                   cur_time_in_ms + 5000)
            data = DataTemplate.enter_event(localIDs_entered, cur_time, video_url_evidence, video_url_engine)
            self.save_event(data)

        # Save EXIT event to database
        if len(localIDs_exited) > 0:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            video_name = '{}.mp4'.format(str(uuid.uuid1()))
            video_path = join(os.getenv('CROPPED_IMAGE_FOLDER'), video_name)
            write_event = cv2.VideoWriter(video_path, fourcc, 5, self.img_size)
            for i in range(0, len(engine_frames)):
                write_event.write(engine_frames[i])
            write_event.release()

            # Cam exit
            video_url_cam_exit = DataTemplate.video_url(os.getenv('ID_CAM_EXIT'), 'CAM_EXIT')
            video_url_engine = 'http://{}:{}/{}'.format(os.getenv('PUBLIC_IP'), os.getenv('CROPPED_IMAGE_PORT'),
                                                        video_name)
            video_url_evidence = vms.get_video_url([video_url_cam_exit], cur_time_in_ms - 5000, cur_time_in_ms + 5000)
            data = DataTemplate.exit_event(localIDs_exited, cur_time, video_url_evidence, video_url_engine)
            self.save_event(data)

        # Save OVERWRAP event to database
        if len(overwrap_cases) > 0:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            video_name = '{}.mp4'.format(str(uuid.uuid1()))
            video_path = join(os.getenv('CROPPED_IMAGE_FOLDER'), video_name)
            write_event = cv2.VideoWriter(video_path, fourcc, 5, self.img_size)
            for i in range(0, len(engine_frames)):
                write_event.write(engine_frames[i])
            write_event.release()

            # Cam west
            videl_url_cam_west = DataTemplate.video_url(os.getenv('ID_CAM_WEST'), 'CAM_WEST')
            # Cam east
            videl_url_cam_east = DataTemplate.video_url(os.getenv('ID_CAM_EAST'), 'CAM_EAST')
            # Cam fish-eye
            videl_url_cam_fish_eye = DataTemplate.video_url(os.getenv('ID_CAM_360'), 'CAM_360')

            video_url_evidence = vms.get_video_url([videl_url_cam_west, videl_url_cam_east, videl_url_cam_fish_eye],
                                                   cur_time_in_ms - 5000, cur_time_in_ms + 5000)
            video_url_engine = 'http://{}:{}/{}'.format(os.getenv('PUBLIC_IP'), os.getenv('CROPPED_IMAGE_PORT'),
                                                        video_name)
            for overwrap_case in overwrap_cases:
                query = {'event_type': 'OVERWRAP', 'local_id': {'$in': [[overwrap_case], [overwrap_case[::-1]]]},
                         'timestamp': {'$gt': cur_time - int(os.getenv('OVERWRAP_OFFSET_TIME'))}}
                same_event = self.find_event(query)
                if len(same_event) > 0: continue
                data = DataTemplate.overwrap_event([overwrap_case], cur_time, video_url_evidence, video_url_engine)
                self.save_event(data)

