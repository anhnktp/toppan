import shutil
import crython
import redis
from pymongo import MongoClient
from multiprocessing import Queue, Value, Process
from configs.cam_data import get_engine_cams, get_evidence_cams
from helpers.settings import *
from modules.DataLoader import DataLoader
from modules.SyncManager import SyncManager
from process_cam_360 import process_cam_360
# from proj3 import process_cam_360
from process_cam_shelf import process_cam_shelf
from process_cam_sacker import process_cam_sacker


@crython.job(expr='@weekly', ctx='multiprocess')
def drop_data():

    # cronjob flush database
    db_url = os.getenv('DB_URL')
    db_name = os.getenv('DB_NAME')
    client = MongoClient(db_url)
    db = client[db_name]
    db['persons'].drop()
    db['events'].drop()

    # cronjob flush cropped_image_folder & video
    shutil.rmtree(os.getenv('CROPPED_IMAGE_FOLDER'))
    os.mkdir(os.getenv('CROPPED_IMAGE_FOLDER'))

    # cronjob to reset count_id
    r = redis.Redis(os.getenv("REDIS_HOST"), int(os.getenv("REDIS_PORT")))
    r.set(name="count_id", value=0)
    r.set(name="count_item", value=0)
    del r

if __name__ == '__main__':
    engine_logger.critical('Starting Exact ReID service...')

    # Start cronjob remove database mongo daily
    crython.start()

    # Get cam config
    engine_cams = get_engine_cams()
    evidence_cams = get_evidence_cams()

    # Init
    if not os.path.exists(os.getenv('CROPPED_IMAGE_FOLDER')):
        os.mkdir(os.getenv('CROPPED_IMAGE_FOLDER'))
        os.mkdir(os.getenv('CROPPED_IMAGE_FOLDER') + 'img')

    cam_data_loaders = []
    list_processes = []
    global_tracks = SyncManager(os.getenv("REDIS_HOST"), int(os.getenv("REDIS_PORT")))
    num_loaded_model = Value('i', 0)

    # Init data loader
    for cam_type in engine_cams:
        cam_loader = DataLoader(engine_cams[cam_type]['RTSP_URL'],
                                engine_cams[cam_type]['FPS'],
                                cam_type,
                                Queue(int(os.getenv('QUEUE_SIZE'))),
                                num_loaded_model)
        cam_data_loaders.append(cam_loader)

    # Init & start engine process
    for cam_data_loader in cam_data_loaders:
        if cam_data_loader._cam_type == 'CAM_360':
            p = Process(target=process_cam_360, args=(cam_data_loader.queue_frame, num_loaded_model, global_tracks))
        elif cam_data_loader._cam_type == 'CAM_SACKER':
            p = Process(target=process_cam_sacker, args=(cam_data_loader.queue_frame, num_loaded_model, global_tracks))
        else:
            p = Process(target=process_cam_shelf, args=(cam_data_loader.queue_frame, cam_data_loader._cam_type, num_loaded_model, global_tracks))
        p.start()
        list_processes.append(p)

    # Start data loader process
    for cam_data_loader in cam_data_loaders:
        cam_data_loader.start()

    # Join all proccess
    for process in list_processes:
        process.join()

    crython.join()

    engine_logger.critical('Stopped ReID service !')
