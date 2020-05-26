from multiprocessing import Queue, Value, Process
from helpers.cam_data import get_engine_cams
from helpers.settings import *
from modules.DataLoader import DataLoader
from process_cam_360 import process_cam_360


if __name__ == '__main__':
    engine_logger.critical('Starting Exact ReID service...')

    # Get cam config
    engine_cams = get_engine_cams()

    # Init engine
    if not os.path.exists(os.getenv('CROPPED_IMAGE_FOLDER')):
        os.mkdir(os.getenv('CROPPED_IMAGE_FOLDER'))
        os.mkdir(os.getenv('CROPPED_IMAGE_FOLDER') + 'img')

    cam_data_loaders = []
    list_processes = []
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
        p = Process(target=process_cam_360, args=(cam_data_loader.queue_frame, num_loaded_model))
        p.start()
        list_processes.append(p)

    # Start data loader process
    for cam_data_loader in cam_data_loaders:
        cam_data_loader.start()

    # Join all proccess
    for process in list_processes:
        process.join()

    # Stop engine
    engine_logger.critical('Stopped ReID service !')
