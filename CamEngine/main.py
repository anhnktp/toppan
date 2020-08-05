from multiprocessing import Queue, Value, Process
from helpers.cam_data import get_engine_cams
from helpers.settings import *
from process_cam_360 import process_cam_360
from process_cam_signage import process_cam_signage
from process_cam_shelf import process_cam_shelf


if __name__ == '__main__':
    engine_logger.critical('Starting Exact ReID service...')

    # Get cam config
    engine_cams = get_engine_cams()

    # Init engine
    if not os.path.exists(os.getenv('OUTPUT_DIR')):
        os.mkdir(os.getenv('OUTPUT_DIR'))

    # cam_data_loaders = []
    list_processes = []
    num_loaded_model = Value('i', 0)

    # Init data loader
    # for cam_type in engine_cams:
    #     cam_loader = DataLoader(engine_cams[cam_type]['RTSP_URL'],
    #                             engine_cams[cam_type]['FPS'],
    #                             cam_type,
    #                             Queue(int(os.getenv('QUEUE_SIZE'))),
    #                             num_loaded_model)
    #     cam_data_loaders.append(cam_loader)

    # Init & start engine process
    for cam_type in engine_cams:
        if cam_type == 'CAM_360':
            p = Process(target=process_cam_360, args=(Queue(int(os.getenv('QUEUE_SIZE'))), num_loaded_model))
        elif cam_type == 'CAM_SIGNAGE':
            p = Process(target=process_cam_signage, args=(Queue(int(os.getenv('QUEUE_SIZE'))), num_loaded_model))
        elif cam_type == 'CAM_SHELF':
            p = Process(target=process_cam_shelf, args=(Queue(int(os.getenv('QUEUE_SIZE'))), num_loaded_model))
        p.start()
        list_processes.append(p)

        # Run consequently 3 cam_type process
        # cam_data_loader.start()
        p.join()

    # # Start data loader process
    # # Process 3 cam_type simultaneously
    # for cam_data_loader in cam_data_loaders:
    #     cam_data_loader.start()
    #
    # # Join all proccess
    # for process in list_processes:
    #     process.join()

    # Stop engine
    engine_logger.critical('Stopped ReID service !')