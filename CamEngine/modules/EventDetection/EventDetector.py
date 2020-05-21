from helpers.time_utils import convert_timestamp_to_human_time

class EventDetector(object):

    def __init__(self, BASKET_FREQ=20):
        super(EventDetector, self).__init__()
        self.basket_freq = BASKET_FREQ

    def update(self, trackers, localIDs_end, csv_writer):
        self.detect_area(trackers, csv_writer)
        self.detect_has_basket_ppl(localIDs_end, csv_writer)

    def detect_area(self, trackers, csv_writer):
        self.localIDs_entered = []
        self.localIDs_exited = []
        self.localIDs_A = []
        self.localIDs_B = []

        for d in trackers:
            if d[-1] < 0: continue
            # Check bit area
            bit_area = int(d[-5])
            human_time = convert_timestamp_to_human_time(d[-4])
            if bit_area == 1:
                self.localIDs_entered.append(int(d[-1]))
                csv_writer.write((1, int(d[-1]), 1200, 'ENTRANCE', int(d[-4]), human_time))
            if bit_area == 2:
                self.localIDs_exited.append(int(d[-1]))
                csv_writer.write((1, int(d[-1]), 1200, 'EXIT', int(d[-4]), human_time))
            if bit_area == 3:
                self.localIDs_A.append(int(d[-1]))
                csv_writer.write((1, int(d[-1]), 1200, 'A', int(d[-4]), human_time))
            if bit_area == 4:
                self.localIDs_B.append(int(d[-1]))
                csv_writer.write((1, int(d[-1]), 1200, 'B', int(d[-4]), human_time))

    def detect_has_basket_ppl(self, localIDs_end, csv_writer):
        for local_id, basket_cnt, basket_time, num_ppl, timestamp in localIDs_end:
            if local_id < 0: continue
            if basket_cnt > self.basket_freq:
                csv_writer.write((1, local_id, 1202, 'HAS BASKET', basket_time, convert_timestamp_to_human_time(basket_time)))
            else:
                csv_writer.write((1, local_id, 1202, 'NO BASKET', basket_time, convert_timestamp_to_human_time(basket_time)))
            csv_writer.write(
                (1, local_id, 1203, 'GROUP {} PEOPLE'.format(num_ppl + 1), timestamp, convert_timestamp_to_human_time(timestamp)))

