from helpers.time_utils import convert_timestamp_to_human_time

class EventDetector(object):

    def __init__(self, BASKET_FREQ=20):
        super(EventDetector, self).__init__()
        self.basket_freq = BASKET_FREQ

    def update(self, trackers, csv_writer):
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
                if int(d[-3]) > self.basket_freq:
                    csv_writer.write((1, int(d[-1]), 1202, 'HAS BASKET', int(d[-2]), convert_timestamp_to_human_time(int(d[-2]))))
                else:
                    csv_writer.write((1, int(d[-1]), 1202, 'NO BASKET', int(d[-4]), human_time))
            if bit_area == 3:
                self.localIDs_A.append(int(d[-1]))
                csv_writer.write((1, int(d[-1]), 1200, 'A', int(d[-4]), human_time))
            if bit_area == 4:
                self.localIDs_B.append(int(d[-1]))
                csv_writer.write((1, int(d[-1]), 1200, 'B', int(d[-4]), human_time))