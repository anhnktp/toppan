from helpers.time_utils import convert_timestamp_to_human_time

class EventDetector(object):

    def __init__(self, BASKET_FREQ=20):
        super(EventDetector, self).__init__()
        self.basket_freq = BASKET_FREQ

    def update_fish_eye(self, trackers, localIDs_end, csv_writer):
        self.detect_area(trackers, csv_writer)
        self.detect_has_basket(localIDs_end, csv_writer)
        self.detect_in_signages(localIDs_end, csv_writer)

    def update_signage(self, localIDs_end, csv_writer):
        """ 
            Faces has structure: [[xmin, ymin, xmax, ymax, person_id]]
        """
        self.detect_accompany_number(localIDs_end, csv_writer)
        self.detect_attention(localIDs_end, csv_writer)

    def detect_area(self, trackers, csv_writer):
        self.localIDs_entered = []
        self.localIDs_exited = []
        self.localIDs_A = []
        self.localIDs_B = []

        for d in trackers:
            if d[-1] < 1: continue
            # Check bit area
            bit_area = int(d[-5])
            human_time = convert_timestamp_to_human_time(d[-4])
            if bit_area == 1:
                self.localIDs_entered.append(int(d[-1]))
                csv_writer.write((1, int(d[-1]), 1200, 'ENTRANCE', d[-4], human_time))
            if bit_area == 2:
                self.localIDs_exited.append(int(d[-1]))
                csv_writer.write((1, int(d[-1]), 1200, 'EXIT', d[-4], human_time))
            if bit_area == 3:
                self.localIDs_A.append(int(d[-1]))
                csv_writer.write((1, int(d[-1]), 1200, 'A', d[-4], human_time))
            if bit_area == 4:
                self.localIDs_B.append(int(d[-1]))
                csv_writer.write((1, int(d[-1]), 1200, 'B', d[-4], human_time))

    def detect_has_basket(self, localIDs_end, csv_writer):
        for local_id, basket_cnt, basket_time, _, _, _, _, timestamp in localIDs_end:
            if local_id < 0: continue
            if basket_cnt > self.basket_freq:
                csv_writer.write(
                    (1, local_id, 1202, 'HAS BASKET', basket_time, convert_timestamp_to_human_time(basket_time)))
            else:
                csv_writer.write(
                    (1, local_id, 1202, 'NO BASKET', timestamp, convert_timestamp_to_human_time(timestamp)))

    def detect_attention(self, localIDs_end, csv_writer):
        """
            Write the attention results to csv file
        """
        for items in localIDs_end:
            local_id = items[0]
            if local_id < 1: continue
            is_attention = items[4]
            start_attention = items[5]
            end_attention = items[8]
            duration = items[6]

            if is_attention == 'has_attention':
                for start, end, duration in zip(start_attention, end_attention, duration):
                    csv_writer.write((1, local_id, 1557, '{}'.format(is_attention), convert_timestamp_to_human_time(start),
                                                            convert_timestamp_to_human_time(end), duration, None, None, None, None))

    def detect_in_signages(self, localIDs_end, csv_writer):
        for items in localIDs_end:
            local_id = items[0]
            if local_id < 0: continue
            sig1_start_time = items[3]
            sig1_end_time = items[4]
            sig2_start_time = items[5]
            sig2_end_time = items[6]

            if sig1_start_time and sig1_end_time:
                csv_writer.write((1, local_id, 1200, 'ENTER SIGNAGE 1', sig1_start_time, convert_timestamp_to_human_time(sig1_start_time)))
                csv_writer.write((1, local_id, 1200, 'LEAVE SIGNAGE 1', sig1_end_time, convert_timestamp_to_human_time(sig1_end_time)))

            if sig2_start_time and sig2_end_time:
                csv_writer.write((1, local_id, 1200, 'ENTER SIGNAGE 2', sig2_start_time, convert_timestamp_to_human_time(sig2_start_time)))
                csv_writer.write((1, local_id, 1200, 'LEAVE SIGNAGE 2', sig2_end_time, convert_timestamp_to_human_time(sig2_end_time)))

    def detect_accompany_number(self, localIDs_end, csv_writer):
        for items in localIDs_end:
            local_id = items[0]
            if local_id < 0: continue
            num_ppl = items[1]
            start_time = items[2]
            end_time = items[3]
            duration_group = items[7]
            csv_writer.write((1, local_id, 1203, 'GROUP {} PEOPLE'.format(num_ppl + 1),
                                convert_timestamp_to_human_time(start_time), convert_timestamp_to_human_time(end_time), duration_group,
                                items[9], items[10], items[11], items[12]))