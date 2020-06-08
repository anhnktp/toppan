from helpers.time_utils import convert_timestamp_to_human_time

class EventDetector(object):

    def __init__(self, BASKET_FREQ=20):
        super(EventDetector, self).__init__()
        self.basket_freq = BASKET_FREQ

    def update(self, trackers, localIDs_end, csv_writer):
        self.detect_area(trackers, csv_writer)
        self.detect_has_basket_ppl(localIDs_end, csv_writer)

    def update_signage(self, trackers, faces, localIDs_end, csv_writer):
        """ 
            Faces has structure: [[xmin, ymin, xmax, ymax, person_id]]
        """
        self.detect_accompany_number(localIDs_end, csv_writer)
        self.write_attention(localIDs_end,csv_writer)

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

    def detect_has_basket_ppl(self, localIDs_end, csv_writer):
        for local_id, basket_cnt, basket_time, num_ppl, timestamp in localIDs_end:
            if local_id < 0: continue
            if basket_cnt > self.basket_freq:
                csv_writer.write((1, local_id, 1202, 'HAS BASKET', basket_time, convert_timestamp_to_human_time(basket_time)))
            else:
                csv_writer.write((1, local_id, 1202, 'NO BASKET', timestamp, convert_timestamp_to_human_time(timestamp)))
            csv_writer.write(
                (1, local_id, 1203, 'GROUP {} PEOPLE'.format(num_ppl + 1), timestamp, convert_timestamp_to_human_time(timestamp)))

    def write_attention(self,localIDs_end,csv_writer):
        """
            Write the attention results to csv file
        """
        for items in localIDs_end:
            local_id = items[0]
            is_attention = items[4]
            start_attention = items[5]
            end_attention = items[8]
            duration = items[6]

            if is_attention == 'has_attention':
                csv_writer.write((1, local_id, 1557, '{}'.format(is_attention), convert_timestamp_to_human_time(start_attention),
                                                        convert_timestamp_to_human_time(end_attention), duration))

    def detect_accompany_number(self, localIDs_end, csv_writer):
        for items in localIDs_end:
            local_id = items[0]
            num_ppl = items[1]
            start_time = items[2]        
            end_time = items[3]
            duration_group = items[7]        
            csv_writer.write((1, local_id, 1203, 'GROUP {} PEOPLE'.format(num_ppl + 1), 
                                convert_timestamp_to_human_time(start_time), convert_timestamp_to_human_time(end_time), duration_group))


