import os
import pandas as pd
import numpy as np


csv_signage_df = pd.read_csv('../outputs/log_signage_attention_01_update_shopperID.csv', parse_dates=[-3,-2], date_parser=lambda x: pd.datetime.strptime(x, '%Y:%m:%d %H:%M:%S.%f'))
csv_signage_df.rename(columns={'Shopper_ID': 'shopper ID'}, inplace=True)
#csv_signage_df['Start_time'] = csv_signage_df['Start_time'].values.astype(np.int64) / 10 ** 9 - 7*3600 + 1
#csv_signage_df['End_time'] = csv_signage_df['End_time'].values.astype(np.int64) / 10 ** 9 - 7*3600 + 1


print(csv_signage_df)
print(type(csv_signage_df.iloc[0,5]))
t1 = csv_signage_df.iloc[0,5]
t2 = csv_signage_df.iloc[0,6]
t3 = csv_signage_df.iloc[1,5]
t4 = csv_signage_df.iloc[1,6]


def compute_time_iou(cam_360_start_time, cam_360_end_time, cam_signage_start_time, cam_signage_end_time):

    if cam_360_end_time <= cam_signage_start_time or cam_360_start_time >= cam_signage_end_time:
        return 0
    else:
        intersection = (min(cam_360_end_time, cam_signage_end_time) - max(cam_signage_start_time, cam_360_start_time)).total_seconds()
        union = (max(cam_signage_end_time, cam_360_end_time) - min(cam_360_start_time, cam_signage_start_time)).total_seconds()
        return intersection / union
