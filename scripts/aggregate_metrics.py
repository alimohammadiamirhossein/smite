import os
import json
import pandas as pd
from natsort import natsorted

def read_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

main_path = "/localhome/aaa324/Project/FLATTEN/SMiTe/SMITE/vis/video_face_5"
save_per_part = False
data_rows = []
for file in natsorted(os.listdir(main_path)):
    file_path = os.path.join(main_path, file)
    if os.path.isdir(file_path):
        file_path = os.path.join(file_path, "metrics.json")
    else:
        continue
    if os.path.exists(file_path):
        data = read_json(file_path)
        data['file'] = file
        data['shared_frames'] = len(data['shared_frames'])
        if save_per_part:
            for key in data['iou_per_part']:
                data[key] = data['iou_per_part'][key]
        del data['iou_per_part']
        data_rows.append(data)
df = pd.DataFrame(data_rows)
df.set_index('file', inplace=True)

mean_row = df.mean(numeric_only=True)
mean_row.name = 'mean'  # Set the index for the mean row
df_with_mean = pd.concat([df, mean_row.to_frame().T])

def custom_format(x):
    if isinstance(x, (int, float)):  # Check if value is numeric
        return f"{x:.2f}" if x > 1 else f"{x:.4f}"
    return x

df_with_mean = df_with_mean.map(custom_format)

df_with_mean.to_csv(os.path.join(main_path, "total_metrics.csv"), mode='w')
