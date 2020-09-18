import os

dr_path = "/home/ahmed/sliding_window_test/yolo_tiny_yolov4_input_size_608/predictions_step_500_wind(500,500)"
gt_path = "./ground_truth"

dr_files = os.listdir(dr_path)
gt_files = os.listdir(gt_path)
print(len(gt_files))

for dr_file in dr_files:
    flag = 0
    for gt_file in gt_files:
        if gt_file == dr_file:
            flag = 1
    if flag ==0:
        open(os.path.join(gt_path, dr_file), 'a')

gt_files = os.listdir(gt_path)
print(len(gt_files))