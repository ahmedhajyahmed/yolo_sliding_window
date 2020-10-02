#!/bin/sh

touch /home/ahmed/test_ahmed.txt
eval "$(conda shell.bash hook)"
conda activate fine_tuning


python automated_pipline.py --dir "/home/ahmed/test_automated_pipline/test_images/" --yolo_type 'tiny_yolov4' --input_size 608 --h5_file 'yolov4-tiny_custom_2.h5' --step_interval 600 900 100 --height_interval 800 1000 50 --width_interval 800 1000 50 >> /home/ahmed/yolov4-tiny_custom_2.log 2>&1


python automated_pipline.py --dir "/home/ahmed/test_automated_pipline/test_images/" --yolo_type 'yolov4' --input_size 608 --h5_file 'yolov4_custom.h5' --step_interval 600 900 100 --height_interval 800 1000 50 --width_interval 800 1000 50 >> /home/ahmed/yolov4_custom.log 2>&1

python automated_pipline.py --dir "/home/ahmed/test_automated_pipline/test_images/" --yolo_type 'yolov4' --input_size 608 --h5_file 'yolov4_custom_1.h5' --step_interval 600 900 100 --height_interval 800 1000 50 --width_interval 800 1000 50 >> /home/ahmed/yolov4_custom_1.log 2>&1