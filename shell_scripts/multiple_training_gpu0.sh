#!/bin/sh

./darknet detector train datasets/Haj_cropped_clean/obj.data datasets/Haj_cropped_clean/configs/yolov4-leaky.cfg yolov4-leaky.weights -dont_show -map >> /home/paul/yolov4-leaky.log 2>&1

./darknet detector train datasets/Haj_cropped_clean/obj.data datasets/Haj_cropped_clean/configs/yolov4-sam-leaky.cfg yolov4-sam-leaky.weights -dont_show -map >> /home/paul/yolov4-sam-leaky.log 2>&1
