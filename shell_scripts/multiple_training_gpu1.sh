#!/bin/sh

#./darknet detector train datasets/Haj_cropped_clean/obj.data datasets/Haj_cropped_clean/configs/yolov4-sam-mish.cfg yolov4-sam-mish.weights -gpus 1 -dont_show -map >> /home/paul/yolov4-sam-mish.log 2>&1

#./darknet detector train datasets/Haj_cropped_clean/obj.data datasets/Haj_cropped_clean/configs/yolov4-sam-leaky.cfg yolov4-sam-leaky.weights -gpus 1 -dont_show -map >> /home/paul/yolov4-sam-leaky.log 2>&1

./darknet detector train datasets/Haj_cropped_clean/obj.data datasets/Haj_cropped_clean/configs/yolov4-mish.cfg yolov4-mish.weights -gpus 1 -dont_show -map >> /home/paul/yolov4-mish.log 2>&1