#!/bin/sh

./darknet detector train datasets/Haj_cropped_clean/obj.data datasets/Haj_cropped_clean/configs/yolov4-custom.cfg yolov4.conv.137 -dont_show -map >> /home/paul/yolov4-custom.log 2>&1

./darknet detector train datasets/Haj_cropped_clean/obj.data datasets/Haj_cropped_clean/configs/yolov4-custom_1.cfg yolov4.conv.137 -dont_show -map >> /home/paul/yolov4-custom_1.log 2>&1

./darknet detector train datasets/Haj_cropped_clean/obj.data datasets/Haj_cropped_clean/configs/yolov4-custom_2.cfg yolov4.conv.137 -dont_show -map >> /home/paul/yolov4-custom_2.log 2>&1


