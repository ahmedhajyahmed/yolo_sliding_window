# -*- coding: utf-8 -*-

import os
import argparse

parser= argparse.ArgumentParser()
parser.add_argument('--img_dir', required=True)
parser.add_argument('--xml_dir', required=True)

args = parser.parse_args()


all_annotations = os.listdir(args.xml_dir)
print(len(all_annotations))
all_iamges = os.listdir(args.img_dir)
for annotation in all_annotations:
    flag = 0
    annotation_base_name = annotation.split('.')[0]
    for image in all_iamges:
        image_base_name = image.split('.')[0]
        if image_base_name == annotation_base_name:
            flag = 1
    if flag == 0:
        os.remove(os.path.join(args.xml_dir, annotation))


all_annotations = os.listdir(args.xml_dir)
print(len(all_annotations))       