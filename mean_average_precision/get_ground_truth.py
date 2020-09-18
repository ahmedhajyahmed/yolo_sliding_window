"""

Adapted from:
    https://github.com/Cartucho/mAP/blob/master/scripts/extra/convert_gt_xml.py

"""
import os
import xml.etree.ElementTree as ET
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        type=str,
                        required=True,
                        help='path to the xml files directory')
    args = parser.parse_args()
    xml_list = os.listdir(args.dir)
    for tmp_file in xml_list:
        # 1. create new file (VOC format)
        with open("./ground_truth/" + tmp_file.replace(".xml", ".txt"), "a") as new_f:
            root = ET.parse(os.path.join(args.dir, tmp_file)).getroot()
            for obj in root.findall('object'):
                obj_name = obj.find('name').text
                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text
                new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    print("Conversion completed!")


if __name__ == '__main__':
    main()
