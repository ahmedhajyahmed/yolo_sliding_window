# -*- coding: utf-8 -*-
"""extract bbox coordinates from xml annotations

Usage:
    python get_bboc_from_xml --xml [xml_path]

Created on Wed Aug 12 17:22:21 2020

Author:
    Ahmed Haj Yahmed
"""
import os
import cv2
import xml.etree.ElementTree as ET
import csv
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', help='xml file containing annotations')
    
    args = vars(parser.parse_args())
    
    tree = ET.parse(args['xml'])
    root = tree.getroot()
    img_path = root[2].text
    img_path = img_path.replace('\\','/')
    
    
    if not(os.path.isdir('./results')):
        os.mkdir('./results')
    base = os.path.basename(img_path)
    file_name = os.path.splitext(base)[0]
    
    with open('./results/out_' + file_name + '.csv', 'w', newline='') as file:
        writer = csv.writer(file, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
        writer.writerow(["bounding_box", "class"])
    img = cv2.imread(img_path)
    
    
    for item in root.findall("./object"):
        class_name = item[0].text
        x1, y1, x2, y2 = [el.text for el in item[4]] 
        print(x1, y1, x2, y2)
        with open('./results/out_' + file_name + '.csv', 'a', newline='') as file:
            writer = csv.writer(file, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
            writer.writerow(["["+x1+","+y1+","+x2+","+y2+"]", class_name])
        cv2.putText(img, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 50, 155), 2)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (140, 50, 155), 2)
        cv2.imwrite('./results/out_'+ base, img)
    
    read_file = pd.read_csv('./results/out_' + file_name + '.csv')
    read_file.to_excel('./results/out_' + file_name + '.xlsx', index=None, header=True)
    os.remove('./results/out_' + file_name + '.csv')
    
if __name__ == '__main__':
    main()
