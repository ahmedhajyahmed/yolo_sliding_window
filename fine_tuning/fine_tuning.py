from datetime import time

from object_detection.yolov4.yolov4 import YoloV4
import tensorflow as tf
from tensorflow.image import non_max_suppression
import os
import cv2
import argparse
import csv
import pandas as pd
import time


def put_boxes(img, objects, output_path):
    for obj in objects:
        x1, y1, x2, y2 = obj['real_bbox']
        classe = obj['class']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, classe, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(output_path, img)


def main():
    parser = argparse.ArgumentParser('Yolo Sliding Window')
    parser.add_argument('--type',
                        type=str,
                        required=True,
                        choices=('min', 'mid', 'max'),
                        help='parameters type.')
    # parser.add_argument('--step',
    #                     type=int,
    #                     default=500,
    #                     help='step of the sliding window.')
    args = parser.parse_args()
    min_steps = [50, 100, 200]
    min_rects = [[50, 50],
                 [50, 100],
                 [100, 200]]
    mid_steps = [300, 400, 500]
    mid_rects = [[200, 300],
                 [300, 300],
                 [300, 400]]
    max_steps = [600, 700, 800]
    max_rects = [[400, 500],
                 [500, 500],
                 [600, 600]]

    if args.type == 'min':
        rects = min_rects
        steps = min_steps
    elif args.type == 'mid':
        rects = mid_rects
        steps = mid_steps
    else:
        rects = max_rects
        steps = max_steps
    model = YoloV4()
    with open('./results_fine_tuning/' + args.type + '.csv', 'w', newline='') as file:
        writer = csv.writer(file, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
        writer.writerow(["test_name", "test_description", "time", "nb crop ", "obj_detected_before_NMS",
                         "obj_detected_after_NMS"])
    all_images = os.listdir('./test_images')
    for image in all_images:

        img = cv2.imread('./test_images/' + image)
        for step in steps:
            for rect in rects:
                start = time.time()
                objects = []
                i = 0
                nb_y_steps = 0
                for y in range(0, img.shape[0], step):
                    nb_x_steps = 0
                    for x in range(0, img.shape[1], step):
                        detect_rect_h = rect[0]
                        detect_rect_w = rect[1]
                        # Crop image
                        img_to_detect = img[y:y + detect_rect_h, x:x + detect_rect_w]
                        out = model.predict(img_to_detect, './results_fine_tuning',
                                            save_img=False, save_dataframe=False,
                                            image_name=image + str(i) + '_step' + str(step) + '_wind[' + str(
                                                rect[0]) + ',' + str(rect[1]) + ']')
                        i = i + 1
                        if not out.empty:
                            for bbox, classe, score in zip(out['coor'], out['class'], out['probability']):
                                if classe in ['person','car','bus','truck']:
                                    real_bbox = [bbox[0] + step * nb_x_steps, bbox[1] + step * nb_y_steps,
                                                 bbox[2] + step * nb_x_steps, bbox[3] + step * nb_y_steps]
                                    obj = {'real_bbox': real_bbox,
                                           'class': classe,
                                           'score': score}
                                    objects.append(obj)
                        nb_x_steps += 1
                    nb_y_steps += 1

                # if args.verbose == 2:
                #     put_boxes(img.copy(), objects, os.path.join(args.results_dir, 'out_sliding_window.jpg'))

                boxes = [obj['real_bbox'] for obj in objects]
                nb_obj_before_NMS = len(boxes)
                scores = [obj['score'] for obj in objects]

                selected_indices = non_max_suppression(
                    boxes, scores, 1000, 0.1, 0.5
                )

                selected_boxes = tf.gather(boxes, selected_indices)
                # print(selected_indices)
                # print(selected_boxes)
                nb_obj_after_NMS = len(selected_boxes)

                selected_objects = []
                for index in selected_indices:
                    selected_objects.append(objects[index])

                # if args.verbose == 1 or args.verbose == 2:
                #     print(selected_objects)
                test_name = image + "_step" + str(step) + '_window[' + str(rect[0]) + ',' + str(rect[1]) + ']'
                put_boxes(img.copy(), selected_objects, os.path.join("results_fine_tuning", test_name + '.jpg'))
                test_description = f"test with step : {str(step)} and window [{str(rect[0])},{str(rect[1])}]"
                end_time = time.time() - start
                with open('./results_fine_tuning/' + args.type + '.csv', 'a', newline='') as file:
                    writer = csv.writer(file, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
                    writer.writerow([test_name, test_description, end_time, i, nb_obj_before_NMS, nb_obj_after_NMS])
    read_file = pd.read_csv('./results_fine_tuning/' + args.type + '.csv')
    read_file.to_excel('./results_fine_tuning/' + args.type + '.xlsx', index=None, header=True)
    os.remove('./results_fine_tuning/' + args.type + '.csv')


if __name__ == '__main__':
    main()
