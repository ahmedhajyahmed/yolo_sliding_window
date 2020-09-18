from object_detection.yolov4.yolov4 import YoloV4
import tensorflow as tf
from tensorflow.image import non_max_suppression
import os
import cv2
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        type=str,
                        required=True,
                        help='path to the input images directory')
    parser.add_argument('--step',
                        type=int,
                        default=500,
                        help='step of the sliding window.')
    parser.add_argument('--height',
                        type=int,
                        default=500,
                        help='height of the sliding window.')
    parser.add_argument('--width',
                        type=int,
                        default=500,
                        help='width of the sliding window.')
    args = parser.parse_args()
    model = YoloV4()
    all_images = os.listdir(args.dir)
    for image in all_images:
        img = cv2.imread(os.path.join(args.dir, image))
        base = os.path.basename(image)
        file_name = os.path.splitext(base)[0]
        with open('./predictions/' + file_name + '.txt', 'a') as file:
            pass
        print(file_name)
        objects = []
        i = 0
        nb_y_steps = 0
        for y in range(0, img.shape[0], args.step):
            nb_x_steps = 0
            for x in range(0, img.shape[1], args.step):
                detect_rect_h = args.height
                detect_rect_w = args.width
                # Crop image
                img_to_detect = img[y:y + detect_rect_h, x:x + detect_rect_w]
                out = model.predict(img_to_detect, './results_fine_tuning',
                                    save_img=False, save_dataframe=False,
                                    image_name=image + str(i) + '_step' + str(args.step) + '_wind[' + str(
                                        args.height) + ',' + str(args.width) + ']')
                i = i + 1
                if not out.empty:
                    for bbox, classe, score in zip(out['coor'], out['class'], out['probability']):
                        real_bbox = [bbox[0] + args.step * nb_x_steps, bbox[1] + args.step * nb_y_steps,
                                     bbox[2] + args.step * nb_x_steps, bbox[3] + args.step * nb_y_steps]
                        if classe == 'person':
                            obj = {'real_bbox': real_bbox,
                                   'class': 'personne',
                                   'score': score}
                            objects.append(obj)
                        elif classe == 'car' or classe == 'bus' or classe == 'truck':
                            obj = {'real_bbox': real_bbox,
                                   'class': 'vehicule',
                                   'score': score}
                            objects.append(obj)
                nb_x_steps += 1
            nb_y_steps += 1

        boxes = [obj['real_bbox'] for obj in objects]

        scores = [obj['score'] for obj in objects]

        selected_indices = []
        if boxes:
            selected_indices = non_max_suppression(
                boxes, scores, 1000, 0.1, 0.5
            )
        selected_objects = []
        for index in selected_indices:
            selected_objects.append(objects[index])
        for obj in selected_objects:
            with open('./predictions/' + file_name + '.txt', 'a') as file:
                file.write("%s %s %s %s %s %s\n" % (obj['class'], str(obj['score']), str(obj['real_bbox'][0]),
                                                    str(obj['real_bbox'][1]), str(obj['real_bbox'][2]),
                                                    str(obj['real_bbox'][3])))


if __name__ == '__main__':
    main()
