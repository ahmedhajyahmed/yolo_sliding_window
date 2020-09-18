"""
Usage : 
   python yolo_sliding_window_my_version_repo.py --image_dir /home/paul/sliding_window_fine_tuning/test_images/
    --step 600 --height 800 --width 800 --results_dir  yolov4-tiny_custom_2

"""
from object_detection.yolov4.yolov4 import YoloV4
import tensorflow as tf
from tensorflow.image import non_max_suppression
import os
import cv2
import argparse
import pandas as pd
import time



def get_args():
    parser = argparse.ArgumentParser('Yolo Sliding Window')
    parser.add_argument('--image_dir',
                        type=str,
                        required=True,
                        help='input image dir path.')
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
    parser.add_argument('--max_output_size',
                        type=int,
                        default=1000,
                        help='the maximum number of boxes to be selected by non-max suppression.')
    parser.add_argument('--iou_threshold',
                        type=float,
                        default=0.01,
                        help='the threshold for deciding whether boxes overlap too much with respect to IOU.')
    parser.add_argument('--score_threshold',
                        type=float,
                        default=0.7,
                        help='the threshold for deciding when to remove boxes based on score.')
    parser.add_argument('--results_dir',
                        type=str,
                        default='./results',
                        help='output directory path.')
    parser.add_argument('--verbose',
                        type=int,
                        choices=(0, 1, 2),
                        default=2,
                        help='verbose = 0 : no logs + not showing intermediate images and crops\n'
                             'verbose = 1 : logs + not showing intermediate images and crops\n'
                             'verbose = 2 : logs + showing intermediate images and crops.')

    args = parser.parse_args()
    return args


def put_boxes(img, objects, output_path):
    for obj in objects:
        x1, y1, x2, y2 = obj['real_bbox']
        classe = obj['class']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, classe, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(output_path, img)


def main(args):
    start = time.time()
    #model = YoloV4()#load_h5 = True, h5_file='weights/yolov4.h5')#tiny=True,NUM_CLASS=2)
    # !!!!!!!!!!!! change this !!!!!!!!!!
    model = YoloV4(NUM_CLASS=2, load_h5 = True, h5_file='../weights/yolov4-custom_1.h5')
    # model = YoloV4(tiny=True,NUM_CLASS=2, load_h5 = True, h5_file='../weights/yolov4-tiny_custom_2.h5')
    all_images = os.listdir(args.image_dir)
    for image in all_images:
      img = cv2.imread(os.path.join(args.image_dir, image))
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
              if args.verbose == 1 or args.verbose == 2:
                  print('croped_image shape', img_to_detect.shape)
  
              if args.verbose == 0 or args.verbose == 1 or args.verbose == 2:
                  out = model.predict(img_to_detect, os.path.join(args.results_dir, 'crops'),
                                      save_img=False, save_dataframe=False, image_name=str(i))
              else:
                  out = model.predict(img_to_detect, os.path.join(args.results_dir, 'crops'), save_dataframe=False,
                                      image_name=str(i))
              i += 1
              # print('!!!!!!!!!!!!!!!! this is out !!!!!!!!!!!!!!!! \n', out)
              if not out.empty:
                  for bbox, classe, score in zip(out['coor'], out['class'], out['probability']):
                      if classe in ['person','car','bus','truck', 'vehicule']:
                          if args.verbose == 1 or args.verbose == 2:
                              print('box', bbox)
                              print('class', classe)
                              print('score', score)
                          real_bbox = [bbox[0] + args.step * nb_x_steps, bbox[1] + args.step * nb_y_steps,
                                       bbox[2] + args.step * nb_x_steps, bbox[3] + args.step * nb_y_steps]
                          obj = {'real_bbox': real_bbox,
                                 'class': classe,
                                 'score': score}
                          objects.append(obj)
                      # if classe == 'person':
                      # box = {'y0': bbox[0], 'x0': bbox[0], 'y1': bbox[0], 'x1': bbox[0]}
                      # boxes = boxes.append(box,ignore_index=True)
                      # print(boxes)
              nb_x_steps += 1
          nb_y_steps += 1
  
      all_files = os.listdir(os.path.join(args.results_dir, 'crops'))
      if args.verbose == 1 or args.verbose == 2:
          print('the model created {} crops'.format(len(all_files)))
  
      if args.verbose == 2:
          image_name = image
          # image_name = os.path.basename(args.image)
          image_name = image_name.split('.')[0].replace(' ','_')
          put_boxes(img.copy(), objects, os.path.join(args.results_dir, 'out_sliding_window'+image_name+'.jpg'))
  
      boxes = [obj['real_bbox'] for obj in objects]
      if args.verbose == 1 or args.verbose == 2:
          print('the number of object detected before non max suppression is : ', len(boxes))
      scores = [obj['score'] for obj in objects]
      # print(objects)
      print(boxes)
      #  print(scores)
  
      selected_indices = []
      selected_boxes = []
      if boxes:
        selected_indices = non_max_suppression(
            boxes, scores, args.max_output_size, args.iou_threshold, args.score_threshold
        )
    
        selected_boxes = tf.gather(boxes, selected_indices)
      # print(selected_indices)
      # print(selected_boxes)
      if args.verbose == 1 or args.verbose == 2:
          print('the number of object detected after non max suppression is : ', len(selected_boxes))
  
      selected_objects = []
      for index in selected_indices:
          selected_objects.append(objects[index])
      
      print('Executed in ',time.time()-start)
      if args.verbose == 1 or args.verbose == 2:
          print(selected_objects)
      image_name = os.path.basename(image)
      image_name = image_name.split('.')[0].replace(' ','_')
      print('img_name', image_name)
      put_boxes(img.copy(), selected_objects, os.path.join(args.results_dir, 'out_non_max_suppression'+image_name+'.jpg'))
      
      output = pd.DataFrame()
      
      for obj in selected_objects:
          output = output.append(obj,ignore_index=True)
      
      output.to_excel(os.path.join(args.results_dir, 'out_test.xlsx'), index=None, header=True)


if __name__ == '__main__':
    main(get_args())
