import os
from tensorflow.image import non_max_suppression
import argparse

"""
 Convert the lines of a file to a list
"""
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', nargs='+', required=True, type=str,
                        help='directories to the predictions for each window')
    args = parser.parse_args()
    first_dir = args.dirs.pop(0)
    all_files = os.listdir(first_dir)
    for file_name in all_files:
        with open('./predictions/' + file_name, 'w') as file:
            pass
        objects = []
        for dir in args.dirs:
            for other_file_name in os.listdir(dir):
                if file_name == other_file_name:
                    lines_list1 = file_lines_to_list(os.path.join(first_dir, file_name))
                    lines_list2 = file_lines_to_list(os.path.join(dir, other_file_name))
                    lines = lines_list1 + lines_list2
                    for line in lines:
                        print(line)
                        class_name, score, left, top, right, bottom = line.split()
                        bbox = [int(left), int(top), int(right), int(bottom)]
                        obj = {"class": class_name, "bbox": bbox, "score": score}
                        objects.append(obj)
        boxes = [obj['bbox'] for obj in objects]

        scores = [float(obj['score']) for obj in objects]

        selected_indices = []
        if boxes:
            selected_indices = non_max_suppression(
                boxes, scores, 1000, 0.1, 0.5
            )
        selected_objects = []
        for index in selected_indices:
            selected_objects.append(objects[index])

        for obj in selected_objects:
            with open('./predictions/' + file_name, 'a') as file:
                file.write("%s %s %s %s %s %s\n" % (obj['class'], str(obj['score']), str(obj['bbox'][0]),
                                                    str(obj['bbox'][1]), str(obj['bbox'][2]),
                                                    str(obj['bbox'][3])))

if __name__ == "__main__":
    main()
