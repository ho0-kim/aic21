import os
import cv2
import argparse
import json
import numpy as np

def check_and_create(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    return folder_path

def main(args):
    with open(f"data/{args.dataset}-tracks.json",'r') as file:
        data = json.load(file)
    result_folder = os.path.join(args.data_root,"tracks",args.dataset)
    check_and_create(result_folder)

    track_list = data.keys()
    for track_id in track_list:
        frames = data[track_id]["frames"]
        boxes = data[track_id]["boxes"]
        track_folder = os.path.join(result_folder,track_id)
        check_and_create(track_folder)

        for frame, box in zip(frames,boxes):
            image = cv2.imread(frame)
            [x, y, width, height] = box
            box = [[x, y], [x+width, y], [x+width, y+height], [x, y+height]]
            pts = np.array(box, np.int32)
            pts = pts.reshape((-1, 1, 2))
            image = cv2.polylines(image, [pts], True, (255, 0, 0), 2)
            image = cv2.resize(image, (0,0), fx = 0.5, fy = 0.5)
            path_frame = os.path.join(track_folder,frame.replace('/','_'))
            cv2.imwrite(path_frame, image)

        try:
            nl = data[track_id]["nl"]
            with open(os.path.join(track_folder,"nl.txt"),"w") as f:
                f.write("\n".join(nl))
        except KeyError:
            pass

if __name__ == '__main__':
    print("Loading parameters...")
    parser = argparse.ArgumentParser(description='Extract video frames')
    parser.add_argument('--data_root', dest='data_root', default='./',
                        help='dataset root path')
    parser.add_argument('--dataset', dest='dataset', default='train',
                        help='train or test')

    args = parser.parse_args()

    main(args)
