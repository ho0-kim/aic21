import os
import cv2
import argparse
import json
import math
import numpy as np

def check_and_create(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    return folder_path

def pos_on_ground(box):
    x, y, width, height = box
    pos = np.array([x+width/2, y+height, 1])
    return pos

def homography_matrix(homography):
    # Note) 10. "train(validation/test)/<subset>/<cam>/calibration.txt". They are the baseline manual calibration results. Each file shows the 3x3 homography matrix at the first line. If the correction of radial distortion is conducted, the 3x3 intrinsic parameter matrix and 1x4 distortion coefficients are also printed. Finally, the reprojection error in pixels is printed as well.
    homography = homography.split(':')[1]
    a,b,c = homography.split(';')
    a = a.strip()
    matrix = np.array([a.split(' '),
             b.split(' '),
             c.split(' ')], dtype = np.float32)
    return matrix

def get_diff_from_prev_val(val_list):
    diff = []
    prev = val_list[0]
    for it in val_list[1:]:
        val = (it[0] - prev[0], it[1] - prev[1])
        diff.append(val)
    return diff

def get_speed(vec_list):
    # 속력의 평균값을 구한다
    # +val: speed up, -val: speed down
    speed = []
    for vec in vec_list:
        speed.append(math.sqrt(vec[0]*vec[0]+vec[1]*vec[1]))
    speed = np.mean(speed)
    return speed

def get_turn(vec_list):
    # 초기속도와 x축의 각도를 구하고 이를 최종속도의 것과 비교한다
    # -90: turn right, +90: turn left
    init_vec = vec_list[0]
    final_vec = vec_list[-1]
    init_angle = math.degrees(math.atan(init_vec[1]/init_vec[0]))
    final_angle = math.degrees(math.atan(final_vec[1]/final_vec[0]))
    return final_angle - init_angle

def main(args):
    track3_folder = args.track3_root
    with open(f"data/{args.dataset}-tracks.json",'r') as file:
        data = json.load(file)
    result = {}
    
    track_list = data.keys()
    for track_id in track_list:
        frames = data[track_id]["frames"]
        boxes = data[track_id]["boxes"]
        [a, b, c, *rest] = frames[0].split('/')
        path_calibration = os.path.join(track3_folder, a, b, c, 'calibration.txt')
        with open(path_calibration,'r') as file:
            homography = file.readline()
            h_matrix = homography_matrix(homography)

        ## h_matrix check - image warp test
        #image = cv2.imread(frames[0])
        #image = image.astype(dtype=np.float32)
        #warp_image = cv2.warpPerspective(image, h_matrix, (1600, 1200))
        #cv2.imwrite('warp.jpg',warp_image)

        pos_list = []

        for box in boxes:
            pos = pos_on_ground(box)
            #pos = h_matrix.dot(pos.transpose())
            pos_list.append((pos[0],pos[1]))

        vec_list = get_diff_from_prev_val(pos_list)
        if len(vec_list) == 0:
            speed, turn = 0, 0
        else:
            speed = get_speed(vec_list)
            turn = get_turn(vec_list)

        motion = {'speed':speed, 'turn':turn, 'positions':pos_list, 'vectors':vec_list}
        result[track_id] = motion

    with open(f'data/{args.dataset}-motions.json', 'w') as file:
        json.dump(result, file, indent = 2, ensure_ascii = False)

if __name__ == '__main__':
    print("Loading parameters...")
    parser = argparse.ArgumentParser(description='Calculate motion vector from bbox')
    parser.add_argument('--data_root', dest='data_root', default='./',
                        help='dataset root path')
    parser.add_argument('--track3_root', dest='track3_root', default='track3',
                        help='root path of track3 data')
    parser.add_argument('--dataset', dest='dataset', default='train',
                        help='train or test')

    args = parser.parse_args()

    main(args)
