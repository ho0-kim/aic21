import os
import cv2
import argparse
import json
import math
import numpy as np

from pykalman import KalmanFilter
import simplejson
    
class PrettyFloat(float):
    def __repr__(self):
        return '%.15g' % self
    
def pretty_floats(obj):
    if isinstance(obj, float) or isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return PrettyFloat(obj)
    elif isinstance(obj, dict):
        return dict((k, pretty_floats(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return list(map(pretty_floats, obj))
    return obj

def check_and_create(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    return folder_path

def pos_on_ground(box):
    x, y, width, height = box
    pos = np.array([x+width/2, y+height, 1])
    return pos

def homography_matrix(homography):
    homography = homography.split(':')[1]
    a,b,c = homography.split(';')
    a = a.strip()
    matrix = np.array([a.split(' '),
             b.split(' '),
             c.split(' ')], dtype = np.float32)
    return matrix

def get_speed(vec_list):
    # 속력의 평균값을 구한다
    # +val: speed up, -val: speed down
    speed = []
    for vec in vec_list:
        speed.append(math.sqrt(vec[0]*vec[0]+vec[1]*vec[1]))
    return speed

def get_turn(vec_list):
    # 초기속도와 x축의 각도를 구하고 이를 최종속도의 것과 비교한다
    # -90: turn right, +90: turn left
    init_vec = vec_list[0]
    final_vec = vec_list[-1]
    init_angle = math.degrees(math.atan(init_vec[1]/init_vec[0]))
    final_angle = math.degrees(math.atan(final_vec[1]/final_vec[0]))
    return final_angle - init_angle

def get_rot_angle(vectors):
    # Hoyoung's way
    if len(vectors) > 9:    # works like average filter
        s = np.mean(vectors[:5], axis=0)
        e = np.mean(vectors[-5:], axis=0)
    elif len(vectors) > 6:
        s = np.mean(vectors[:3], axis=0)
        e = np.mean(vectors[-3:], axis=0)
    else:
        s = vectors[0]
        e = vectors[-1]
    cross = np.cross(s, e)
    s_norm = np.linalg.norm(s)
    e_norm = np.linalg.norm(e)

    # stop vector case
    if s_norm == 0.0 or e_norm == 0.0:
        return [0.0, s_norm, e_norm]

    return [np.arcsin(cross/(s_norm*e_norm)), s_norm, e_norm]

def _smoother(_3dpts):
    vx = _3dpts[1, 0] - _3dpts[0, 0]
    vy = _3dpts[1, 1] - _3dpts[0, 1]

    initial_state_mean = [_3dpts[0,0], vx, _3dpts[0,1], vy]
    transition_matrix = [[1,1,0,0],
                        [0,1,0,0],
                        [0,0,1,1],
                        [0,0,0,1]]
    observation_matrix = [[1,0,0,0],
                        [0,0,1,0]]

    kf = KalmanFilter(transition_matrices=transition_matrix,
                    observation_matrices=observation_matrix,
                    initial_state_mean=initial_state_mean)
    kf = kf.em(_3dpts, n_iter=1)
    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(_3dpts)
    smoothed = []
    for i in range(len(smoothed_state_means)):
        smoothed.append([smoothed_state_means[i,0], smoothed_state_means[i,2]])
    return np.array(smoothed)

def stop_motion(spd_list):
    #1초(10프레임) 동안 1이하의 속도일경우
    count = 0
    for spd in spd_list:
        if spd > 1.:
            count = 0
        elif count == 9:
            return True
        else:
            count += 1
    return False

def slow_motion(acc_list):
    #1초(10프레임) 동안 감속일경우
    count = 0
    prev = acc_list[0]
    for acc in acc_list:
        if acc > prev:
            count = 0
            prev = acc
        elif count == 9:
            return True
        else:
            count += 1
            prev = acc
    return False

def main(args):
    with open(f"data/{args.dataset}-tracks.json",'r') as file:
        data = json.load(file)
    result = {}
    
    track_list = list(data.keys())
    for track_id in track_list:
        frames = data[track_id]["frames"]
        boxes = data[track_id]["boxes"]
        try:
            nl = data[track_id]["nl"]
        except:
            nl = ''
        [a, b, c, *rest] = frames[0].split('/')
        path_calibration = os.path.join(a, b, c, 'calibration.txt')
        with open(path_calibration,'r') as file:
            homography = file.readline()
            h_matrix = homography_matrix(homography)
            h_matrix_inv = np.zeros((3,3), np.float32)
            cv2.invert(h_matrix, h_matrix_inv)

        pos_list = []

        for box in boxes:
            pos = pos_on_ground(box)
            pos = h_matrix_inv.dot(pos.transpose())
            pos_list.append((pos[0]/pos[2]*110000.,pos[1]/pos[2]*110000.))
        pos_list = np.array(pos_list)
        pos_list = _smoother(pos_list) if pos_list.shape[0] != 1 else pos_list

        vec_list = np.diff(pos_list, n=1, axis=0)
        if len(vec_list) <= 1:
            turn, s_norm, e_norm = 0, 0, 0
            is_stop, is_down = True, False
            speed_list = get_speed(vec_list)
            acc_list = [ .0 ]
        else:
            turn, s_norm, e_norm = get_rot_angle(vec_list)
            turn = turn*180/np.pi
            speed_list = get_speed(vec_list)
            is_stop = stop_motion(speed_list)
            acc_list = np.diff(speed_list, n=1, axis=0)
            is_down = slow_motion(acc_list)

        motion = {'turn':turn, 's_norm':s_norm, 'e_norm':e_norm, 'is_stop':is_stop, 'is_down':is_down, 'positions':pos_list.tolist(), 'speeds':speed_list, 'nl':nl}
        result[track_id] = motion

    with open(f'data/{args.dataset}-motions.json', 'w') as file:
        #json.dump(result, file, indent = 2, ensure_ascii = False)
        simplejson.dump(pretty_floats(result), file, indent = 2, ensure_ascii = False)

if __name__ == '__main__':
    print("Loading parameters...")
    parser = argparse.ArgumentParser(description='Calculate motion vector from bbox')
    parser.add_argument('--data_root', dest='data_root', default='./',
                        help='dataset root path')
    parser.add_argument('--dataset', dest='dataset', default='train',
                        help='train or test')

    args = parser.parse_args()

    main(args)
