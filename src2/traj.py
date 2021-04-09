import os
import re
import json
import csv

import pandas as pd
import numpy as np
import math

# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise

from pykalman import KalmanFilter

class Movement:
    def __init__(self, tracks):
        self.tracks = tracks

    def _read_calib(self, calib_file):
        with open(calib_file, 'r') as f:
            l = f.readline()
            float_nums = re.findall(r"[-+]?\d*\.\d+|\d+", l)
        return np.array([[ float(float_nums[i]) 
                        for i in range(j, j+3) ] 
                        for j in [0,3,6]])

    def _read_calib_2(self, uuid):
        calib_file = self.tracks[uuid]["frames"][0]
        calib_file = os.path.dirname(os.path.dirname(calib_file))
        calib_file = os.path.join(calib_file, "calibration.txt")
        return self._read_calib(calib_file)

    def _pt_on_3d(self, h, x, y):
        p = np.array([[x],[y],[1]])
        h = np.linalg.inv(h)
        q = np.dot(h, p).squeeze()

        return np.array([q[0]/q[2], q[1]/q[2]]) 

    def _pt_on_3d_2(self, h, x, y, s_3dpt):
        p = np.array([[x],[y],[1]])
        h = np.linalg.inv(h)
        q = np.dot(h, p).squeeze()

        return np.array([q[0]/q[2], q[1]/q[2]]) - s_3dpt

    def _get_traj(self, uuid):
        bbox_list = self.tracks[uuid]["boxes"]
        calib_file = self.tracks[uuid]["frames"][0]
        calib_file = os.path.dirname(os.path.dirname(calib_file))
        calib_file = os.path.join(calib_file, "calibration.txt")
        H = self._read_calib(calib_file)
        _3dpt_list = []
        _2dpt_list = []
        for _i, (x, y, w, h) in enumerate(bbox_list):
            if _i == 0:
                _3dpt_list.append(np.array([0., 0.]))
                s_3dpt = self._pt_on_3d(H, x + w/2, y)
            else:
                _3dpt_list.append((self._pt_on_3d(H, x + w/2, y) - s_3dpt) * 110000.)
            _2dpt_list.append([x + w/2, y + h])

        return np.array(_3dpt_list), np.array(_2dpt_list), s_3dpt # 3D Points, 2D points, starting point

    def _smoother(self, _3dpts):
        vx = _3dpts[1, 0] - _3dpts[0, 0]
        vy = _3dpts[1, 1] - _3dpts[0, 1]

        initial_state_mean = [_3dpts[0,0], vx, _3dpts[0,1], vy]
        transition_matrix = [[1,1,0,0],
                            [0,1,0,0],
                            [0,0,1,1],
                            [0,0,0,1]]
        observation_matrix = [[1,0,0,0],
                            [0,0,1,0]]

        # Kalman smoothing
        kf = KalmanFilter(transition_matrices=transition_matrix,
                        observation_matrices=observation_matrix,
                        initial_state_mean=initial_state_mean)
        kf = kf.em(_3dpts, n_iter=1)
        (smoothed_state_means, smoothed_state_covariances) = kf.smooth(_3dpts)
        smoothed = []
        for i in range(len(smoothed_state_means)):
            smoothed.append([smoothed_state_means[i,0], smoothed_state_means[i,2]])
        kf_smoothed = np.array(smoothed)

        # Moving Average smoothing
        smoothed = []
        for _ in range(2):  # padding with the same value
            kf_smoothed = np.insert(kf_smoothed, 0, kf_smoothed[0], axis=0)
            kf_smoothed = np.insert(kf_smoothed, -1, kf_smoothed[-1], axis=0)
        
        avg_x = np.convolve(kf_smoothed[:,0], np.ones(5)/5, mode='valid')
        avg_y = np.convolve(kf_smoothed[:,1], np.ones(5)/5, mode='valid')
        avg_smoothed = np.stack([avg_x, avg_y], axis=1)

        return avg_smoothed

    def _get_vectors(self, _3dpts):
        return np.diff(_3dpts, axis=0)

    def _get_rot_angle(self, vectors):
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

        return np.arcsin(cross/(s_norm*e_norm))

    def _read_det(self, uuid):
        det_file = self.tracks[uuid]["frames"][0]
        det_file = os.path.dirname(os.path.dirname(det_file))
        det_file = os.path.join(det_file, "det")
        det_file = os.path.join(det_file, "det_mask_rcnn.txt") #mask_rcnn, ssd512, yolo3

        return pd.read_csv(det_file, names=['frame', 'pass', 'left', 'top', 'width', 'height', 'conf', 'pass1', 'pass2', 'pass3'])

    def _select_frame(self, uuid, df):
        image_paths = self.tracks[uuid]["frames"]
        frame_idx = []
        for image_path in image_paths:
            frame_idx.append(int(os.path.basename(image_path).split('.')[0]))
        df = df[frame_idx[0] <= df['frame']]
        df = df[frame_idx[-1] >= df['frame']]
        return df
    
    def _get_multicar_pos(self, df, sp, H):   
        # get multiple cars' postion in multiple frames
        # input:
        #   df: pd.Dataframe
        #   sp: starting 3d points from _get_traj
        df['2dx'] = df['left'] + df['width']/2
        df['2dy'] = df['top'] + df['height']

        _3dpt_df = self._pt_on_3d(H, df['2dx'], df['2dy'])
        df['3dx'] = (_3dpt_df[0] - sp[0]) * 110000.
        df['3dy'] = (_3dpt_df[1] - sp[1]) * 110000.

        return df

    def _get_rear_car(self, df, _sm_3dpts, vec, _2dpts):
        bboxes = []
        candi_coords = []
        frame_ids = []

        initial_frame = df.frame[:1].item()

        for i in range(1, len(_sm_3dpts)):
            frame_id = initial_frame + i
            df_frame = df[df.frame == frame_id]
            cars_coord = df_frame[['3dx', '3dy']].to_numpy()
            dist_from_maincar = []
            candidates = []
            for j, car_coord in enumerate(cars_coord):
                dist = []
                for _3dpt in _sm_3dpts[:i+1]:
                    dist.append(np.linalg.norm(car_coord-_3dpt))
                if len(dist) < 2: continue
                min_index = dist.index(min(dist))
                if min_index == 0:
                    second_index = 1
                elif min_index == len(dist)-1:
                    second_index = min_index-1
                else:
                    if dist[min_index-1] < dist[min_index+1]:
                        second_index = min_index - 1
                    else:
                        second_index = min_index + 1
                a = dist[min_index]
                b = dist[second_index]
                c = np.linalg.norm(_sm_3dpts[min_index] - _sm_3dpts[second_index])
                s = (a + b + c) / 2
                dist_from_path = np.sqrt(s*(s-a)*(s-b)*(s-c)) * 2 / c
                if dist_from_path < 1:
                    d = np.linalg.norm(car_coord - _sm_3dpts[i])
                    if d < 3: continue
                    candidates.append(j)
                    dist_from_maincar.append(d)
            if len(candidates) < 1: continue
            
            if i == len(vec): vec_i = i-1
            else: vec_i = i
            if vec[vec_i, 1] == 0: vec[vec_i, 1] = 1e-15
            l = -vec[vec_i,0] / vec[vec_i, 1]
            new_candidates = []
            new_dist = []
            for d, candidate in zip(dist_from_maincar, candidates):
                candi_coord = cars_coord[candidate]
                y = l*(candi_coord[0]-_sm_3dpts[i,0]) + _sm_3dpts[i,1]
                if vec[vec_i, 1] > 0 and y > candi_coord[1]:
                    new_candidates.append(candidate)
                    new_dist.append(d)
                elif vec[vec_i, 1] < 0 and y < candi_coord[1]:
                    new_candidates.append(candidate)
                    new_dist.append(d)
            if len(new_candidates) < 1: continue
            candi_index = new_dist.index(min(new_dist))
            candidate = new_candidates[candi_index]
            candi_coord = cars_coord[candidate]
            bbox = df_frame[['left', 'top', 'width', 'height']].to_numpy()[candidate]

            bboxes.append(bbox)
            candi_coords.append(candi_coord)
            frame_ids.append(frame_id)

        car_id = 0
        cars = dict()
        for i in range(len(bboxes)-1):
            if i == 0:
                cars.update({car_id: []})
            if np.linalg.norm(candi_coords[i] - candi_coords[i+1]) > 3:
                car_id += 1
            if car_id not in cars:
                cars.update({car_id: []})
            cars[car_id].append([frame_ids[i], bboxes[i]])

        max_value = -1
        for key in cars:
            if len(cars[key]) > max_value and len(cars[key]) > len(_sm_3dpts)//20:
                max_value = len(cars[key])
                max_key = key

        if len(bboxes) == 0 or max_value == -1:
            return {'rear': 0, 'frame_num': 0, 'rear_bbox': [0, 0, 0, 0]}

        vec_2d = mov._get_vectors(mov._smoother(_2dpts))
        if vec_2d.mean(axis=0)[1] > 0:
            return {'rear': 1, 'frame_num': int(cars[max_key][0][0]), 'rear_bbox': cars[max_key][0][1].tolist()}
        else:
            return {'rear': 1, 'frame_num': int(cars[max_key][-1][0]), 'rear_bbox': cars[max_key][-1][1].tolist()}
    
    def rear_car(self, uuid):
        df = self._read_det(uuid)
        df = self._select_frame(uuid, df)
        _3dpts, _2dpts, s_3dpt = self._get_traj(uuid)
        H = self._read_calib_2(uuid)
        df = self._get_multicar_pos(df, s_3dpt, H)
        _sm_3dpts = self._smoother(_3dpts)
        vec = self._get_vectors(_sm_3dpts)
        ret = self._get_rear_car(df, _sm_3dpts, vec, _2dpts)

        frame_file = self.tracks[uuid]["frames"][0]
        frame_file = os.path.dirname(frame_file)
        frame_file = os.path.join(frame_file, f"{str(ret['frame_num']).zfill(6)}.jpg")

        ret.pop('frame_num', None)
        ret["rear_frame"] = frame_file
        # ret["rear_bbox"] = ret["rear_bbox"].tolist()
        return ret

    def _get_front_car(self, df, _sm_3dpts, vec, _2dpts):
        bboxes = []
        candi_coords = []
        frame_ids = []

        initial_frame = df.frame[:1].item()

        for i in range(len(_sm_3dpts)):
            frame_id = initial_frame + i
            df_frame = df[df.frame == frame_id]
            cars_coord = df_frame[['3dx', '3dy']].to_numpy()
            dist_from_maincar = []
            candidates = []
            for j, car_coord in enumerate(cars_coord):
                dist = []
                for _3dpt in _sm_3dpts[i:]:
                    dist.append(np.linalg.norm(car_coord-_3dpt))
                if len(dist) < 2: continue
                min_index = dist.index(min(dist))
                if min_index == 0:
                    second_index = 1
                elif min_index == len(dist)-1:
                    second_index = min_index-1
                else:
                    if dist[min_index-1] < dist[min_index+1]:
                        second_index = min_index - 1
                    else:
                        second_index = min_index + 1
                a = dist[min_index]
                b = dist[second_index]
                c = np.linalg.norm(_sm_3dpts[i + min_index] - _sm_3dpts[i + second_index])
                s = (a + b + c) / 2
                dist_from_path = np.sqrt(s*(s-a)*(s-b)*(s-c)) * 2 / c
                if dist_from_path < 1:
                    d = np.linalg.norm(car_coord - _sm_3dpts[i])
                    if d < 3: continue
                    candidates.append(j)
                    dist_from_maincar.append(d)
            if len(candidates) < 1: continue
            
            if i == len(vec): vec_i = i-1
            else: vec_i = i
            if vec[vec_i, 1] == 0: vec[vec_i, 1] = 1e-15
            l = -vec[vec_i,0] / vec[vec_i, 1]
            new_candidates = []
            new_dist = []
            for d, candidate in zip(dist_from_maincar, candidates):
                candi_coord = cars_coord[candidate]
                y = l*(candi_coord[0]-_sm_3dpts[i,0]) + _sm_3dpts[i,1]
                if vec[vec_i, 1] > 0 and y < candi_coord[1]:
                    new_candidates.append(candidate)
                    new_dist.append(d)
                elif vec[vec_i, 1] < 0 and y > candi_coord[1]:
                    new_candidates.append(candidate)
                    new_dist.append(d)
            if len(new_candidates) < 1: continue
            candi_index = new_dist.index(min(new_dist))
            candidate = new_candidates[candi_index]
            candi_coord = cars_coord[candidate]
            bbox = df_frame[['left', 'top', 'width', 'height']].to_numpy()[candidate]

            bboxes.append(bbox)
            candi_coords.append(candi_coord)
            frame_ids.append(frame_id)

        car_id = 0
        cars = dict()
        for i in range(len(bboxes)-1):
            if i == 0:
                cars.update({car_id: []})
            if np.linalg.norm(candi_coords[i] - candi_coords[i+1]) > 3:
                car_id += 1
            if car_id not in cars:
                cars.update({car_id: []})
            cars[car_id].append([frame_ids[i], bboxes[i]])

        max_value = -1
        for key in cars:
            if len(cars[key]) > max_value and len(cars[key]) > len(_sm_3dpts)//20:
                max_value = len(cars[key])
                max_key = key

        if len(bboxes) == 0 or max_value == -1:
            return {'front': 0, 'frame_num': 0, 'front_bbox': [0, 0, 0, 0]}

        vec_2d = mov._get_vectors(mov._smoother(_2dpts))
        if vec_2d.mean(axis=0)[1] > 0:
            return {'front': 1, 'frame_num': int(cars[max_key][0][0]), 'front_bbox': cars[max_key][0][1].tolist()}
        else:
            return {'front': 1, 'frame_num': int(cars[max_key][-1][0]), 'front_bbox': cars[max_key][-1][1].tolist()}

    def front_car(self, uuid):
        df = self._read_det(uuid)
        df = self._select_frame(uuid, df)
        _3dpts, _2dpts, s_3dpt = self._get_traj(uuid)
        H = self._read_calib_2(uuid)
        df = self._get_multicar_pos(df, s_3dpt, H)
        _sm_3dpts = self._smoother(_3dpts)
        vec = self._get_vectors(_sm_3dpts)
        ret = self._get_front_car(df, _sm_3dpts, vec, _2dpts)

        frame_file = self.tracks[uuid]["frames"][0]
        frame_file = os.path.dirname(frame_file)
        frame_file = os.path.join(frame_file, f"{str(ret['frame_num']).zfill(6)}.jpg")

        ret.pop('frame_num', None)
        ret["front_frame"] = frame_file
        # ret["front_bbox"] = ret["front_bbox"].tolist()
        return ret


if __name__ == '__main__':
    print(f'running script {__file__}')

    import matplotlib.pyplot as plt

    # json_file = 'data/train-tracks.json'
    json_file = 'data/test-tracks.json'
    with open(json_file, 'r') as f:
        tracks = json.load(f)
    
    uuids = list(tracks.keys())

    mov = Movement(tracks)

    # uuids[0] = '97ead54f-042a-4497-a17f-514716553337'
    # uuids[0] = 'a9d0b0b6-038a-41b0-b682-b24b87042a6a'

    # right left lane
    # out_file = 'data/train-2d-dir.json'
    out_file = 'data/test-2d-dir.json'
    out = {}
    for i, uuid in enumerate(uuids):
        if len(tracks[uuid]["frames"]) > 1:
            _, _2dpts, _ = mov._get_traj(uuid)
            _sm_2dpts = mov._smoother(_2dpts)
            _2d_vec = mov._get_vectors(_sm_2dpts) 
            
            print(i, np.mean(_2d_vec[:,1]))
            # if vec[y] < 0, up (right lane) | if vec[y] >0, down (left lane)
            out[uuid] = "up" if np.mean(_2d_vec[:,1]) < 0 else "down"
        else:
            print(i, "num_frame: 1")
            out[uuid] = "none"
    
    json.dump(out, open(out_file, 'w'), indent=4)


    # front rear car
    """
    # 166 1055 1056 1057
    # out_file = 'data/train-vicinity.json'
    out_file = 'data/test-vicinity.json'
    out = {}
    for i, uuid in enumerate(uuids):
        print(i)
        if len(tracks[uuid]["frames"]) < 2:
            subout = {}
            subout.update({'rear': 0, 'rear_frame': '.', 'rear_bbox': [0, 0, 0, 0]})
            subout.update({'front': 0, 'front_frame': '.', 'front_bbox': [0, 0, 0, 0]})
        else:
            subout = {}
            subout.update(mov.rear_car(uuid))
            subout.update(mov.front_car(uuid))
        out[uuid] = subout
    
    json.dump(out, open(out_file, 'w'), indent=4)
    """

    # first test
    """
    _3dpt, _2dpt, _ = mov._get_traj(uuids[0])
    print(uuids[0])
    print('--- 3D Points ---')
    print(_3dpt)
    print('--- 2D Points ---')
    print(_2dpt)
    # _3dpt, _, _ = mov._get_traj("97ead54f-042a-4497-a17f-514716553337")
    
    print('Kalman Filtered points')
    print(mov._smoother(_3dpt))

    _sm_3dpt = mov._smoother(_3dpt)

    # plt.plot(_3dpt[:, 0], 'bo',
    #      _3dpt[:, 1], 'ro',
    #      _sm_3dpt[:, 0], 'b--',
    #      _sm_3dpt[:, 1], 'r--',)
    plt.plot(_3dpt[:, 0], _3dpt[:, 1], 'b--',
        _sm_3dpt[:, 0], _sm_3dpt[:, 1], 'r--')
    plt.savefig('traj.jpg')

    vectors = mov._get_vectors(_sm_3dpt)
    print('Vectors')
    print(vectors)

    ang = mov._get_rot_angle(vectors)
    print(ang*180/np.pi)

    print(uuids[0:10])
    for i in range(10):
        uuid = uuids[i]
        _3dpt, _, _ = mov._get_traj(uuid)
        _sm_3dpt = mov._smoother(_3dpt)
        vectors = mov._get_vectors(_sm_3dpt)
        ang = mov._get_rot_angle(vectors)
        print(ang*180/np.pi)

        # img = cv2.imread(tracks[uuid]["frames"][0])
        # h, w, _ = img.shape
        plt.clf()
        # plt.xlim([0, w])
        # plt.ylim([0, h])
        plt.plot(_3dpt[:, 0], _3dpt[:, 1], 'b--',
            _sm_3dpt[:, 0], _sm_3dpt[:, 1], 'r--')
        plt.savefig(f'traj_{i}.jpg')

    # vectors = mov._get_vectors(_2dpt)
    # ang = mov._get_rot_angle(vectors)
    # print(ang*180/np.pi)

    # _sm_2dpt = mov._smoother(_2dpt)
    # vectors = mov._get_vectors(_sm_2dpt)
    # ang = mov._get_rot_angle(vectors)
    # print(ang*180/np.pi)
    """


    # 2021-03-29
    """
    print("=== Test @ 21-03-30 ===")
    _3d_test = [[42.498980], [-90.686593], [1]]
    uuid = uuids[0]
    bbox_list = tracks[uuid]["boxes"]
    calib_file = tracks[uuid]["frames"][0]
    calib_file = os.path.dirname(os.path.dirname(calib_file))
    calib_file = os.path.join(calib_file, "calibration.txt")
    homo = mov._read_calib(calib_file)
    print(np.dot(homo, _3d_test))
    map_coord = np.float32([[42.499413, -90.694053],
                [42.499411, -90.693989],
                [42.499407, -90.694488],
                [42.499456, -90.694864],
                [42.499522, -90.695260],
                [42.499336, -90.693895],
                [42.499281, -90.693927]])
    img_coord = np.float32([[1255, 758],
                [1247, 865],
                [877, 518],
                [948, 502],
                [987, 491],
                [433, 644],
                [148, 470]])
    H, mask = cv2.findHomography(map_coord, img_coord, cv2.RANSAC, 100.)
    print(H)
    print(mask)
    inv_H = np.linalg.inv(H)
    _x = img_coord[0,0]
    _y = img_coord[0,1]
    res = np.dot(inv_H, [[_x],[_y], [1]])
    print(res)
    print(res[2, 0])
    scalar = res[2, 0]
    _x_world = res[0, 0] / scalar
    _y_world = res[1, 0] / scalar
    print(np.dot(inv_H, [[_x],[_y], [1]]))
    print(_x_world, _y_world)
    _3dpt_list = []
    _2dpt_list = []
    for x, y, w, h in bbox_list:
        _3dpt_list.append(mov._pt_on_3d(H, x + w/2, y))
        _2dpt_list.append([x + w/2, y + h])
    print('--- 3D Points ---')
    print(np.array(_3dpt_list))
    print('--- 2D Points ---')
    print(np.array(_2dpt_list))
    """