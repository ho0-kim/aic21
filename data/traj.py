import os
import re
import json

import cv2
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

    def _pt_on_3d(self, h, x, y):
        p = np.array([[x],[y],[1]])
        h = np.linalg.inv(h)
        q = np.dot(h, p).squeeze()

        return np.array([q[0]/q[2], q[1]/q[2]]) 

    def _get_traj(self, uuid):
        bbox_list = self.tracks[uuid]["boxes"]
        calib_file = self.tracks[uuid]["frames"][0]
        calib_file = os.path.dirname(os.path.dirname(calib_file))
        calib_file = os.path.join(calib_file, "calibration.txt")
        H = self._read_calib(calib_file)
        # _, inv_H = cv2.invert(H)
        _3dpt_list = []
        _2dpt_list = []
        for _i, (x, y, w, h) in enumerate(bbox_list):
            if _i == 0:
                _3dpt_list.append(np.array([0., 0.]))
                s_3dpt = self._pt_on_3d(H, x + w/2, y)
            else:
                _3dpt_list.append((self._pt_on_3d(H, x + w/2, y) - s_3dpt) * 110000.)
            _2dpt_list.append([x + w/2, y + h])
        # print(_x_list[0])
        # plt.plot(_x_list, _y_list, 'r--')
        # plt.xlim([0, 1600])
        # plt.ylim([0, 1200])
        # plt.savefig('unproj.jpg')
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

        kf = KalmanFilter(transition_matrices=transition_matrix,
                        observation_matrices=observation_matrix,
                        initial_state_mean=initial_state_mean)
        kf = kf.em(_3dpts, n_iter=1)
        (smoothed_state_means, smoothed_state_covariances) = kf.smooth(_3dpts)
        smoothed = []
        for i in range(len(smoothed_state_means)):
            smoothed.append([smoothed_state_means[i,0], smoothed_state_means[i,2]])
        return np.array(smoothed)

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

        # init_vec = vectors[0]
        # final_vec = vectors[-1]
        # init_angle = math.degrees(math.atan(init_vec[1]/init_vec[0]))
        # final_angle = math.degrees(math.atan(final_vec[1]/final_vec[0]))

        # print(f'GET ANGLE With Shawn\'s algorithm {final_angle - init_angle}')

        return np.arcsin(cross/(s_norm*e_norm))

if __name__ == '__main__':
    print(f'running script {__file__}')

    import matplotlib.pyplot as plt

    json_file = 'data/train-tracks.json'
    with open(json_file, 'r') as f:
        tracks = json.load(f)
    
    uuids = list(tracks.keys())

    mov = Movement(tracks)

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
