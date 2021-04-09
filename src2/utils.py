import json
import os
import re
from stop_words import ENGLISH_STOP_WORDS as ESW
from car_colors import COLORS
from car_types import TYPES
from collections import Counter

import cv2
import torch
import torch.nn.functional as F

def tokenize(sentence):
    tokens = []
    sentence = sentence.lower()
    word = re.sub(r'[^a-zA-Z ]', '', sentence).split()

    for w in word:
        w = w.lower()
        #if w not in ESW:
        tokens.append(w)
    return tokens

def hasColors(tokens):
    colors = [item for sublist in COLORS for item in sublist]
    bHas = False
    color = None
    for t in tokens:
        if t in colors:
            bHas = True
            color = t
            break
    return bHas, color

def hasTypes(tokens):
    types = [item for sublist in TYPES for item in sublist]
    bHas = False
    type = None
    for i, t in enumerate(tokens):
        if t in types:
            bHas = True
            if t == 'truck' and tokens[i-1] == 'cargo': type = 'cargo truck'
            else: type = t
            break
    return bHas, type

def getColorLabel(tokens):
    bHas, color = hasColors(tokens)
    label = -1

    if bHas:
        for i, clr in enumerate(COLORS):
            if color in clr:
                label = i
                break
        assert (label < len(COLORS))

    return color, label

def getTypeLabel(tokens):
    bHas, type = hasTypes(tokens)
    label = -1

    if bHas:
        for i, mdl in enumerate(TYPES):
            if type in mdl:
                label = i
                break
        assert (label < len(TYPES))

    return type, label

def getTotalColorLabel():
    return len(COLORS)

def getTotalTypeLabel():
    return len(TYPES)

def getColorList(nls):
    colors = list()
    color_count = 0

    for s in nls:
        tokens = tokenize(s)
        color, color_label = getColorLabel(tokens)
        colors.append(color_label)

        if color_label >= 0:
            color_count += 1

    assert (color_count >= 0)
    return colors

def getTypeList(nls):
    types = list()
    type_count = 0

    for s in nls:
        tokens = tokenize(s)
        type, type_label = getTypeLabel(tokens)
        types.append(type_label)

        if type_label >= 0:
            type_count += 1

    assert (type_count >= 0)
    return types

def main():
    file_path = os.path.join('../../data/train-tracks.json')

    with open(file_path, 'rt') as f:
        json_data = json.load(f)

    nls = {}
    color_stat = [0] * len(COLORS)
    type_stat = [0] * len(TYPES)
    for k, v in json_data.items():
        nl = v['nl']
        nls.update({k: nl})
        no_color_count = 0
        no_type_count = 0
        for s in nl:
            tokens = tokenize(s)

            color, color_label = getColorLabel(tokens)
            color_stat[color_label] += 1

            if color_label < 0:
                no_color_count += 1
            ''''
            if color_label < 0:
                print(s)
            '''

            type, type_label = getTypeLabel(tokens)
            type_stat[type_label] += 1
            if type_label < 0:
                no_type_count += 1
                print(s)

            #print(tokens)
        # if no_color_count == len(nl):
        #     print(nl)
        if no_type_count == len(nl):
            print(nl)
    #print(nls)
    #print(color_stat)
    print(type_stat)

keyword_right = [
    'turn right',
    'turns right',
    'turning right',
    'right turn',
    'takes a right',
    'took a right'
]

keyword_left = [
    'turn left',
    'turns left',
    'turning left',
    'left turn',
    'turned left',
    'takes a left'
]

keyword_up = [
    'speed up',
    'speeds up',
]

keyword_down = [
    'slow',
    'slows',
    'slowly'
    'speed down',
    'speeds down',
]

keyword_stop = [ 
    'wait',
    'stops',
    'stopped',
    'waiting'
]

def is_right_turn(nl):
    for keyword in keyword_right:
        if nl.lower().find(keyword) != -1:
            return True
    return False

def is_left_turn(nl):
    for keyword in keyword_left:
        if nl.lower().find(keyword) != -1:
            return True
    return False

def is_speed_up(nl):
    for keyword in keyword_up:
        if nl.lower().find(keyword) != -1:
            return True
    return False

def is_speed_down(nl):
    for keyword in keyword_down:
        if nl.lower().find(keyword) != -1:
            return True
    return False

def is_stop_motion(nl):
    for keyword in keyword_stop:
        if nl.lower().find(keyword) != -1:
            return True
    return False

def motion_detection(nls):
    right, left, up, down, stop = 0, 0, 0, 0, 0
    for nl in nls:
        right += 1 if is_right_turn(nl) else 0
        left += 1 if is_left_turn(nl) else 0
        up += 1 if is_speed_up(nl) else 0
        down += 1 if is_speed_down(nl) else 0
        stop += 1 if is_stop_motion(nl) else 0
    return [right, left, up, down, stop]

def motion_calculation(track_id, threshold):
    if os.path.exists('data/test-motions.json'):
        with open("data/test-motions.json", 'r') as file:
            motion_data = json.load(file)
    else:
        with open("../data/test-motions.json", 'r') as file:
            motion_data = json.load(file)

    turn = motion_data[track_id]['turn']
    down = 1 if motion_data[track_id]['is_down'] else 0
    stop = 1 if motion_data[track_id]['is_stop'] else 0
    right = 1 if turn > threshold else 0 # alternative value: 20
    left = 1 if turn < -threshold else 0

    return [right, left, 0, down, stop]

class Vicinity:
    def __init__(self, json_path, cfg):
        self.data_cfg = cfg["data"]
        self.keyword_rear = [
            'followed by',
            'in front of',
            'behind by'
        ]

        self.keyword_front = [
            'behind',
            'following',
            'follows',
            'after'
        ]
        with open(json_path, "r") as f:
            self.vicinity_json = json.load(f)

    def _has_rear_car(self, nl):
        for keyword in self.keyword_rear:
            if nl.lower().find(keyword) != -1:
                return True
        return False

    def _has_front_car(self, nl):
        for keyword in self.keyword_front:
            if nl.lower().find(keyword) != -1:
                return True
        return False

    def _has_color(self, nl):
        for color_label in range(len(COLORS)):
            for color_word in COLORS[color_label]:
                if nl.lower().find(color_word) != -1:
                    return color_label
        return -1

    def _has_type(self, nl):
        for type_label in range(len(TYPES)):
            for type_word in TYPES[type_label]:
                if nl.lower().find(type_word) != -1:
                    return type_label
        return -1
    
    def calculation(self, track_id, nls, model_color, model_type):
        score = [0., 0., 0., 0.]    # score [rear color, rear type, front color, front type]
        count = [0, 0, 0, 0]        # count [rear color, rear type, front color, front type]
        for nl in nls:
            if self._has_rear_car(nl):
                color_label = self._has_color(nl[len(nl) // 2:])
                type_label = self._has_type(nl[len(nl) // 2:])
                count[0] += 1 if color_label > 0 else 0
                count[1] += 1 if type_label > 0 else 0
                if self.vicinity_json[track_id]["rear"] == 1:
                    if color_label > 0 or type_label > 0:
                        frame = cv2.imread(self.vicinity_json[track_id]["rear_frame"])
                        box = self.vicinity_json[track_id]["rear_bbox"]
                        crop = frame[int(box[1]):int(box[1] + box[3]), int(box[0]): int(box[0] + box[2]), :]
                        crop = cv2.resize(crop, dsize=tuple(self.data_cfg["crop_size"]))
                        crop = torch.from_numpy(crop).permute([2, 0, 1]).unsqueeze_(dim=0).to(dtype=torch.float32).cuda()

                        if color_label > 0:
                            t = model_color.forward(crop)[0]
                            score[0] += F.softmax(t, dim=0)[color_label]
                        if type_label > 0:
                            t = model_type.forward(crop)[0]
                            score[1] += F.softmax(t, dim=0)[type_label]

            elif self._has_front_car(nl):
                color_label = self._has_color(nl[len(nl) // 2:])
                type_label = self._has_type(nl[len(nl) // 2:])
                count[2] += 1 if color_label > 0 else 0
                count[3] += 1 if type_label > 0 else 0
                if self.vicinity_json[track_id]["front"] == 1:
                    if color_label > 0 or type_label > 0:
                        frame = cv2.imread(self.vicinity_json[track_id]["front_frame"])
                        box = self.vicinity_json[track_id]["front_bbox"]
                        crop = frame[int(box[1]):int(box[1] + box[3]), int(box[0]): int(box[0] + box[2]), :]
                        crop = cv2.resize(crop, dsize=tuple(self.data_cfg["crop_size"]))
                        crop = torch.from_numpy(crop).permute([2, 0, 1]).unsqueeze_(dim=0).to(dtype=torch.float32).cuda()

                        if color_label > 0:
                            t = model_color.forward(crop)[0]
                            score[2] += F.softmax(t, dim=0)[color_label]
                        if type_label > 0:
                            t = model_type.forward(crop)[0]
                            score[3] += F.softmax(t, dim=0)[type_label]
        return score, count

class RightLeftLane:
    def __init__(self, json_path):
        # self.keyword_leftlane = [
        #     'left lane'
        # ]
        # self.keyword_rightlane = [
        #     'right lane'
        # ]
        with open(json_path, "r") as f:
            self._2d_direction = json.load(f)

    def _on_left_lane(self, nl):
        e = nl.lower().find('left lane')
        s = 0 if e < 10 else e - 10
        if e != -1:
            if 'on' in nl.lower()[s:e] or 'in' in nl.lower()[s:e]:
                return True
        return False
    
    def _on_right_lane(self, nl):
        e = nl.lower().find('right lane')
        s = 0 if e < 10 else e - 10
        if e != -1:
            if 'on' in nl.lower()[s:e] or 'in' in nl.lower()[s:e]:
                return True
        return False
    
    def calculation(self, track_id, nls):
        score = [0., 0.]    # score [left lane, right lane]
        count = [0, 0]      # count [left lane, right lane]
        for nl in nls:
            if self._on_left_lane(nl):
                count[0] += 1
                score[0] += 1 if self._2d_direction[track_id] == "down" else 0
            elif self._on_right_lane(nl):
                count[1] += 1
                score[1] += 1 if self._2d_direction[track_id] == "up" else 0
        return score, count

class TrafficLight:
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.light = json.load(f)
    
    def has_trafficlight(self, nl):
        s = nl.lower()[len(nl)//2:].rfind('light')
        if s != -1:
            e = s + 10 if len(nl[s:]) > 10 else s + len(nl[s:])
            bHas, _ = hasColors(tokenize(nl[s:e]))
            if not bHas:
                return True
        return False

    def calculation(self, track_id, nls):
        score = 0
        count = 0
        for nl in nls:
            if self.has_trafficlight(nl):
                count += 1
                try:
                    score += 1 if self.light[track_id] else 0
                except KeyError:
                    continue
        return [score], [count]




if __name__ == '__main__':
    main()

